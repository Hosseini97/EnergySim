import jax
import jax.numpy as jnp
import equinox as eqx
import time

# --- Energysim Imports ---
from energysim.sim.simulator import JAXSimulator
from energysim.rl.vector_env import VectorizedEnergyEnv
from energysim.rl.helpers import extract_obs
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, SystemActions
)

# --- Custom Lab Imports ---
from build_my_house import create_2_room_house

# ==========================================
# 1. The Brain: Policy Architecture
# ==========================================
class PPOPolicy(eqx.Module):
    trunk: eqx.nn.MLP
    actor_mean: eqx.nn.Linear
    critic_head: eqx.nn.Linear
    action_log_std: jax.Array
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.trunk = eqx.nn.MLP(obs_dim, hidden_dim, hidden_dim, depth=2, key=k1)
        self.actor_mean = eqx.nn.Linear(hidden_dim, action_dim, key=k2)
        self.critic_head = eqx.nn.Linear(hidden_dim, 1, key=k3)
        self.action_log_std = jnp.zeros(action_dim)
        
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        features = jax.nn.relu(self.trunk(obs))
        mean = self.actor_mean(features)
        mean = jax.nn.tanh(mean) # Bounded to [-1, 1]
        value = self.critic_head(features)[0]
        return mean, self.action_log_std, value

def map_actions(norm_actions: jax.Array, n_envs: int, n_rooms: int) -> SystemActions:
    bat_w = norm_actions[:, 0] * 3000.0
    hp_w = (norm_actions[:, 1:1+n_rooms] + 1.0) * 1000.0 # Map [-1, 1] to [0, 2000]
    ac_w = jnp.zeros((n_envs, n_rooms))
    stor_w = jnp.zeros((n_envs, n_rooms))
    
    return SystemActions(
        battery_power_w=bat_w, heat_pump_power_w=hp_w,
        ac_power_w=ac_w, storage_discharge_w=stor_w
    )

def main():
    print("--- 🧠 Energysim PPO Agent Evaluation ---")
    
   # ==========================================
    # 2. Reconstruct the Physics Environment
    # ==========================================
    print("Loading Dataset and Physics Configs...")
    dt_seconds = 900 
    
    # EXACT match to train.py
    dataset = SimulationDataset(
        file_path="/home/emmanuel-gendy/Documents/ppo_jax/heeten_training_master.csv", 
        dt_seconds=dt_seconds
    )
    
    t_config = create_2_room_house()
    n_rooms = len(t_config.room_air_indices)
    room_indices = jnp.array(t_config.room_air_indices)
    
    r_config = RewardConfig(price_weight=1.0, comfort_weight=5.0)
    
    simulator = JAXSimulator(
        dt_seconds=dt_seconds, t_config=t_config, r_config=r_config,
        b_config=BatteryConfig(), hp_config=HeatPumpConfig(), ac_config=AirConditionerConfig()
    )
    
    num_envs = 2048
    env = VectorizedEnergyEnv(simulator=simulator, dataset=dataset, num_envs=num_envs)
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    
    # ==========================================
    # 3. Load the Trained Neural Network
    # ==========================================
    print("Awakening the Neural Network...")
    obs_dim = n_rooms + 5
    action_dim = 1 + n_rooms 
    
    # A. Create a dummy skeleton with the exact same architecture
    dummy_key = jax.random.PRNGKey(0)
    empty_policy = PPOPolicy(obs_dim, action_dim, hidden_dim=64, key=dummy_key)
    
    # B. Inject the trained weights from the .eqx file
    # (Ensure the path matches where your train.py actually saves the file)
    model_path = "/home/emmanuel-gendy/Documents/ppo_jax/jax_ppo_model.eqx" 
    trained_policy = eqx.tree_deserialise_leaves(model_path, empty_policy)
    
    # Vectorize the network and observation extractor
    vmap_policy = jax.vmap(trained_policy)
    vmap_extract_obs = jax.vmap(extract_obs, in_axes=(0, None, None))
    
    dataset_length = len(env.shared_exo_data.price)

    # ==========================================
    # 4. The JIT-Compiled Inference Loop
    # ==========================================
    @eqx.filter_jit
    def run_inference(init_state, policy):
        
        def step_fn(curr_state, _):
            # 1. Extract Exogenous Data
            t = curr_state.time_idx[0]
            exo_batch = jax.tree.map(lambda x: x[t], env.shared_exo_data)
            
            # 2. Agent Observation
            obs = vmap_extract_obs(curr_state.sim.state, exo_batch, room_indices)
            
            # 3. Deterministic Action Selection (NO NOISE)
            mean_action, _, _ = vmap_policy(obs)
            
            # 4. Map to Physical Watts
            phys_actions = map_actions(mean_action, num_envs, n_rooms)
            
            # 5. Step Environment
            next_state, rewards, done, info = env.step(curr_state, phys_actions)
            
            return next_state, rewards
        
        _, all_rewards = jax.lax.scan(
            step_fn, init_state, None, length=dataset_length
        )
        
        return all_rewards

    # ==========================================
    # 5. Execute and Measure (DIAGNOSTIC VERSION)
    # ==========================================
    print("Starting JAX Compilation and Inference...")
    
    # Run inference to get rewards AND the info dict (to see temperatures)
    # Note: We need to modify run_inference slightly to return 'info'
    @eqx.filter_jit
    def run_inference_with_info(init_state, policy):
        def step_fn(curr_state, _):
            t = curr_state.time_idx[0]
            exo_batch = jax.tree.map(lambda x: x[t], env.shared_exo_data)
            obs = vmap_extract_obs(curr_state.sim.state, exo_batch, room_indices)
            mean_action, _, _ = vmap_policy(obs)
            phys_actions = map_actions(mean_action, num_envs, n_rooms)
            next_state, rewards, done, info = env.step(curr_state, phys_actions)
            return next_state, (rewards, info['room_temps'])
        
        _, (all_rewards, all_temps) = jax.lax.scan(step_fn, init_state, None, length=960)
        return all_rewards, all_temps

    all_rewards, all_temps = run_inference_with_info(state, trained_policy)
    
    # 1. Check the Raw Reward (Is it ~ -4.0 or ~ -400.0?)
    avg_step_reward = jnp.mean(all_rewards)
    print(f"DEBUG: Average Step Reward from Env: {avg_step_reward:.4f}")

    # 2. Check the Temperatures (Is the house freezing?)
    avg_temp = jnp.mean(all_temps)
    print(f"DEBUG: Average House Temperature: {avg_temp:.2f}°C")

    # 3. Corrected Metric Calculation
    # REMOVE THE * 100.0 multiplier here!
    total_reward_per_env = jnp.sum(all_rewards, axis=0) 
    mean_daily_cost = -jnp.mean(total_reward_per_env) / (960 / (86400 / 900))
    
    print("-" * 40)
    print(f"💰 Baseline RBC Daily Cost:  € 370.07")
    print(f"🧠 PPO Agent Daily Cost:     € {mean_daily_cost:.2f}")
    print("-" * 40)

if __name__ == "__main__":
    main()