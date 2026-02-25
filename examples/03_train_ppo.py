import time
from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from energysim.sim.simulator import JAXSimulator
from energysim.rl.vector_env import VectorizedEnergyEnv
from energysim.rl.helpers import extract_obs
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, 
    ThermalStorageConfig, PVConfig, SystemActions, SystemState, ExogenousData
)
import sample_data_generator
from build_my_house import create_2_room_house

# ==========================================
# 1. PPO Actor-Critic Network
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

# ==========================================
# 2. State Extraction & Action Mapping
# ==========================================
def map_actions(norm_actions: jax.Array, n_envs: int, n_rooms: int) -> SystemActions:
    bat_w = norm_actions[:, 0] * 3000.0
    hp_w = (norm_actions[:, 1:1+n_rooms] + 1.0) * 1000.0 # Map [-1, 1] to [0, 2000]
    ac_w = jnp.zeros((n_envs, n_rooms))
    stor_w = jnp.zeros((n_envs, n_rooms))
    return SystemActions(
        battery_power_w=bat_w, heat_pump_power_w=hp_w,
        ac_power_w=ac_w, storage_discharge_w=stor_w
    )

# ==========================================
# 3. GAE and Loss Functions
# ==========================================
@jax.jit
def compute_gae(rewards, values, next_value, dones, gamma=0.99, gae_lambda=0.95):
    """Calculates Generalized Advantage Estimation via a reverse scan."""
    def scan_fn(carry, transition):
        last_gae, next_val = carry
        reward, value, done = transition
        
        # done acts as a mask (0 if done, 1 if not done)
        mask = 1.0 - done
        delta = reward + gamma * next_val * mask - value
        gae = delta + gamma * gae_lambda * mask * last_gae
        
        return (gae, value), gae

    # Scan backwards through time
    rev_transitions = (rewards[::-1], values[::-1], dones[::-1])
    _, rev_advantages = jax.lax.scan(scan_fn, (jnp.zeros_like(next_value), next_value), rev_transitions)
    
    advantages = rev_advantages[::-1]
    returns = advantages + values
    return advantages, returns

@eqx.filter_value_and_grad(has_aux=True)
def ppo_loss(dynamic_policy, static_policy, obs, actions, old_log_probs, advantages, returns, clip_ratio=0.2):
    """Calculates the PPO clipped surrogate loss for Optax."""
    policy = eqx.combine(dynamic_policy, static_policy)
    
    # Evaluate current policy on the flattened batch
    mean, log_std, values = jax.vmap(policy)(obs)
    std = jnp.exp(log_std)
    
    # Calculate new log probabilities
    new_log_probs = -0.5 * jnp.sum(((actions - mean) / std)**2 + 2*log_std + jnp.log(2*jnp.pi), axis=-1)
    
    # Probability Ratio (pi_theta / pi_theta_old)
    ratio = jnp.exp(new_log_probs - old_log_probs)
    
    # Normalize advantages for training stability
    norm_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # 1. Actor Loss (Clipped)
    surr1 = ratio * norm_adv
    surr2 = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * norm_adv
    actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))
    
    # 2. Critic Loss (MSE)
    critic_loss = jnp.mean(jnp.square(values - returns))
    
    # 3. Entropy Bonus (Encourages exploration)
    entropy = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1))
    
    # Total combined loss
    total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
    
    return total_loss, (actor_loss, critic_loss, entropy)

# ==========================================
# 4. Main Training Loop
# ==========================================
def train():
    print("--- Starting PPO Vectorized Training ---")
    
    NUM_ENVS = 2048
    ROLLOUT_STEPS = 64
    EPOCHS = 200
    PPO_OPT_EPOCHS = 4 # Number of gradient steps per rollout
    LR = 3e-4
    
    # 1. Setup Env
    sample_data_generator.create_sample_data(n_days=10)
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, 900)
    t_config = create_2_room_house()
    n_rooms = len(t_config.room_air_indices)
    room_indices = jnp.array(t_config.room_air_indices)
    
    sim = JAXSimulator(
        dt_seconds=900, t_config=t_config, r_config=RewardConfig(price_weight=1.0, comfort_weight=5.0),
        b_config=BatteryConfig(), hp_config=HeatPumpConfig(), ac_config=AirConditionerConfig(), 
        ts_config=ThermalStorageConfig(), pv_config=PVConfig()
    )
    env = VectorizedEnergyEnv(sim, dataset, NUM_ENVS)
    
    # 2. Setup Policy
    key = jax.random.PRNGKey(42)
    key, model_key, env_key = jax.random.split(key, 3)
    
    obs_dim = n_rooms + 5
    action_dim = 1 + n_rooms 
    
    policy = PPOPolicy(obs_dim, action_dim, hidden_dim=64, key=model_key)
    # 1. Setup Scheduler (Decay by 90% over 150 epochs)
    # Total steps = ROLLOUT_STEPS * NUM_ENVS * EPOCHS (but optimizer sees them as updates)
    # Total update steps = EPOCHS * PPO_OPT_EPOCHS
    lr_scheduler = optax.exponential_decay(
        init_value=3e-4,
        transition_steps=PPO_OPT_EPOCHS * 5, # Reduce every 5 epochs
        decay_rate=0.98,                     # Gentle decay
        end_value=5e-6
    )

    optimizer = optax.adam(lr_scheduler)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_inexact_array))
    
    env_state = env.reset(env_key)

    # --- PPO Rollout Step (Jitted) ---
    @eqx.filter_jit
    def collect_and_train(policy, opt_state, current_env_state, rng):
        
        # A. Collect Trajectories
        def step_fn(carry, _):
            e_state, k = carry
            k, sample_key = jax.random.split(k)
            
            t = e_state.time_idx[0]
            exo_batch = jax.tree.map(lambda x: x[t], env.shared_exo_data)
            
            obs = jax.vmap(extract_obs, in_axes=(0, None, None))(e_state.sim.state, exo_batch, room_indices)
            
            mean, log_std, value = jax.vmap(policy)(obs)
            std = jnp.exp(log_std)
            
            noise = jax.random.normal(sample_key, mean.shape)
            norm_actions = mean + noise * std
            log_prob = -0.5 * jnp.sum(((norm_actions - mean) / std)**2 + 2*log_std + jnp.log(2*jnp.pi), axis=-1)
            
            phys_actions = map_actions(norm_actions, NUM_ENVS, n_rooms)
            next_e_state, reward, done, _ = env.step(e_state, phys_actions)
            
            # Scale reward down slightly for stable neural net gradients
            reward = reward / 100.0 
            
            transition = (obs, norm_actions, reward, value, log_prob, done.astype(jnp.float32))
            return (next_e_state, k), transition
            
        (final_env_state, rng), transitions = jax.lax.scan(
            step_fn, (current_env_state, rng), None, length=ROLLOUT_STEPS
        )
        obs, actions, rewards, values, log_probs, dones = transitions
        
        # B. Calculate Final Next Value (for GAE bootstrap)
        t_final = final_env_state.time_idx[0]
        exo_final = jax.tree.map(lambda x: x[t_final], env.shared_exo_data)
        obs_final = jax.vmap(extract_obs, in_axes=(0, None, None))(final_env_state.sim.state, exo_final, room_indices)
        _, _, next_values = jax.vmap(policy)(obs_final)
        
        # C. Compute Advantages
        advantages, returns = compute_gae(rewards, values, next_values, dones)
        
        # D. Flatten batch dims (ROLLOUT_STEPS, NUM_ENVS, ...) -> (ROLLOUT_STEPS * NUM_ENVS, ...)
        def flatten(x): return x.reshape(-1, *x.shape[2:])
        obs_flat, act_flat, lp_flat, adv_flat, ret_flat = map(flatten, (obs, actions, log_probs, advantages, returns))
        
        # E. Update Loop (PPO Epochs)
        
        # 1. Partition the policy BEFORE the scan!
        # dyn_p contains the weights/biases. stat_p contains the relu functions and network structure.
        dyn_policy, stat_policy = eqx.partition(policy, eqx.is_inexact_array)
        
        def update_epoch(carry, _):
            dyn_p, opt_k = carry
            
            # ppo_loss already beautifully expects the separated dyn_p and stat_policy!
            (loss, aux), grads = ppo_loss(dyn_p, stat_policy, obs_flat, act_flat, lp_flat, adv_flat, ret_flat)
            
            updates, new_opt_k = optimizer.update(grads, opt_k, dyn_p)
            new_dyn_p = eqx.apply_updates(dyn_p, updates)
            
            return (new_dyn_p, new_opt_k), (loss, aux)

        # 2. Run the scan carrying ONLY the dynamic weights and the optimizer state
        (new_dyn_policy, new_opt_state), (losses, aux_metrics) = jax.lax.scan(
            update_epoch, (dyn_policy, opt_state), None, length=PPO_OPT_EPOCHS
        )
        
        # 3. Recombine the updated weights with the static architecture
        new_policy = eqx.combine(new_dyn_policy, stat_policy)
        
        return new_policy, new_opt_state, final_env_state, rng, rewards, losses[-1]

    # --- Training Loop Execution ---
    print(f"Compiling Graph for {NUM_ENVS} parallel environments...")
    
    for epoch in range(EPOCHS):
        start = time.perf_counter()
        
        key, rollout_key = jax.random.split(key)
        
        # Execute the fully compiled rollout + GAE + Update loop!
        policy, opt_state, env_state, _, rewards, final_loss = collect_and_train(
            policy, opt_state, env_state, rollout_key
        )
        
        dur = time.perf_counter() - start
        avg_reward = float(jnp.mean(rewards)) * 100.0 # Unscale for printing
        fps = (NUM_ENVS * ROLLOUT_STEPS) / dur
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Avg Reward: {avg_reward:8.2f} | Loss: {final_loss:6.3f} | FPS: {fps:,.0f}")

if __name__ == "__main__":
    train()