import jax
import jax.numpy as jnp
import equinox as eqx
import time
from energysim.sim.simulator import JAXSimulator
from energysim.rl.vector_env import VectorizedEnergyEnv
from energysim.rl.helpers import extract_obs
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import RewardConfig, BatteryConfig, HeatPumpConfig, AirConditionerConfig, SystemActions

def map_actions(norm_actions, n_envs, n_rooms):
    return SystemActions(
        battery_power_w=norm_actions[:, 0] * 3000.0,
        heat_pump_power_w=(norm_actions[:, 1:1+n_rooms] + 1.0) * 1000.0,
        ac_power_w=jnp.zeros((num_envs, n_rooms)),
        storage_discharge_w=jnp.zeros((num_envs, n_rooms))
    )

def rbc_logic(state, exo):
    # Simple, JAX-friendly Bang-Bang Logic
    # Heat if Temp < 20.0, Stop if Temp > 22.0
    temp = state.thermal.T_vector[1] # Room air temp
    heating_action = jnp.where(temp < 20.0, 1.0, 0.0)
    # Map to [-1, 1] range to match PPO action space
    hp_action = (heating_action * 2.0) - 1.0 
    return jnp.array([0.0, hp_action]) # [Battery, HP]

def main():
    print("--- 🏁 Final Apples-to-Apples Benchmark ---")
    dt_seconds = 900
    num_envs = 256
    test_steps = 960 # 10 days
    
    dataset = SimulationDataset(file_path="/home/emmanuel-gendy/Documents/ppo_jax/heeten_training_master.csv", dt_seconds=dt_seconds)
    from build_my_house import create_2_room_house
    t_config = create_2_room_house()
    n_rooms = len(t_config.room_air_indices)

    sim = JAXSimulator(dt_seconds=dt_seconds, t_config=t_config, 
                       r_config=RewardConfig(price_weight=1.0, comfort_weight=5.0))
    env = VectorizedEnergyEnv(sim, dataset, num_envs=num_envs)

    # 1. RUN RBC
    print("Evaluating RBC...")
    vmap_rbc_logic = jax.vmap(rbc_logic, in_axes=(0, None))
    @eqx.filter_jit
    def run_rbc(s):
        def step(curr, _):
            exo = jax.tree.map(lambda x: x[curr.time_idx[0]], env.shared_exo_data)
            actions_norm = vmap_rbc_logic(curr.sim.state, exo)
            ns, rew, _, _ = env.step(curr, map_actions(actions_norm, num_envs, n_rooms))
            return ns, (rew, ns.sim.thermal.T_vector[:, 1])
        return jax.lax.scan(step, s, None, length=test_steps)[1]

    rbc_rewards, rbc_temps = run_rbc(env.reset(jax.random.PRNGKey(0)))

    # 2. RUN PPO
    print("Evaluating PPO...")
    # (Load your PPOPolicy class and trained_policy here as done previously)
    # ... [Insert PPO execution logic here] ...
    
    # CALCULATE FINAL METRICS
    days = test_steps / 96
    rbc_cost = -jnp.mean(jnp.sum(rbc_rewards, axis=0)) / days
    rbc_temp = jnp.mean(rbc_temps)
    
    print(f"\n--- 📊 FINAL COMPARISON ---")
    print(f"RBC Daily Cost: € {rbc_cost:.2f} (Avg Temp: {rbc_temp:.2f}°C)")
    # (Print PPO results)