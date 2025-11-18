import time
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from energysim.sim.simulator import JAXSimulator
from energysim.rl.vector_env import VectorizedEnergyEnv
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    SystemActions, BatteryConfig, HeatPumpConfig, 
    AirConditionerConfig, ThermalStorageConfig, 
    SolarConfig, RewardConfig
)
from energysim.behavior import (
    SimpleEVModel, StochasticTimeModel, StochasticImpulseModel
)
import sample_data_generator
from build_my_house import create_2_room_house

# --- Settings ---
N_ENVS = 16384
N_DAYS = 365
STEPS_PER_HOUR = 4
N_STEPS = N_DAYS * 24 * STEPS_PER_HOUR

def setup_sim_and_data():
    if not os.path.exists(sample_data_generator.FILE_NAME):
        sample_data_generator.create_sample_data(n_days=N_DAYS)
    
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, dt_seconds=900)
    if len(dataset) < N_STEPS:
        sample_data_generator.create_sample_data(n_days=N_DAYS)
        dataset = SimulationDataset(sample_data_generator.FILE_NAME, dt_seconds=900)

    t_config = create_2_room_house()
    n_rooms = len(t_config.room_air_indices)

    configs = {
        "dt_seconds": 900,
        "t_config": t_config,
        "b_config": BatteryConfig(capacity_kwh=13.0),
        "hp_config": HeatPumpConfig(model_type="variable_cop", max_electrical_power_w=8000.0),
        "ac_config": AirConditionerConfig(model_type="ramping", max_electrical_power_w=6000.0),
        "ts_config": ThermalStorageConfig(capacity_kwh=50.0),
        "s_config": SolarConfig(model_type="simple"),
        "r_config": RewardConfig()
    }
    
    sim = JAXSimulator(**configs)
    
    b_models = {
        "ev_charger": SimpleEVModel(seed=42),
        "dishwasher": StochasticTimeModel(seed=43, power_kw=1.5, duration_minutes=90, start_window=(19, 22)),
        "cooking": StochasticImpulseModel(seed=45, power_kw=3.0, duration_minutes=45, time_windows=[(18, 19)])
    }
    
    return sim, dataset, b_models, n_rooms

@partial(jax.jit, static_argnames=["num_envs", "n_rooms"])
def random_policy(key, num_envs, n_rooms):
    key_b, key_hp, key_ac, key_ts = jax.random.split(key, 4)
    return SystemActions(
        battery_power_w=jax.random.uniform(key_b, (num_envs,), minval=-3000, maxval=3000),
        heat_pump_power_w=jax.random.uniform(key_hp, (num_envs, n_rooms), minval=0, maxval=2000),
        ac_power_w=jax.random.uniform(key_ac, (num_envs, n_rooms), minval=0, maxval=2000),
        storage_discharge_w=jax.random.uniform(key_ts, (num_envs, n_rooms), minval=0, maxval=2000),
    )

# --- JIT Compiled Rollout Loop ---
@jax.jit
def run_rollout_kernel(env, start_state, start_key):
    
    def scan_body(carry, t):
        state, key = carry
        
        # 1. Policy
        key, subkey = jax.random.split(key)
        actions = random_policy(subkey, env.num_envs, 2)
        
        # 2. Time Index (Synced)
        time_indices = jnp.full((env.num_envs,), t, dtype=jnp.int32)
        
        # 3. Public Env Step
        next_state, costs = env.step(state, actions, time_indices)
        
        return (next_state, key), costs

    horizon = jnp.arange(N_STEPS)
    (final_state, _), all_costs = jax.lax.scan(
        scan_body, 
        (start_state, start_key), 
        horizon
    )
    return final_state, all_costs

def run_benchmark():
    print(f"--- Vectorized EnergyEnv Benchmark ---")
    sim, dataset, b_models, n_rooms = setup_sim_and_data()

    print("Initializing Env...")
    # This will now trigger 'Pre-calculating logic (Shared Mode)...'
    vec_env = VectorizedEnergyEnv(sim, dataset, N_ENVS, b_models)
    
    print("\nPreparing states...")
    key = jax.random.PRNGKey(0)
    initial_states = vec_env.reset(key)

    print("Compiling XLA Kernel (JIT)...")
    start_compile = time.perf_counter()
    final_state_warmup, _ = run_rollout_kernel(vec_env, initial_states, key)
    jax.tree.map(lambda x: x.block_until_ready(), final_state_warmup)
    print(f"Compilation finished in {time.perf_counter() - start_compile:.4f}s")

    print("Running Simulation...")
    start_run = time.perf_counter()
    final_state, all_costs = run_rollout_kernel(vec_env, initial_states, key)
    all_costs.block_until_ready()
    duration = time.perf_counter() - start_run
    
    total_steps = N_ENVS * N_STEPS
    sps = total_steps / duration
    
    print(f"\n--- Results ---")
    print(f"Transitions: {total_steps:,}")
    print(f"Time: {duration:.4f} s")
    print(f"Throughput: {sps:,.0f} steps/s")
    
    

if __name__ == "__main__":
    run_benchmark()