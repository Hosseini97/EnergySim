import time
import jax
import jax.numpy as jnp
from energysim.sim.simulator import JAXSimulator
from energysim.rl.vector_env import VectorizedEnergyEnv
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, 
    ThermalStorageConfig, PVConfig, SystemActions
)
from energysim.behavior import SimpleEVModel
import sample_data_generator
from build_my_house import create_2_room_house

def benchmark_vector():
    print("--- Benchmark: Vectorized RL Environment ---")
    
    NUM_ENVS = 4096
    STEPS = 100000
    
    # 1. Setup Data & Simulator
    sample_data_generator.create_sample_data(n_days=30)
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, 900)
    t_config = create_2_room_house()
    n_rooms = len(t_config.room_air_indices)
    
    sim = JAXSimulator(
        dt_seconds=900, 
        t_config=t_config, 
        r_config=RewardConfig(), 
        b_config=BatteryConfig(), 
        hp_config=HeatPumpConfig(),
        ac_config=AirConditionerConfig(), 
        ts_config=ThermalStorageConfig(), 
        pv_config=PVConfig()
    )
    
    vec_env = VectorizedEnergyEnv(
        simulator=sim,
        dataset=dataset,
        num_envs=NUM_ENVS,
        behavioral_models={"ev": SimpleEVModel()}
    )
    
    # 2. Cleaner Random Policy (No dummy broadcasting needed)
    def random_policy(key):
        k1, k2, k3 = jax.random.split(key, 3)
        
        # Directly instantiate the PyTree with batched arrays
        return SystemActions(
            battery_power_w=jax.random.uniform(k1, (NUM_ENVS,), minval=-3000, maxval=3000),
            heat_pump_power_w=jax.random.uniform(k2, (NUM_ENVS, n_rooms), minval=0, maxval=2000),
            ac_power_w=jax.random.uniform(k3, (NUM_ENVS, n_rooms), minval=0, maxval=2000),
            storage_discharge_w=jnp.zeros((NUM_ENVS, n_rooms))
        )

    # 3. Rollout Loop
    @jax.jit
    def run_rollout(start_state, start_key):
        
        def scan_body(carry, _):
            state, key = carry
            key, pol_key = jax.random.split(key)
            
            actions = random_policy(pol_key)
            next_state, reward, done, info = vec_env.step(state, actions)
            
            return (next_state, key), reward

        _, rewards = jax.lax.scan(scan_body, (start_state, start_key), None, length=STEPS)
        return rewards

    # 4. Execution
    key = jax.random.PRNGKey(0)
    state = vec_env.reset(key)
    
    print(f"Compiling for {NUM_ENVS} parallel environments...")
    start = time.perf_counter()
    run_rollout(state, key).block_until_ready()
    print(f"Compilation done: {time.perf_counter()-start:.2f}s")
    
    print("Simulating...")
    start = time.perf_counter()
    run_rollout(state, key).block_until_ready()
    duration = time.perf_counter() - start
    
    total_steps = NUM_ENVS * STEPS
    print("\nRESULTS:")
    print(f"Total Transitions: {total_steps:,}")
    print(f"Time: {duration:.4f}s")
    print(f"Throughput: {total_steps/duration:,.0f} steps/second")

if __name__ == "__main__":
    benchmark_vector()