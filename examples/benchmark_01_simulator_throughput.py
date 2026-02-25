import time
import jax
import jax.numpy as jnp
import equinox as eqx
from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, 
    ThermalStorageConfig, PVConfig, SystemActions
)
import sample_data_generator
from build_my_house import create_2_room_house

def benchmark():
    print("--- Benchmark: Raw JAX Simulator Throughput ---")
    
    # 1. Massive Dataset (10 Year)
    N_DAYS = 3650
    sample_data_generator.create_sample_data(n_days=N_DAYS)
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, 900)
    
    # Load all data into GPU memory
    all_exo = dataset.get_forecast(0, len(dataset))
    
    # 2. Init Simulator
    t_config = create_2_room_house()

    sim = JAXSimulator(
        dt_seconds=900, t_config=t_config, r_config=RewardConfig(),
        b_config=BatteryConfig(), hp_config=HeatPumpConfig(), 
        ac_config=AirConditionerConfig(), ts_config=ThermalStorageConfig(), 
        pv_config=PVConfig()
    )
    initial_sim = sim.reset()

    # Pass the ENTIRE simulator PyTree as an argument, not just its state
    @jax.jit
    def run_full_year(sim_carry, exo_data):
        
        def step_fn(carry, exo):
            curr_sim, = carry

            action = SystemActions(
                battery_power_w=jnp.array(0.0),
                heat_pump_power_w=jnp.array([0.0, 0.0]),
                ac_power_w=jnp.array([0.0, 0.0]),
                storage_discharge_w=jnp.array([0.0, 0.0])
            )
            
            # Step the simulator that is threaded through the loop
            new_sim, outputs = curr_sim.step(action, exo)
            
            # Must return exactly (new_carry, step_output)
            # We don't need to save outputs for a pure speed benchmark, so we return None as output
            return (new_sim, ), None

        # Thread the full simulator through the scan
        final_carry, _ = jax.lax.scan(step_fn, (sim_carry, ), exo_data)
        
        return final_carry

    # 4. Warmup
    print("Compiling JAX Kernel...")
    start = time.perf_counter()
    
    # Run and block_until_ready on a specific array inside the returned PyTree
    final_carry_warmup = run_full_year(initial_sim, all_exo)
    
    # Wait for the GPU to finish execution. JAX executes asynchronously!
    # We grab an arbitrary leaf from the final state to force synchronization.
    final_carry_warmup[0].state.thermal.T_vector.block_until_ready()
    
    print(f"Compilation finished in {time.perf_counter() - start:.4f}s")

    # 5. Benchmark
    print(f"Running {len(dataset)} steps...")
    start = time.perf_counter()
    
    final_carry_bench = run_full_year(initial_sim, all_exo)
    final_carry_bench[0].state.thermal.T_vector.block_until_ready()
    
    end = time.perf_counter()
    
    duration = end - start
    sps = len(dataset) / duration
    print(f"\nRESULTS:")
    print(f"Time: {duration:.4f} seconds")
    print(f"Speed: {sps:,.0f} steps/second")

if __name__ == "__main__":
    benchmark()