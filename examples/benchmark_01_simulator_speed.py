# examples/benchmark_01_simulator_speed.py
import os
import time
import jax
import jax.numpy as jnp

# Core simulator
from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset

# Configs and data structs
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig,
    ThermalStorageConfig, SolarConfig, SystemActions, ExogenousData
)

# House builder and data generator
import sample_data_generator
from build_my_house import create_2_room_house

# --- Define simulation length ---
N_DAYS = 180 # ~6 months
N_STEPS = N_DAYS * sample_data_generator.STEPS_PER_DAY

def get_sim_and_data():
    """Helper to set up simulator and data"""
    # --- 1. Create Sample Data ---
    if not os.path.exists(sample_data_generator.FILE_NAME):
        sample_data_generator.create_sample_data(n_days=N_DAYS)
    dataset = SimulationDataset(
        sample_data_generator.FILE_NAME, 
        dt_seconds=sample_data_generator.DT_SECONDS
    )
    # Ensure dataset is long enough
    if len(dataset) < N_STEPS:
        sample_data_generator.create_sample_data(n_days=N_DAYS)
        dataset = SimulationDataset(
            sample_data_generator.FILE_NAME, 
            dt_seconds=sample_data_generator.DT_SECONDS
        )

    # --- 2. Define All Configurations ---
    t_config = create_2_room_house()
    N_ROOMS = len(t_config.room_air_indices)
    
    configs = {
        "dt_seconds": sample_data_generator.DT_SECONDS,
        "t_config": t_config,
        "b_config": BatteryConfig(capacity_kwh=13.0, max_power_kw=5.0),
        "hp_config": HeatPumpConfig(model_type="variable_cop", max_electrical_power_w=8000.0),
        "ac_config": AirConditionerConfig(model_type="ramping", max_electrical_power_w=6000.0),
        "ts_config": ThermalStorageConfig(capacity_kwh=50.0, max_charge_kw=10.0),
        "s_config": SolarConfig(model_type="simple", panel_area_m2=30.0, efficiency=0.2),
        "r_config": RewardConfig(price_weight=1.0, comfort_weight=10.0)
    }
    
    sim = JAXSimulator(**configs)
    
    # --- 3. Pre-load all data and actions ---
    
    # "Do Nothing" actions
    null_actions = SystemActions(
        battery_power_w=jnp.array(0.0),
        heat_pump_power_w=jnp.zeros(N_ROOMS),
        ac_power_w=jnp.zeros(N_ROOMS),
        storage_discharge_w=jnp.zeros(N_ROOMS)
    )
    
    # Get all exogenous data as a single stacked PyTree
    # This is a list of PyTrees, one for each step
    exo_data_list = [dataset[i] for i in range(N_STEPS)]
    # Stack them into one PyTree where each leaf has shape (N_STEPS, ...)
    all_exo_data = jax.tree.map(lambda *x: jnp.stack(x), *exo_data_list)

    # Manually broadcast gains (this is done in the RL env / MPC script)
    split = jnp.array([0.6, 0.4]) # 60/40 split
    all_exo_data = all_exo_data.replace(
        solar_gains_w = jnp.outer(all_exo_data.solar_gains_w, split),
        occupancy_gains_w = jnp.outer(all_exo_data.occupancy_gains_w, split),
        device_gains_w = jnp.zeros((N_STEPS, N_ROOMS)) # Assume 0 device gains
    )
    
    return sim, sim.reset(), all_exo_data, null_actions

def benchmark_simulator_speed():
    """
    Measures the raw JAX sim.scan_step() speed using lax.scan.
    This is the theoretical maximum performance.
    """
    print("--- Running Benchmark 01: Raw Simulator Speed ---")
    
    sim, initial_state, all_exo_data, null_actions = get_sim_and_data()

    # Create the function to be scanned
    # It must take (carry, x) and return (new_carry, y)
    def scan_body(state, exo_data_k):
        # We use null_actions for a pure simulation test
        next_state, cost = sim.scan_step(null_actions, exo_data_k)
        # return (new_state, (cost, next_state))
        return next_state, (cost, next_state)

    # JIT-compile the scan function
    @jax.jit
    def run_simulation(init_state, exo_data_all_steps):
        final_state, (all_costs, all_states) = jax.lax.scan(
            scan_body, init_state, exo_data_all_steps
        )
        return final_state, all_costs

    # --- Run 1: JIT Compilation ---
    print(f"Running simulation for {N_STEPS} steps ({N_DAYS} days)...")
    print("Running JIT compilation step (warmup)...")
    start_warmup = time.perf_counter()
    final_state_warmup, all_costs_warmup = run_simulation(initial_state, all_exo_data)
    # Block until JIT is finished
    final_state_warmup.battery.soc.block_until_ready()
    warmup_time = time.perf_counter() - start_warmup
    print(f"JIT compilation took: {warmup_time:.4f} s")

    # --- Run 2: Timed Execution ---
    print("Running timed benchmark...")
    start_time = time.perf_counter()
    final_state, all_costs = run_simulation(initial_state, all_exo_data)
    # Block until execution is finished
    final_state.battery.soc.block_until_ready()
    total_time = time.perf_counter() - start_time
    
    steps_per_sec = N_STEPS / total_time
    
    print("\n--- Benchmark 01 Results ---")
    print(f"Total time for {N_STEPS} steps: {total_time:.4f} s")
    print(f"Simulator speed: {steps_per_sec:,.2f} steps/sec")
    print(f"Total cost (do nothing): {all_costs.sum():,.2f}")
    
    # Cleanup
    os.remove(sample_data_generator.FILE_NAME)

if __name__ == "__main__":
    benchmark_simulator_speed()