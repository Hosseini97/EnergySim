# examples/benchmark_03_rl_env_speed.py
import os
import time
import numpy as np
import gymnasium as gym

# Core components
from energysim.sim.simulator import JAXSimulator
from energysim.rl.env import EnergySimEnv # Use the updated Env
from energysim.core.data.dataset import SimulationDataset

# Configs and data structs
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, 
    AirConditionerConfig, ThermalStorageConfig, SolarConfig
)

# All behavioral models
from energysim.behavior import (
    SimpleEVModel,
    StochasticTimeModel,
    ThermostaticLoadModel,
    StochasticImpulseModel
)

# House builder and data generator
import sample_data_generator
from build_my_house import create_2_room_house

# --- Define simulation length ---
N_DAYS = 30 # RL env is faster than MPC, can run for longer
N_STEPS = N_DAYS * sample_data_generator.STEPS_PER_DAY

def run_rl_env_benchmark():
    """
    Measures the `env.step()` speed for a random agent.
    This includes the Python overhead of behavioral models,
    data merging, and observation flattening.
    """
    print("--- Running Benchmark 03: RL Environment Speed ---")

    # --- 1. Create Sample Data ---
    if not os.path.exists(sample_data_generator.FILE_NAME):
        sample_data_generator.create_sample_data(n_days=N_DAYS)
    dataset = SimulationDataset(
        sample_data_generator.FILE_NAME, 
        dt_seconds=sample_data_generator.DT_SECONDS
    )
    if len(dataset) < N_STEPS:
        sample_data_generator.create_sample_data(n_days=N_DAYS)
        dataset = SimulationDataset(
            sample_data_generator.FILE_NAME, 
            dt_seconds=sample_data_generator.DT_SECONDS
        )

    # --- 2. Define All Configurations ---
    t_config = create_2_room_house()
    
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

    # --- 3. Instantiate Simulator and Behavioral Models ---
    sim = JAXSimulator(**configs)

    behavioral_models = {
        "ev_charger": SimpleEVModel(seed=42, arrival_hour=18, charge_rate_kw=7.0),
        "dishwasher": StochasticTimeModel(seed=43, power_kw=1.5, duration_minutes=90, start_window=(19, 22), daily_prob=0.5),
        "water_heater": ThermostaticLoadModel(seed=44, power_kw=4.0, capacity_kwh=10.0, setpoint_soc=0.9, deadband_soc=0.3),
        "cooking": StochasticImpulseModel(seed=45, power_kw=3.0, duration_minutes=45, time_windows=[(7, 8), (12, 13), (18, 19)], prob_per_window=0.8)
    }
    
    # --- 4. Create The Environment ---
    env = EnergySimEnv(
        simulator=sim,
        dataset=dataset,
        behavioral_models=behavioral_models,
        internal_gain_devices=["dishwasher", "cooking"]
    )
    
    # --- 5. Run The Simulation Loop (Gym Standard) ---
    print(f"Running simulation for {N_STEPS} steps ({N_DAYS} days)...")
    
    obs, info = env.reset(seed=42)
    terminated = truncated = False
    
    # --- Run 1: Warmup ---
    # Run a few steps to JIT-compile the sim.step() inside the env
    print("Running warmup steps...")
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    
    print("Warmup complete. Running timed benchmark...")
    # Reset again for a fair start
    obs, info = env.reset(seed=42)
    terminated = truncated = False
    total_cost = 0.0

    # --- Run 2: Timed Execution ---
    start_time = time.perf_counter()
    for i in range(N_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_cost -= reward # reward is -cost
        
        if terminated or truncated:
            # This loop should match N_STEPS, but just in case
            print(f"Simulation ended early at step {i}")
            break
            
    total_time = time.perf_counter() - start_time

    print("Simulation complete.")
    env.close()

    # --- 6. Show Results ---
    steps_per_sec = N_STEPS / total_time
    
    print("\n--- Benchmark 03 Results ---")
    print(f"Total time for {N_STEPS} steps: {total_time:.4f} s")
    print(f"Gym Env speed: {steps_per_sec:,.2f} steps/sec")
    print(f"Total cost (random agent): {total_cost:,.2f}")
    
    # Cleanup
    os.remove(sample_data_generator.FILE_NAME)

if __name__ == "__main__":
    run_rl_env_benchmark()