# examples/03_run_zonal_rl_env.py
import os
import pandas as pd
import jax.numpy as jnp
import numpy as np
import gymnasium as gym

# Core components
from energysim.sim.simulator import JAXSimulator
from energysim.rl.env import EnergySimEnv # Use the updated Env
from energysim.core.data.dataset import SimulationDataset

# Configs and data structs
from energysim.core.shared.data_structs import (
    ThermalConfig, BatteryConfig, RewardConfig, HeatPumpConfig, 
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


def run_zonal_rl_env():
    """
    Demonstrates running the Zonal-Aware EnergySimEnv wrapper.
    
    This is the cleanest approach for RL. The environment automatically:
    - Resets all behavioral models.
    - Calls `model.step()` at each timestep.
    - Differentiates internal gains (e.g., cooking) from external loads (e.g., EV).
    - Broadcasts scalar gains (e.g., solar) to zonal vectors.
    - Builds a flat, zonal-aware observation vector.
    - Unflattens a flat, zonal-aware action vector.
    - Returns the standard (obs, reward, terminated, truncated, info) tuple.
    """
    print("--- Running Example 03: Gymnasium Zonal RL Environment ---")

    # --- 1. Create Sample Data ---
    if not os.path.exists(sample_data_generator.FILE_NAME):
        sample_data_generator.create_sample_data()
    dataset = SimulationDataset(
        sample_data_generator.FILE_NAME, 
        dt_seconds=sample_data_generator.DT_SECONDS
    )

    # --- 2. Define All Configurations ---
    t_config = create_2_room_house()
    N_ROOMS = len(t_config.room_air_indices)
    print(f"Loaded ThermalConfig for {N_ROOMS} rooms.")
    
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
    # Use the (now zonal-aware) Env
    env = EnergySimEnv(
        simulator=sim,
        dataset=dataset,
        behavioral_models=behavioral_models,
        # Tell the env which devices create internal heat
        internal_gain_devices=["dishwasher", "cooking"]
    )
    env = gym.wrappers.ClipAction(env)

    # --- 5. Run The Simulation Loop (Gym Standard) ---
    print(f"Running simulation for {len(dataset)} steps with random actions...")
    results = []
    obs, info = env.reset(seed=42)
    terminated = truncated = False
    
    # Get slices for logging from the env's internal map
    T_vec_slice = env.unwrapped.obs_map_slices["state.thermal.T_vector"]
    room_air_indices = env.unwrapped.room_air_indices
    ev_load_slice = env.unwrapped.obs_map_slices["exo.ev_charger_load_w"]

    while not (terminated or truncated):
        # Get a random action from the zonal action space
        action = env.action_space.sample()
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Get room temps from the full observation vector
        T_vector = obs[T_vec_slice]
        room_temps = T_vector[room_air_indices] # e.g., T_vector[[1, 2]]

        results.append({
            "step": env.unwrapped._current_step,
            "reward": reward,
            "cost": info.get("cost", 0.0),
            "room_temp_0": room_temps[0],
            "room_temp_1": room_temps[1],
            "ev_load": obs[ev_load_slice].sum() # .sum() since it's a 1-element slice
        })
        
        if env.unwrapped._current_step % 24 == 0:
            print(f"Completed step {env.unwrapped._current_step}, Reward: {reward:.2f}, T_room0: {room_temps[0]:.2f}")


    print("Simulation complete.")
    env.close()

    # --- 6. Show Results ---
    results_df = pd.DataFrame(results)
    print("Zonal RL Environment Results (first 5 steps):")
    print(results_df.head())
    
    print("\nZonal RL Environment Results (last 5 steps):")
    print(results_df.tail())

    # --- 7. Cleanup ---
    os.remove(sample_data_generator.FILE_NAME)

if __name__ == "__main__":
    run_zonal_rl_env()