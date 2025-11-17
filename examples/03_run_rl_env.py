# examples/03_run_rl_env.py
import os
import pandas as pd
from energysim.sim.simulator import JAXSimulator
from energysim.rl.env import EnergySimEnv
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    ThermalConfig, BatteryConfig, RewardConfig, HeatPumpConfig,
    ThermalStorageConfig, SolarConfig
)
from energysim.behavior.ev_charger import SimpleEVModel

import gymnasium as gym
import sample_data_generator    

def run_rl_env():
    """
    Demonstrates using the EnergySimEnv Gymnasium wrapper.
    
    This is the cleanest approach. The environment automatically:
    - Resets all behavioral models.
    - Calls `model.step()` at each timestep.
    - Merges behavioral loads and gains into the ExogenousData.
    - Builds a flat observation vector.
    - Returns the standard (obs, reward, terminated, truncated, info) tuple.
    """
    print("--- Running Example 03: Gymnasium RL Environment ---")
    
    # --- 1. Create Sample Data ---
    if not os.path.exists(sample_data_generator.FILE_NAME):
        sample_data_generator.create_sample_data()
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, dt_seconds=sample_data_generator.DT_SECONDS)

    # --- 2. Define All Configurations ---
    configs = {
        "dt_seconds": sample_data_generator.DT_SECONDS,
        "t_config": ThermalConfig(model_type="2R2C", setpoint=20.0),
        "b_config": BatteryConfig(capacity_kwh=10.0, max_power_kw=5.0),
        "hp_config": HeatPumpConfig(max_electrical_power_w=5000.0, cop_heating=3.0),
        "ts_config": ThermalStorageConfig(capacity_kwh=50.0, max_charge_kw=10.0),
        "s_config": SolarConfig(model_type="simple", panel_area_m2=25.0, efficiency=0.2),
        "r_config": RewardConfig(price_weight=1.0, comfort_weight=10.0)
    }
    
    # --- 3. Instantiate Simulator and Behavioral Models ---
    sim = JAXSimulator(**configs)
    
    # Create a dictionary of all behavioral models
    # The keys *must* match the device names in ExogenousData
    # e.g., "ev_charger" -> "ev_charger_load_w"
    behavioral_models = {
        "ev_charger": SimpleEVModel(seed=42, arrival_hour=18, charge_rate_kw=7.0)
        # "dishwasher": SimpleDishwasherModel(seed=123),
        # ... etc.
    }
    
    # --- 4. Create The Environment ---
    env = EnergySimEnv(
        simulator=sim,
        dataset=dataset,
        behavioral_models=behavioral_models
    )
    
    # You can optionally wrap it (e.g., for normalization)
    env = gym.wrappers.ClipAction(env)

    # --- 5. Run The Simulation Loop (Gym Standard) ---
    print(f"Running simulation for {len(dataset)} steps with random actions...")
    results = []
    obs, info = env.reset(seed=42)
    terminated = truncated = False
    
    while not (terminated or truncated):
        # Get a random action
        action = env.action_space.sample()
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log results
        results.append({
            "step": env.unwrapped._current_step,
            "reward": reward,
            "cost": info.get("cost", 0.0),
            # Get room temp from the complex observation vector
            "room_temp": obs[env.unwrapped.obs_map["state.thermal.room_temp"]]
        })
        
    print("Simulation complete.")
    env.close()
    
    # --- 6. Show Results ---
    results_df = pd.DataFrame(results)
    print("RL Environment Results (first 5 steps):")
    print(results_df.head())
    
    # --- 7. Cleanup ---
    os.remove(sample_data_generator.FILE_NAME)

if __name__ == "__main__":
    run_rl_env()