import os
import pandas as pd
import jax.numpy as jnp
import numpy as np

# Core simulator
from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset

# Configs and data structs
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig,
    ThermalStorageConfig, SolarConfig, SystemActions, ThermalConfig
)

# House builder and data generator
import sample_data_generator
from build_my_house import create_2_room_house


def run_simple_sim():
    """
    Demonstrates running the JAXSimulator directly with NO behavioral models,
    but WITH a simple "bang-bang" (on/off) thermostat.
    """
    print("--- Running Example 01: Simple Simulator (with Thermostat) ---")

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
        "hp_config": HeatPumpConfig(model_type="ramping", max_electrical_power_w=8000.0),
        "ac_config": AirConditionerConfig(model_type="ramping", max_electrical_power_w=6000.0),
        "ts_config": ThermalStorageConfig(capacity_kwh=50.0, max_charge_kw=10.0),
        "s_config": SolarConfig(model_type="simple", panel_area_m2=30.0, efficiency=0.2),
        "r_config": RewardConfig(price_weight=1.0, comfort_weight=10.0)
    }

    # --- 3. Instantiate Simulator ---
    sim = JAXSimulator(**configs)

    # --- 4. Run The Simulation Loop ---
    print(f"Running simulation for {len(dataset)} steps...")
    results = []
    
    state = sim.reset()

    for i in range(len(dataset)):
        # 1. Get base data from dataset
        exo_base = dataset[i]
        
        # 2. Create merged ExogenousData for this step
        scalar_solar_gains = exo_base.solar_gains_w
        scalar_occupancy_gains = exo_base.occupancy_gains_w
        split = jnp.array([0.6, 0.4])
        
        solar_gains_zonal_w = scalar_solar_gains * split
        occupancy_gains_zonal_w = scalar_occupancy_gains * split
        device_gains_zonal_w = jnp.zeros(N_ROOMS)

        exo_k = exo_base.replace(
            solar_gains_w=solar_gains_zonal_w,
            occupancy_gains_w=occupancy_gains_zonal_w,
            device_gains_w=device_gains_zonal_w
        )

        # --- 4. UPDATED: Define Zonal Actions (Simple Thermostat) ---
        # Get the room temperatures from the *current* state
        room_temps = state.thermal.T_vector[jnp.array(t_config.room_air_indices)]
        
        # --- Define Electrical Actions ---
        # Heat if below 20.5C, Cool if above 22.5C
        hp_action_w = jnp.where(room_temps < 20.5, 4000.0, 0.0) # 4kW electrical per room
        ac_action_w = jnp.where(room_temps > 22.5, 3000.0, 0.0) # 3kW electrical per room
        
        # --- Define Thermal Discharge Action ---
        # Tell the storage tank to discharge heat when the HP is on.
        discharge_request_w = hp_action_w * 3.0 # Request 3x thermal
        
        actions = SystemActions(
            battery_power_w=jnp.array(0.0),      # Do nothing with battery
            heat_pump_power_w=hp_action_w,     # Zonal action, shape (2,)
            ac_power_w=ac_action_w,            # Zonal action, shape (2,)
            storage_discharge_w=discharge_request_w
        )
        # --- End of Update ---

        # 5. Step the simulator
        next_state, cost = sim.step(actions, exo_k) # <-- Use the new 'actions'

        # 6. Log results
        results.append({
            "step": i,
            "room_temp_0": float(room_temps[0]),
            "room_temp_1": float(room_temps[1]),
            "battery_soc": float(state.battery.soc),
            "storage_soc": float(state.storage.soc),
            "base_load_w": float(exo_k.base_load_w),
            "solar_gen_w": float(sim._solar.calculate(exo_k).pv_generation_w),
            "cost": float(cost)
        })

        # 7. Update state for next loop
        state = next_state

    print("Simulation complete.")

    # --- 6. Show Results ---
    results_df = pd.DataFrame(results)
    print("Simple Simulation Results (first 5 steps):")
    print(results_df.head())
    
    print("\nSimple Simulation Results (last 5 steps):")
    print(results_df.tail())

    # --- 7. Cleanup ---
    os.remove(sample_data_generator.FILE_NAME)


if __name__ == "__main__":
    run_simple_sim()