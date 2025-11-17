# examples/01_run_advanced_zonal_simulation.py
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
    ThermalStorageConfig, SolarConfig, SystemActions
)

# --- NEW: Import all behavioral models ---
from energysim.behavior import (
    SimpleEVModel,
    StochasticTimeModel,
    ThermostaticLoadModel,
    StochasticImpulseModel
)

# --- NEW: Import the house builder and data generator ---
import sample_data_generator
from build_my_house import create_2_room_house


def run_zonal_open_loop_sim():
    """
    Demonstrates running the JAXSimulator with a multi-zone ThermalConfig.
    
    This approach shows how to:
    1. Load a valid ThermalConfig from the RCNetworkBuilder.
    2. Instantiate *all* behavioral models.
    3. Manually merge all behavioral loads and gains at each step.
    4. Define *zonal* SystemActions (e.g., shape (2,)) for a simple thermostat.
    5. Call the simulator's `step()` function.
    """
    print("--- Running Example 01: Open-Loop Zonal Simulation ---")

    # --- 1. Create Sample Data ---
    if not os.path.exists(sample_data_generator.FILE_NAME):
        sample_data_generator.create_sample_data()
    dataset = SimulationDataset(
        sample_data_generator.FILE_NAME, 
        dt_seconds=sample_data_generator.DT_SECONDS
    )

    # --- 2. Define All Configurations ---
    
    # --- NEW: Load the 2-room house config ---
    t_config = create_2_room_house()
    N_ROOMS = len(t_config.room_air_indices) # Should be 2
    print(f"Loaded ThermalConfig for {N_ROOMS} rooms.")

    # Use more advanced, stateful models
    hp_config = HeatPumpConfig(
        model_type="variable_cop", 
        max_electrical_power_w=8000.0, # 8kW total
        ramp_rate_w_per_sec=1000.0
    )
    ac_config = AirConditionerConfig(
        model_type="ramping", 
        max_electrical_power_w=6000.0, # 6kW total
        ramp_rate_w_per_sec=1000.0
    )
    
    configs = {
        "dt_seconds": sample_data_generator.DT_SECONDS,
        "t_config": t_config,
        "b_config": BatteryConfig(capacity_kwh=13.0, max_power_kw=5.0),
        "hp_config": hp_config,
        "ac_config": ac_config,
        "ts_config": ThermalStorageConfig(capacity_kwh=50.0, max_charge_kw=10.0),
        "s_config": SolarConfig(model_type="simple", panel_area_m2=30.0, efficiency=0.2),
        "r_config": RewardConfig(price_weight=1.0, comfort_weight=10.0)
    }

    # --- 3. Instantiate Simulator and Behavioral Models ---
    sim = JAXSimulator(**configs)

    # --- NEW: Instantiate all behavioral models ---
    behavioral_models = {
        "ev_charger": SimpleEVModel(seed=42, arrival_hour=18, charge_rate_kw=7.0),
        "dishwasher": StochasticTimeModel(
            seed=43, power_kw=1.5, duration_minutes=90, 
            start_window=(19, 22), daily_prob=0.5
        ),
        "water_heater": ThermostaticLoadModel(
            seed=44, power_kw=4.0, capacity_kwh=10.0, 
            setpoint_soc=0.9, deadband_soc=0.3
        ),
        "cooking": StochasticImpulseModel(
            seed=45, power_kw=3.0, duration_minutes=45,
            time_windows=[(7, 8), (12, 13), (18, 19)], # Breakfast, Lunch, Dinner
            prob_per_window=0.8
        )
    }

    # --- 4. Run The Simulation Loop ---
    print(f"Running simulation for {len(dataset)} steps...")
    results = []
    
    # Reset simulator and models
    state = sim.reset()
    for model in behavioral_models.values():
        model.reset()

    for i in range(len(dataset)):
        # 1. Get base data from dataset
        exo_base = dataset[i]

        # 2. Run all behavioral models
        behavioral_loads = {
            key: model.step(i, sample_data_generator.DT_SECONDS, state)
            for key, model in behavioral_models.items()
        }

        # 3. Calculate and merge device loads & gains
        
        # --- NEW: Differentiate between total load and thermal gains ---

        # Total electrical load for cost calculation (all devices)
        total_device_load_w = sum(behavioral_loads.values())

        # Total *internal thermal gains* (only some devices)
        # We assume EV charger and water heater are not in the living space
        # and only contribute heat via their electrical load.
        internal_gains_load_w = (
            behavioral_loads["dishwasher"] +
            behavioral_loads["cooking"]
        )
        
        # --- Distribute ALL gains to zones ---
        # Get scalar values from the base dataset
        scalar_solar_gains = exo_base.solar_gains_w
        scalar_occupancy_gains = exo_base.occupancy_gains_w

        # Simple assumption: 60% to living room (idx 0), 40% to bedroom (idx 1)
        # --- FIX: Use internal_gains_load_w, not total_device_load_w ---
        device_gains_zonal_w = jnp.array([
            internal_gains_load_w * 0.6,
            internal_gains_load_w * 0.4
        ])
        solar_gains_zonal_w = jnp.array([
            scalar_solar_gains * 0.6,
            scalar_solar_gains * 0.4
        ])
        occupancy_gains_zonal_w = jnp.array([
            scalar_occupancy_gains * 0.6,
            scalar_occupancy_gains * 0.4
        ])


        # Create the merged ExogenousData PyTree for this step
        exo_k = exo_base.replace(
            # Behavioral loads
            ev_charger_load_w=jnp.array(behavioral_loads["ev_charger"]),
            dishwasher_load_w=jnp.array(behavioral_loads["dishwasher"]),
            water_heater_load_w=jnp.array(behavioral_loads["water_heater"]),
            cooking_load_w=jnp.array(behavioral_loads["cooking"]),
            device_gains_w=device_gains_zonal_w,
            solar_gains_w=solar_gains_zonal_w,
            occupancy_gains_w=occupancy_gains_zonal_w
        )

        # 4. Define Zonal Actions (Simple Thermostat)
        room_temps = state.thermal.T_vector[jnp.array(t_config.room_air_indices)]
        
        # Heat if below 20.5C, Cool if above 22.5C
        hp_action_w = jnp.where(room_temps < 20.5, 2000.0, 0.0) # 3kW per room
        ac_action_w = jnp.where(room_temps > 22.5, 1000.0, 0.0) # 3kW per room
        
        # --- Define Thermal Discharge Action ---
        # Tell the storage tank to discharge heat when the HP is on.
        # We'll request the thermal equivalent (e.g., using a fixed COP of 3.0 for this simple logic).
        # The storage model will clip this based on its actual SOC.
        discharge_request_w = hp_action_w * 3.0
        
        actions = SystemActions(
            battery_power_w=jnp.array(0.0),      # Do nothing with battery
            heat_pump_power_w=hp_action_w,     # Zonal action, shape (2,)
            ac_power_w=ac_action_w,            # Zonal action, shape (2,)
            storage_discharge_w=discharge_request_w # <-- FIX
        )

        # 5. Step the simulator
        next_state, cost = sim.step(actions, exo_k)

        # 6. Log results
        results.append({
            "step": i,
            "room_temp_0": float(room_temps[0]),
            "room_temp_1": float(room_temps[1]),
            "battery_soc": float(state.battery.soc),
            "storage_soc": float(state.storage.soc),
            "ev_load_w": behavioral_loads["ev_charger"],
            "total_device_load_w": total_device_load_w,
            "hp_action_0": float(hp_action_w[0]),
            "cost": float(cost)
        })

        # 7. Update state for next loop
        state = next_state

    print("Simulation complete.")

    # --- 5. Show Results ---
    results_df = pd.DataFrame(results)
    print("Zonal Simulation Results (first 5 steps):")
    print(results_df.head())
    
    print("\nZonal Simulation Results (last 5 steps):")
    print(results_df.tail())

    # --- 6. Cleanup ---
    os.remove(sample_data_generator.FILE_NAME)


if __name__ == "__main__":
    run_zonal_open_loop_sim()