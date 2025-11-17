# examples/02_run_zonal_mpc.py
import os
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np

# Core simulator and controller
from energysim.sim.simulator import JAXSimulator
from energysim.control.mpc_solver import JAX_MPC_Solver
from energysim.core.data.dataset import SimulationDataset

# Configs and data structs
from energysim.core.shared.data_structs import (
    ThermalConfig, BatteryConfig, RewardConfig, HeatPumpConfig, 
    AirConditionerConfig, ThermalStorageConfig, SolarConfig, SystemActions, 
    ExogenousData, SystemState
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
from typing import Dict
from energysim.behavior.base import AbstractBehavioralModel

# --- Helper function to manage complex data merging for MPC ---

def get_merged_exo_data(
    dataset: SimulationDataset,
    behavioral_models: Dict[str, AbstractBehavioralModel],
    start_idx: int,
    current_state: SystemState,
    n_rooms: int,
    is_forecast: bool = False,
    horizon: int = 1
) -> ExogenousData:
    """
    Fetches base data and merges behavioral model outputs.
    - If is_forecast=False, gets a single step.
    - If is_forecast=True, gets a forecast over the horizon.
    """
    
    # 1. Get base data from the dataset
    if is_forecast:
        base_exo = dataset.get_forecast(start_idx, horizon)
    else:
        base_exo = dataset[start_idx] # This is a single PyTree
    
    # 2. Run all behavioral models (either .step() or .forecast())
    behavioral_loads = {}
    internal_gains_load_w = jnp.zeros(horizon) if is_forecast else 0.0
    
    for key, model in behavioral_models.items():
        field_name = f"{key}_load_w"
        if hasattr(base_exo, field_name):
            if is_forecast:
                pred_w = model.forecast(
                    start_idx, horizon, dataset.dt_seconds, current_state
                )
            else:
                pred_w = model.step(
                    start_idx, dataset.dt_seconds, current_state
                )
            
            behavioral_loads[field_name] = jnp.array(pred_w)
            
            # 3. Differentiate internal gains from total load
            if key in ["dishwasher", "cooking"]: # EV/WaterHeater are not internal gains
                internal_gains_load_w += pred_w

    # 4. Broadcast scalar gains to zonal vectors
    # Simple assumption: 60% to living room (idx 0), 40% to bedroom (idx 1)
    # This matches the logic from the open-loop example
    split = jnp.array([0.6, 0.4])
    
    if is_forecast:
        # base_exo.solar_gains_w has shape (horizon,)
        # We need to broadcast it to (horizon, n_rooms)
        solar_gains_zonal_w = jnp.outer(base_exo.solar_gains_w, split)
        occupancy_gains_zonal_w = jnp.outer(base_exo.occupancy_gains_w, split)
        device_gains_zonal_w = jnp.outer(internal_gains_load_w, split)
    else:
        # All gains are scalars, broadcast to (n_rooms,)
        solar_gains_zonal_w = base_exo.solar_gains_w * split
        occupancy_gains_zonal_w = base_exo.occupancy_gains_w * split
        device_gains_zonal_w = internal_gains_load_w * split

    # 5. Merge results into the base data
    merged_exo = base_exo.replace(
        **behavioral_loads,
        solar_gains_w=solar_gains_zonal_w,
        occupancy_gains_w=occupancy_gains_zonal_w,
        device_gains_w=device_gains_zonal_w
    )
    return merged_exo


def run_zonal_mpc_sim():
    print("--- Running Example 02: Closed-Loop Zonal MPC Control ---")

    # --- 1. Create Sample Data ---
    if not os.path.exists(sample_data_generator.FILE_NAME):
        sample_data_generator.create_sample_data()
    dataset = SimulationDataset(
        sample_data_generator.FILE_NAME, 
        dt_seconds=sample_data_generator.DT_SECONDS
    )

    # --- 2. Define All Configurations ---
    N_HORIZON = 24 # 6-hour horizon (24 steps * 15 min)
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

    # --- 3. Instantiate Components ---
    sim = JAXSimulator(**configs)
    
    # Use the (now zonal-aware) MPC solver
    mpc = JAX_MPC_Solver(N_horizon=N_HORIZON, **configs)

    behavioral_models = {
        "ev_charger": SimpleEVModel(seed=42, arrival_hour=18, charge_rate_kw=7.0),
        "dishwasher": StochasticTimeModel(seed=43, power_kw=1.5, duration_minutes=90, start_window=(19, 22), daily_prob=0.5),
        "water_heater": ThermostaticLoadModel(seed=44, power_kw=4.0, capacity_kwh=10.0, setpoint_soc=0.9, deadband_soc=0.3),
        "cooking": StochasticImpulseModel(seed=45, power_kw=3.0, duration_minutes=45, time_windows=[(7, 8), (12, 13), (18, 19)], prob_per_window=0.8)
    }
    
    # --- 4. Run The Simulation Loop ---
    print(f"Running simulation for {len(dataset) - N_HORIZON} steps...")
    results = []
    
    state = sim.reset()
    for model in behavioral_models.values():
        model.reset()

    warm_start_actions = None # Let MPC use its default

    for i in range(len(dataset) - N_HORIZON):
        
        # 1. Get *current* (k) merged exogenous data
        exo_k = get_merged_exo_data(
            dataset, behavioral_models, i, state, N_ROOMS, is_forecast=False
        )

        # 2. Get *forecast* merged exogenous data
        exo_forecast = get_merged_exo_data(
            dataset, behavioral_models, i, state, N_ROOMS, is_forecast=True, horizon=N_HORIZON
        )

        # 3. Solve MPC
        actions = mpc.solve(state, exo_forecast, warm_start_actions)

        # 4. Step the simulator
        next_state, cost = sim.step(actions, exo_k)

        # 5. Log results
        room_temps = state.thermal.T_vector[jnp.array(t_config.room_air_indices)]
        results.append({
            "step": i,
            "room_temp_0": float(room_temps[0]),
            "room_temp_1": float(room_temps[1]),
            "battery_soc": float(state.battery.soc),
            "storage_soc": float(state.storage.soc),
            "hp_action_0": float(actions.heat_pump_power_w[0]),
            "hp_action_1": float(actions.heat_pump_power_w[1]),
            "cost": float(cost)
        })

        # 6. Update state for next loop
        state = next_state
        
        if i % 24 == 0:
            print(f"Completed step {i}, Cost: {cost:.2f}, T_room0: {room_temps[0]:.2f}, T_room1: {room_temps[1]:.2f}")

    print("Simulation complete.")

    # --- 5. Show Results ---
    results_df = pd.DataFrame(results)
    print("Zonal MPC Simulation Results (first 5 steps):")
    print(results_df.head())
    
    print("\nZonal MPC Simulation Results (last 5 steps):")
    print(results_df.tail())

    # --- 6. Cleanup ---
    os.remove(sample_data_generator.FILE_NAME)


if __name__ == "__main__":
    run_zonal_mpc_sim()