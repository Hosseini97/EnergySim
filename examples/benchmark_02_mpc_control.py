# examples/benchmark_02_mpc_control.py
import os
import time
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

# --- Define simulation length ---
N_DAYS = 7 # MPC is slow, so we test for 1 week
N_HORIZON = 4 * 6 # 6-hour horizon @ 15-min steps
N_STEPS = N_DAYS * sample_data_generator.STEPS_PER_DAY

# Use the same data-merging helper from the MPC example
def get_merged_exo_data(
    dataset: SimulationDataset,
    behavioral_models: Dict[str, AbstractBehavioralModel],
    start_idx: int,
    current_state: SystemState,
    n_rooms: int,
    is_forecast: bool = False,
    horizon: int = 1
) -> ExogenousData:
    
    if is_forecast:
        base_exo = dataset.get_forecast(start_idx, horizon)
    else:
        base_exo = dataset[start_idx]
    
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
            if key in ["dishwasher", "cooking"]:
                internal_gains_load_w += pred_w

    split = jnp.array([0.6, 0.4])
    
    if is_forecast:
        solar_gains_zonal_w = jnp.outer(base_exo.solar_gains_w, split)
        occupancy_gains_zonal_w = jnp.outer(base_exo.occupancy_gains_w, split)
        device_gains_zonal_w = jnp.outer(internal_gains_load_w, split)
    else:
        solar_gains_zonal_w = base_exo.solar_gains_w * split
        occupancy_gains_zonal_w = base_exo.occupancy_gains_w * split
        device_gains_zonal_w = internal_gains_load_w * split

    merged_exo = base_exo.replace(
        **behavioral_loads,
        solar_gains_w=solar_gains_zonal_w,
        occupancy_gains_w=occupancy_gains_zonal_w,
        device_gains_w=device_gains_zonal_w
    )
    return merged_exo


def run_mpc_benchmark():
    print("--- Running Benchmark 02: MPC Speed & Quality ---")

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
    mpc = JAX_MPC_Solver(N_horizon=N_HORIZON, **configs)

    behavioral_models = {
        "ev_charger": SimpleEVModel(seed=42, arrival_hour=18, charge_rate_kw=7.0),
        "dishwasher": StochasticTimeModel(seed=43, power_kw=1.5, duration_minutes=90, start_window=(19, 22), daily_prob=0.5),
        "water_heater": ThermostaticLoadModel(seed=44, power_kw=4.0, capacity_kwh=10.0, setpoint_soc=0.9, deadband_soc=0.3),
        "cooking": StochasticImpulseModel(seed=45, power_kw=3.0, duration_minutes=45, time_windows=[(7, 8), (12, 13), (18, 19)], prob_per_window=0.8)
    }
    
    null_actions = SystemActions(
        battery_power_w=jnp.array(0.0),
        heat_pump_power_w=jnp.zeros(N_ROOMS),
        ac_power_w=jnp.zeros(N_ROOMS),
        storage_discharge_w=jnp.zeros(N_ROOMS)
    )

    # --- 4. Run Baseline (Do Nothing) Simulation ---
    print(f"Running baseline 'Do Nothing' sim for {N_STEPS} steps...")
    baseline_total_cost = 0.0
    
    state = sim.reset()
    for model in behavioral_models.values():
        model.reset()

    start_baseline = time.perf_counter()
    for i in range(N_STEPS):
        exo_k = get_merged_exo_data(
            dataset, behavioral_models, i, state, N_ROOMS, is_forecast=False
        )
        next_state, cost = sim.step(null_actions, exo_k)
        baseline_total_cost += cost
        state = next_state
    baseline_time = time.perf_counter() - start_baseline
    
    print("Baseline complete.")

    # --- 5. Run MPC Simulation ---
    print(f"Running MPC sim for {N_STEPS - N_HORIZON} steps...")
    mpc_total_cost = 0.0
    solve_times = []
    
    state = sim.reset()
    for model in behavioral_models.values():
        model.reset()

    start_mpc = time.perf_counter()
    for i in range(N_STEPS - N_HORIZON):
        # 1. Get current data
        exo_k = get_merged_exo_data(
            dataset, behavioral_models, i, state, N_ROOMS, is_forecast=False
        )
        # 2. Get forecast data
        exo_forecast = get_merged_exo_data(
            dataset, behavioral_models, i, state, N_ROOMS, is_forecast=True, horizon=N_HORIZON
        )

        # 3. Solve MPC (and time it)
        t_solve_start = time.perf_counter()
        actions = mpc.solve(state, exo_forecast)
        actions.battery_power_w.block_until_ready() # Ensure solve is finished
        t_solve_end = time.perf_counter()
        solve_times.append(t_solve_end - t_solve_start)

        # 4. Step the simulator
        next_state, cost = sim.step(actions, exo_k)
        mpc_total_cost += cost
        state = next_state
    
    mpc_time = time.perf_counter() - start_mpc
    print("MPC simulation complete.")

    # --- 6. Show Results ---
    avg_solve_time_ms = np.mean(solve_times) * 1000
    
    print("\n--- Benchmark 02 Results ---")
    print(f"Simulation length: {N_DAYS} days ({N_STEPS} steps)")
    print(f"MPC Horizon: {N_HORIZON} steps")
    print("\n--- Decision Quality ---")
    print(f"Baseline (Do Nothing) Total Cost: {baseline_total_cost:,.2f} €")
    print(f"MPC Control Total Cost: {mpc_total_cost:,.2f} €")
    
    print("\n--- Performance ---")
    print(f"Baseline Sim Time: {baseline_time:.4f} s")
    print(f"MPC Sim Time: {mpc_time:.4f} s")
    print(f"Average MPC solve() time: {avg_solve_time_ms:.2f} ms/step")
    
    # Cleanup
    os.remove(sample_data_generator.FILE_NAME)


if __name__ == "__main__":
    run_mpc_benchmark()