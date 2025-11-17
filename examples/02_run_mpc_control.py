# examples/02_run_mpc_control.py
import os
import pandas as pd
import jax.numpy as jnp
from energysim.sim.simulator import JAXSimulator
from energysim.control.mpc_solver import JAX_MPC_Solver
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    ThermalConfig, BatteryConfig, RewardConfig, HeatPumpConfig,
    ThermalStorageConfig, SolarConfig
)
from energysim.behavior.ev_charger import SimpleEVModel
import sample_data_generator

def run_mpc_control_sim():
    """
    Demonstrates running a closed-loop simulation with the JAX_MPC_Solver.
    
    This approach involves:
    1. Instantiating the JAXSimulator (the "plant").
    2. Instantiating the JAX_MPC_Solver (the "controller").
    3. At each step:
        a. Run behavioral models to get *current* loads.
        b. Merge loads into *current* ExogenousData.
        c. Get the *forecast* data from the dataset.
        d. Run behavioral models' `.forecast()` method.
        e. Merge behavioral forecasts into the forecast data.
        f. Call `mpc.solve()` to get optimal actions.
        g. Call `sim.step()` with those actions and *current* data.
    """
    print("--- Running Example 02: Closed-Loop MPC Control ---")
    
    # --- 1. Create Sample Data ---
    if not os.path.exists(sample_data_generator.FILE_NAME):
        sample_data_generator.create_sample_data()
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, dt_seconds=sample_data_generator.DT_SECONDS)

    # --- 2. Define All Configurations ---
    N_HORIZON = 24 # 6-hour horizon (24 steps * 15 min)
    configs = {
        "dt_seconds": sample_data_generator.DT_SECONDS,
        "t_config": ThermalConfig(model_type="2R2C", setpoint=20.0),
        "b_config": BatteryConfig(capacity_kwh=10.0, max_power_kw=5.0),
        "hp_config": HeatPumpConfig(max_electrical_power_w=5000.0, cop_heating=3.0),
        "ts_config": ThermalStorageConfig(capacity_kwh=50.0, max_charge_kw=10.0),
        "s_config": SolarConfig(model_type="simple", panel_area_m2=25.0, efficiency=0.2),
        "r_config": RewardConfig(price_weight=1.0, comfort_weight=10.0)
    }
    
    # --- 3. Instantiate Components ---
    sim = JAXSimulator(**configs)
    mpc = JAX_MPC_Solver(N_horizon=N_HORIZON, **configs)
    
    # Use the same EV model
    ev_model = SimpleEVModel(seed=42, arrival_hour=18, charge_rate_kw=7.0)
    
    # --- 4. Run The Simulation Loop ---
    print(f"Running simulation for {len(dataset) - N_HORIZON} steps...")
    results = []
    state = sim.reset()
    ev_model.reset()
    
    # We must stop early so the MPC has a full forecast
    for i in range(len(dataset) - N_HORIZON):
        # 1. Get *current* (k) data and run behavioral models
        exo_base_k = dataset[i]
        ev_load_k = ev_model.step(i, sample_data_generator.DT_SECONDS, state)
        device_gains_k = ev_load_k
        exo_k = exo_base_k.replace(
            ev_charger_load_w=jnp.array(ev_load_k),
            device_gains_w=jnp.array(device_gains_k)
        )
        
        # 2. Get *forecast* data
        exo_forecast_base = dataset.get_forecast(i, N_HORIZON)
        
        # 3. Run behavioral *forecasts*
        # (Here we use a simple forecast, a real one would be more complex)
        ev_load_forecast = ev_model.forecast(i, N_HORIZON,  sample_data_generator.DT_SECONDS, state)
        device_gains_forecast = ev_load_forecast # Assume gains = load
        
        exo_forecast = exo_forecast_base.replace(
            ev_charger_load_w=ev_load_forecast,
            device_gains_w=device_gains_forecast
        )
        
        # 4. Solve MPC
        # We pass the current state and the full forecast
        actions = mpc.solve(state, exo_forecast)
        
        # 5. Step the simulator
        next_state, cost = sim.step(actions, exo_k)
        
        # 6. Log results
        results.append({
            "step": i,
            "room_temp": state.thermal.room_temp,
            "battery_soc": state.battery.soc,
            "storage_soc": state.storage.soc,
            "battery_action": actions.battery_power_w,
            "hp_action": actions.heat_pump_power_w,
            "cost": cost
        })
        
        # 7. Update state for next loop
        state = next_state

    print("Simulation complete.")
    
    # --- 5. Show Results ---
    results_df = pd.DataFrame(results)
    print("MPC Simulation Results (first 5 steps):")
    print(results_df.head())
    
    # --- 6. Cleanup ---
    os.remove(sample_data_generator.FILE_NAME)

if __name__ == "__main__":
    run_mpc_control_sim()