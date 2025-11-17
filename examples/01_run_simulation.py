# examples/01_run_simulation.py
import os
import pandas as pd
import jax.numpy as jnp
from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    ThermalConfig, BatteryConfig, RewardConfig, HeatPumpConfig,
    ThermalStorageConfig, SolarConfig, SystemActions
)
from energysim.behavior.ev_charger import SimpleEVModel
import sample_data_generator

def run_open_loop_sim():
    """
    Demonstrates running the JAXSimulator directly.
    
    This approach requires you to manually:
    1. Get base data from the dataset.
    2. Run the `step()` function of all behavioral models.
    3. Manually merge the results into the `ExogenousData` PyTree.
    4. Define an `SystemActions` (e.g., from a simple script or "do nothing").
    5. Call the simulator's `step()` function.
    """
    print("--- Running Example 01: Open-Loop JAXSimulator ---")
    
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
    ev_model = SimpleEVModel(seed=42, arrival_hour=18, charge_rate_kw=7.0)
    
    # --- 4. Define a "Do Nothing" Action ---
    # We will apply this same (non-optimal) action at every step.
    null_actions = SystemActions(
        battery_power_w=jnp.array(0.0),
        heat_pump_power_w=jnp.array(0.0),
        ac_power_w=jnp.array(0.0),
        storage_discharge_w=jnp.array(0.0)
    )
    
    # --- 5. Run The Simulation Loop ---
    print(f"Running simulation for {len(dataset)} steps...")
    results = []
    state = sim.reset()
    ev_model.reset()
    
    for i in range(len(dataset)):
        # 1. Get base data from dataset
        exo_base = dataset[i]
        
        # 2. Run behavioral models
        ev_load_w = ev_model.step(i, sample_data_generator.DT_SECONDS, state)
        
        # 3. Calculate and merge device loads & gains
        # (Here we only have one device)
        device_gains_w = ev_load_w 
        
        exo_k = exo_base.replace(
            ev_charger_load_w=jnp.array(ev_load_w),
            device_gains_w=jnp.array(device_gains_w)
        )
        
        # 4. Step the simulator
        next_state, cost = sim.step(null_actions, exo_k)
        
        # 5. Log results
        results.append({
            "step": i,
            "room_temp": state.thermal.room_temp,
            "battery_soc": state.battery.soc,
            "storage_soc": state.storage.soc,
            "ev_load_w": ev_load_w,
            "cost": cost
        })
        
        # 6. Update state for next loop
        state = next_state
        
    print("Simulation complete.")
    
    # --- 6. Show Results ---
    results_df = pd.DataFrame(results)
    print("Simulation Results (first 5 steps):")
    print(results_df.head())
    
    # --- 7. Cleanup ---
    os.remove(sample_data_generator.FILE_NAME)

if __name__ == "__main__":
    run_open_loop_sim()