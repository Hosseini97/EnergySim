# examples/01_run_simple_simulation.py
import jax.numpy as jnp
import pandas as pd
import numpy as np
import equinox as eqx  # <--- REQUIRED IMPORT

from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, 
    ThermalStorageConfig, SolarConfig, SystemActions
)
import sample_data_generator
from build_my_house import create_2_room_house

def run():
    # 1. Setup Data & Config
    sample_data_generator.create_sample_data(n_days=3)
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, sample_data_generator.DT_SECONDS)
    t_config = create_2_room_house()
    
    n_rooms = int(len(t_config.room_air_indices))

    sim = JAXSimulator(
        dt_seconds=sample_data_generator.DT_SECONDS,
        t_config=t_config,
        r_config=RewardConfig(),
        b_config=BatteryConfig(capacity_kwh=13.0),
        hp_config=HeatPumpConfig(model_type="ramping", max_electrical_power_w=4000.0),
        ac_config=AirConditionerConfig(model_type="ramping", max_electrical_power_w=4000.0),
        ts_config=ThermalStorageConfig(),
        s_config=SolarConfig()
    )

    # 2. Simulation Loop
    state = sim.reset()
    
    prev_action = SystemActions(
        battery_power_w=jnp.array(0.0),
        heat_pump_power_w=jnp.zeros(n_rooms),
        ac_power_w=jnp.zeros(n_rooms),
        storage_discharge_w=jnp.zeros(n_rooms)
    )
    
    results = []
    print("Running Open-Loop Simulation...")

    split_factors = jnp.array([0.6, 0.4])

    for i in range(len(dataset)):
        exo_base = dataset[i]

        # --- FIX: Use eqx.tree_at instead of .replace() ---
        # This creates a new copy of the struct with the updated vector values
        exo = eqx.tree_at(
            lambda e: (e.solar_gains_w, e.occupancy_gains_w, e.device_gains_w),
            exo_base,
            (
                exo_base.solar_gains_w * split_factors,      # Scalar -> Vector
                exo_base.occupancy_gains_w * split_factors,  # Scalar -> Vector
                jnp.zeros(n_rooms)                           # Scalar -> Vector
            )
        )
        
        # --- Controller ---
        temps = state.thermal.T_vector[jnp.array(t_config.room_air_indices)]
        hp_w = jnp.where(temps < 20.0, 2000.0, 0.0) 
        ac_w = jnp.where(temps > 23.0, 1500.0, 0.0)
        
        action = SystemActions(
            battery_power_w=jnp.array(0.0),
            heat_pump_power_w=hp_w,
            ac_power_w=ac_w,
            storage_discharge_w=hp_w * 3.0
        )
        
        new_sim, cost = sim.step(action, prev_action, exo)
        state = new_sim.state
        prev_action = action
        
        results.append({
            "step": i,
            "temp_0": float(temps[0]),
            "temp_1": float(temps[1]),
            "cost": float(cost)
        })

    print(pd.DataFrame(results).head())

if __name__ == "__main__":
    run()