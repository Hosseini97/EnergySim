# energysim/core/models/objectives.py
import jax.numpy as jnp
from jax import jit
from functools import partial
from ..shared.data_structs import (
    SystemState, SystemActions, ExogenousData,
    HeatPumpOutput, AirConditionerOutput, ThermalStorageOutput, SolarOutput, # <-- ADDED SolarOutput
    ThermalConfig, BatteryConfig, RewardConfig,
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, SolarConfig, # <-- ADDED SolarConfig
    Array
)

@partial(jit, static_argnames=["dt_seconds"])
def f_cost_step(
    state: SystemState,
    actions: SystemActions,
    exogenous: ExogenousData,
    hp_output: HeatPumpOutput,
    ac_output: AirConditionerOutput,
    storage_output: ThermalStorageOutput,
    solar_output: SolarOutput, # <--- NEW
    configs: tuple[ThermalConfig, BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, SolarConfig], # <-- ADDED SolarConfig
    dt_seconds: float
) -> Array:
    """
    Calculates the total cost for a single timestep.
    """
    t_conf, b_conf, r_conf, hp_conf, ac_conf, ts_conf, s_conf = configs

    # --- 1. Calculate Electrical Cost ---
    hp_electrical_power_w = jnp.sum(hp_output.electrical_power_w)
    ac_electrical_power_w = jnp.sum(ac_output.electrical_power_w)
    
    # --- NEW: Sum all load components ---
    total_load_w = (
        exogenous.base_load_w
        + exogenous.ev_charger_load_w
        + exogenous.dishwasher_load_w
        + exogenous.clothes_dryer_load_w
        + exogenous.water_heater_load_w
        + exogenous.cooking_load_w
    )

    net_grid_power_w = (
        total_load_w
        + actions.battery_power_w    # Battery action is still scalar
        + hp_electrical_power_w      # Total from all rooms
        + ac_electrical_power_w      # Total from all rooms
        - solar_output.pv_generation_w
    )

    net_grid_energy_kwh = (net_grid_power_w * (dt_seconds / 3600.0)) / 1000.0
    cost_euros = jnp.fmax(0.0, net_grid_energy_kwh) * exogenous.price

    # --- 2. Calculate Comfort Cost ---
    # Get the T_vector from the state
    T_vector = state.thermal.T_vector
    
    # Use the indices from the config to select the room temps
    room_temps = T_vector[jnp.array(t_conf.room_air_indices)]
    
    # The rest of the logic works on this new `room_temps` vector
    temp_error = room_temps - t_conf.setpoint
    comfort_violation = jnp.fmax(0.0, jnp.abs(temp_error) - t_conf.comfort_band)
    total_comfort_penalty = jnp.sum(comfort_violation**2)

    # --- 3. Calculate Waste Penalty ---
    # Sum the waste from all rooms
    total_rejected_heat_w = jnp.sum(storage_output.rejected_heat_w)
    rejected_heat_kwh = (total_rejected_heat_w * (dt_seconds / 3600.0)) / 1000.0
    waste_penalty = rejected_heat_kwh * exogenous.price

    # --- 4. Total Weighted Cost ---
    total_cost = (
        (cost_euros * r_conf.price_weight) +
        (total_comfort_penalty * r_conf.comfort_weight) + # Use the summed penalty
        (waste_penalty * r_conf.price_weight)
    )

    return total_cost