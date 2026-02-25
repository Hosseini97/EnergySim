import jax.numpy as jnp
from jax import jit
from functools import partial
from ..core.shared.data_structs import (
    SystemOutputs, SystemState, SystemActions, ExogenousData,
    ThermalConfig, BatteryConfig, RewardConfig,
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, PVConfig,
    Array
)

@partial(jit, static_argnames=["dt_seconds"])
def f_cost_step(
    state: SystemState,
    actions: SystemActions,
    exogenous: ExogenousData,
    outputs: SystemOutputs,
    configs: tuple[ThermalConfig, BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, PVConfig],
    dt_seconds: float
) -> Array:
    """
    Bare-bones cost function for debugging. Focuses ONLY on temperature and basic electricity cost.
    """
    t_conf, b_conf, r_conf, _, _, _, _ = configs

    # ==========================================
    # 1. Economic Cost (Net Grid Energy)
    # ==========================================
    uncontrollable_load_w = (
        exogenous.base_load_w + exogenous.ev_charger_load_w + 
        exogenous.dishwasher_load_w + exogenous.clothes_dryer_load_w + 
        exogenous.water_heater_load_w + exogenous.cooking_load_w
    )

    net_grid_power_w = (
        uncontrollable_load_w
        + actions.battery_power_w
        + jnp.sum(outputs.hp.electrical_power_w)
        + jnp.sum(outputs.ac.electrical_power_w)
        - outputs.pv.pv_generation_w
    )

    net_grid_energy_kwh = (net_grid_power_w * dt_seconds) / 3600000.0
    
    # Simple asymmetric pricing: full price to buy, 20% to sell (prevents flat gradients)
    cost_euros = jnp.where(
        net_grid_energy_kwh > 0,
        net_grid_energy_kwh * exogenous.price,
        net_grid_energy_kwh * (exogenous.price * 0.20)
    )

    # ==========================================
    # 2. Comfort Cost (Quadratic Error)
    # ==========================================
    room_temps = state.thermal.T_vector[jnp.array(t_conf.room_air_indices)]
    comfort_penalty = jnp.sum((room_temps - t_conf.setpoint)**2)

    # ==========================================
    # 3. Total Weighted Objective
    # ==========================================
    total_cost = (cost_euros * r_conf.price_weight) + (comfort_penalty * r_conf.comfort_weight)

    return total_cost


@partial(jit, static_argnames=[])
def f_terminal_cost(
    final_state: SystemState,
    initial_state: SystemState, # Kept for signature compatibility
    configs: tuple,
    exo_forecast_end: ExogenousData
) -> Array:
    """
    Bare-bones terminal cost. Values leftover battery and heavily penalizes final temp deviations.
    """
    t_conf, b_conf, r_conf, _, _, _, _ = configs

    # 1. Battery Terminal Value (Negative cost = Reward)
    final_energy_kwh = final_state.battery.soc * b_conf.capacity_kwh
    battery_term_cost = -(final_energy_kwh * exo_forecast_end.price) * r_conf.price_weight

    # 2. Thermal Terminal Constraint
    room_temps = final_state.thermal.T_vector[jnp.array(t_conf.room_air_indices)]
    thermal_term_cost = jnp.sum((room_temps - t_conf.setpoint)**2) * 1000.0 

    return battery_term_cost + thermal_term_cost