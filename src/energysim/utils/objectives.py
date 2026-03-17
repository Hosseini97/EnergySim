import jax.numpy as jnp
from jax import jit
from functools import partial
from ..core.shared.data_structs import (
    SystemOutputs, SystemState, SystemActions, ExogenousData,
    ThermalConfig, BatteryConfig, RewardConfig,
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, PVConfig,
    Array
)

def _effective_battery_grid_power_w(
    state: SystemState,
    requested_battery_power_w: Array,
    b_conf: BatteryConfig,
    dt_seconds: float,
) -> Array:
    """Compute feasible battery grid power from SOC, efficiency, and power limits."""
    one_way_eff = jnp.sqrt(b_conf.efficiency)
    max_power_w = b_conf.max_power_w

    # SOC-limited feasible charging/discharging over one control interval.
    available_discharge_batt_j = jnp.clip(state.battery.soc, 0.0, 1.0) * b_conf.capacity_j
    available_charge_batt_j = (1.0 - jnp.clip(state.battery.soc, 0.0, 1.0)) * b_conf.capacity_j

    # Convert battery-side energy limits to grid-side power limits.
    max_discharge_w_soc = (available_discharge_batt_j * one_way_eff) / dt_seconds
    max_charge_w_soc = (available_charge_batt_j / one_way_eff) / dt_seconds

    lower_bound_w = -jnp.minimum(max_power_w, max_discharge_w_soc)
    upper_bound_w = jnp.minimum(max_power_w, max_charge_w_soc)
    return jnp.clip(requested_battery_power_w, lower_bound_w, upper_bound_w)

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
    t_conf, b_conf, r_conf, hp_conf, ac_conf, ts_conf, _ = configs

    # ==========================================
    # 1. Economic Cost (Net Grid Energy)
    # ==========================================
    uncontrollable_load_w = (
        exogenous.base_load_w + exogenous.ev_charger_load_w + 
        exogenous.dishwasher_load_w + exogenous.clothes_dryer_load_w + 
        exogenous.water_heater_load_w + exogenous.cooking_load_w
    )

    effective_battery_power_w = _effective_battery_grid_power_w(
        state=state,
        requested_battery_power_w=actions.battery_power_w,
        b_conf=b_conf,
        dt_seconds=dt_seconds,
    )

    net_grid_power_w = (
        uncontrollable_load_w
        + effective_battery_power_w
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
    # 3. Control Regularization
    # ==========================================
    n_rooms = len(t_conf.room_air_indices)
    hp_max_room_w = hp_conf.max_electrical_power_w / n_rooms
    ac_max_room_w = ac_conf.max_electrical_power_w / n_rooms
    storage_max_room_w = ts_conf.max_discharge_w / n_rooms

    norm_hp = outputs.hp.electrical_power_w / jnp.maximum(hp_max_room_w, 1.0)
    norm_ac = outputs.ac.electrical_power_w / jnp.maximum(ac_max_room_w, 1.0)
    norm_storage = actions.storage_discharge_w / jnp.maximum(storage_max_room_w, 1.0)
    norm_battery = effective_battery_power_w / jnp.maximum(b_conf.max_power_w, 1.0)

    # Penalize impossible battery commands (e.g., requesting +5kW at full SOC).
    norm_batt_infeasible = (
        (actions.battery_power_w - effective_battery_power_w)
        / jnp.maximum(b_conf.max_power_w, 1.0)
    )

    # Discourage simultaneous HP+AC operation in the same zone.
    hvac_conflict_penalty = jnp.sum(norm_hp * norm_ac)
    action_l2_penalty = (
        0.5 * jnp.sum(norm_hp**2)
        + 0.5 * jnp.sum(norm_ac**2)
        + 0.3 * jnp.sum(norm_storage**2)
        + 0.1 * (norm_battery**2)
    )
    battery_throughput_penalty = norm_battery**2
    infeasible_battery_penalty = norm_batt_infeasible**2

    # ==========================================
    # 4. Total Weighted Objective
    # ==========================================
    total_cost = (
        (cost_euros * r_conf.price_weight)
        + (comfort_penalty * r_conf.comfort_weight)
        + (action_l2_penalty * 5.0)
        + (hvac_conflict_penalty * 25.0)
        + (battery_throughput_penalty * 8.0)
        + (infeasible_battery_penalty * 20.0)
    )

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
