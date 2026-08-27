from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jax import jit

from ..core.shared.data_structs import (
    SystemActions,
    SystemOutputs,
    SystemState,
    ExogenousData,
    ThermalConfig,
    BatteryConfig,
    RewardConfig,
    HeatPumpConfig,
    AirConditionerConfig,
    ThermalStorageConfig,
    PVConfig,
    Array,
)


# ---------------------------------------------------------------------
# Economic assumptions
# ---------------------------------------------------------------------

EXPORT_PRICE_FRACTION = 0.20

TERMINAL_SOC_TARGET = 0.50
TERMINAL_SOC_WEIGHT_EUR_PER_KWH2 = 0.03

TINY_QP_REGULARIZATION = 1e-9


class BatteryQPStaticData(NamedTuple):
    """Fixed hard-MPC QP data for a chosen horizon and battery configuration."""

    Q: Array
    A: Array
    l_base: Array
    u_base: Array
    terminal_soc_coeff: Array
    idx_import: Array
    idx_export: Array
    row_grid: Array
    row_import_bound: Array
    row_export_bound: Array
    row_soc: Array
    capacity_wh: Array
    max_power_w: Array
    import_price_scale: Array
    import_fee_offset: Array
    export_price_scale: Array
    export_price_offset: Array
    terminal_linear_scale: Array
    terminal_soc_target: Array


# ---------------------------------------------------------------------
# Battery-only action bounds
# ---------------------------------------------------------------------

def build_battery_only_bounds(
    N_horizon: int,
    n_rooms: int,
    b_conf: BatteryConfig,
) -> tuple[SystemActions, SystemActions]:
    """
    Static physical action bounds.

    Sign convention:
        battery_power_w > 0: charging
        battery_power_w < 0: discharging

    These bounds only define the final SystemActions interface.
    The hard SOC constraints are handled in the QP formulation.
    """

    scalar_shape = (N_horizon,)
    zonal_shape = (N_horizon, n_rooms)

    action_min = SystemActions(
        battery_power_w=jnp.full(scalar_shape, -b_conf.max_power_w),
        heat_pump_power_w=jnp.zeros(zonal_shape),
        ac_power_w=jnp.zeros(zonal_shape),
        storage_discharge_w=jnp.zeros(zonal_shape),
    )

    action_max = SystemActions(
        battery_power_w=jnp.full(scalar_shape, b_conf.max_power_w),
        heat_pump_power_w=jnp.zeros(zonal_shape),
        ac_power_w=jnp.zeros(zonal_shape),
        storage_discharge_w=jnp.zeros(zonal_shape),
    )

    return action_min, action_max


# ---------------------------------------------------------------------
# Forecast extraction
# ---------------------------------------------------------------------

def extract_battery_only_forecast(
    exo_forecast: ExogenousData,
    N_horizon: int,
    pv_conf,
):
    """
    Extract load, PV and price forecast for battery-only MPC.

    For passthrough PV, solar_irradiance_w_m2 already stores PV power.
    Otherwise, use the same simple area/efficiency calculation as SimplePVModel.
    """

    load_w = (
        exo_forecast.base_load_w
        + exo_forecast.ev_charger_load_w
        + exo_forecast.dishwasher_load_w
        + exo_forecast.clothes_dryer_load_w
        + exo_forecast.water_heater_load_w
        + exo_forecast.cooking_load_w
    )

    if pv_conf.model_type == "passthrough":
        pv_w = exo_forecast.solar_irradiance_w_m2
    elif hasattr(pv_conf, "peak_power_w"):
        pv_w = (
            exo_forecast.solar_irradiance_w_m2
            / 1000.0
            * pv_conf.peak_power_w
        )
    else:
        temp_factor = (
            1.0
            + (exo_forecast.ambient_temp - pv_conf.reference_temp_c)
            * pv_conf.temp_coefficient
        )
        pv_w = jnp.fmax(
            0.0,
            exo_forecast.solar_irradiance_w_m2
            * pv_conf.panel_area_m2
            * pv_conf.efficiency
            * temp_factor,
        )

    price = exo_forecast.price

    return (
        load_w[:N_horizon],
        pv_w[:N_horizon],
        price[:N_horizon],
    )


# ---------------------------------------------------------------------
# QP objective and constraints
# ---------------------------------------------------------------------

@partial(jit, static_argnames=["dt_seconds"])
def f_stage_cost(
    current_state: SystemState,
    next_state: SystemState,
    actions: SystemActions,
    outputs: SystemOutputs,
    exogenous: ExogenousData,
    configs: tuple[
        ThermalConfig,
        BatteryConfig,
        RewardConfig,
        HeatPumpConfig,
        AirConditionerConfig,
        ThermalStorageConfig,
        PVConfig,
    ],
    dt_seconds: float,
) -> Array:
    """
    Stage electricity bill, plus an optional thermal-comfort penalty.

    The QP enforces battery feasibility, so this cost uses the returned battery
    action directly without a second feasibility repair.

    Comfort is a DEAD-BAND penalty: zero anywhere inside
    [setpoint - comfort_band, setpoint + comfort_band], quadratic outside it. The
    band is not slack to be tidied away -- it IS the flexibility a HEMS trades on.
    A bare setpoint penalises every deviation, which leaves an agent no room to
    shift heating in time and so nothing to learn.

    It is off by default (comfort_weight = 0.0) so this function is unchanged for
    every existing caller. See RewardConfig.comfort_weight for the units and for
    how to pick a value that is not a guess.
    """
    del current_state

    t_conf, _, r_conf, _, _, _, _ = configs

    uncontrollable_load_w = (
        exogenous.base_load_w
        + exogenous.ev_charger_load_w
        + exogenous.dishwasher_load_w
        + exogenous.clothes_dryer_load_w
        + exogenous.water_heater_load_w
        + exogenous.cooking_load_w
    )

    controllable_electric_power_w = (
        jnp.sum(outputs.hp.electrical_power_w)
        + jnp.sum(outputs.ac.electrical_power_w)
    )

    net_grid_w = (
        uncontrollable_load_w
        + controllable_electric_power_w
        + actions.battery_power_w
        - outputs.pv.pv_generation_w
    )

    import_w = jnp.maximum(net_grid_w, 0.0)
    export_w = jnp.maximum(-net_grid_w, 0.0)
    energy_factor_kwh_per_w = dt_seconds / 3600000.0

    # Import and export are priced separately -- see RewardConfig. The defaults
    # leave this identical to billing both directions at the wholesale price.
    # VAT applies to the WHOLE bill -- energy and grid fees -- not to the energy alone,
    # so the fee is added before the tax rather than after. That is how German household
    # billing works, and it also matters numerically: wholesale reaches -0.05 EUR/kWh on
    # ~2.7% of real steps, and taxing first turns that into -0.0595 before any fee is
    # applied, so a 0.05 fee still leaves the import price negative. Adding first means
    # any fee above the most negative wholesale price is enough.
    #
    # A negative import price is not a rounding detail: it means consumption pays, and a
    # perfect-foresight optimizer will find it. The thermal oracle discovered it and used
    # the building as a heat sink -- 163 kWh of heat pump against a thermostat's 44, and
    # 5661 degree-hours of overheating, to earn 2.49 EUR in a week.
    #
    # Defaults are still 0.0 / 0.0, where this is identical to the previous ordering.
    import_price = (
        exogenous.price + r_conf.import_grid_fee_eur_per_kwh
    ) * (1.0 + r_conf.import_tax_rate)
    if r_conf.export_price_eur_per_kwh is None:
        export_price = EXPORT_PRICE_FRACTION * exogenous.price
    else:
        export_price = r_conf.export_price_eur_per_kwh

    buy_cost = import_w * import_price
    sell_revenue = export_w * export_price
    electricity_eur = (buy_cost - sell_revenue) * energy_factor_kwh_per_w

    # Comfort is charged on the temperature the action PRODUCED, not the one it
    # started from, so the penalty is attributable to the action taken.
    room_temps = next_state.thermal.T_vector[jnp.array(t_conf.room_air_indices)]
    if r_conf.comfort_penalize_overheating:
        excursion_c = jnp.maximum(
            0.0, jnp.abs(room_temps - t_conf.setpoint) - t_conf.comfort_band
        )
    else:
        # Cold side only -- for an action space that cannot cool. See
        # RewardConfig.comfort_penalize_overheating.
        excursion_c = jnp.maximum(
            0.0, (t_conf.setpoint - t_conf.comfort_band) - room_temps
        )
    hours = dt_seconds / 3600.0
    comfort_eur = r_conf.comfort_weight * jnp.sum(excursion_c ** 2) * hours

    # Both terms are scaled by price_weight together, so comfort_weight stays a
    # pure EUR/(degC^2.h) exchange rate and does not silently absorb the scaling.
    return (electricity_eur + comfort_eur) * r_conf.price_weight


@jit
def f_terminal_cost(
    final_state: SystemState,
    initial_state: SystemState,
    configs: tuple,
    exo_forecast_end: ExogenousData,
) -> Array:
    """Terminal SOC penalty corresponding to the QP terminal objective."""
    del initial_state, exo_forecast_end

    _, _, r_conf, _, _, _, _ = configs
    soc_error = final_state.battery.soc - TERMINAL_SOC_TARGET
    energy_error_kwh = soc_error * configs[1].capacity_kwh

    return (
        TERMINAL_SOC_WEIGHT_EUR_PER_KWH2
        * energy_error_kwh**2
        * r_conf.price_weight)


@jit
def f_trajectory_cost(
    action_sequence: SystemActions,
    initial_sim,
    exo_sequence: ExogenousData,
) -> Array:
    """Compatibility hook; the hard MPC trajectory cost is encoded in the QP."""
    del initial_sim, exo_sequence
    return jnp.zeros_like(jnp.sum(action_sequence.battery_power_w))


def build_battery_qp_static_data(
    *,
    N_horizon: int,
    b_conf: BatteryConfig,
    r_conf: RewardConfig,
    dt_seconds: float,
    export_price_fraction: float = EXPORT_PRICE_FRACTION,
    terminal_soc_target: float = TERMINAL_SOC_TARGET,
    terminal_soc_weight: float = TERMINAL_SOC_WEIGHT_EUR_PER_KWH2,
    tiny_regularization: float = TINY_QP_REGULARIZATION,
):
    """Build the fixed hard-MPC QP matrices and index slices once."""

    N = N_horizon
    n_vars = 4 * N
    n_rows = 6 * N + 1

    idx_charge = jnp.arange(N)
    idx_discharge = N + jnp.arange(N)
    idx_import = 2 * N + jnp.arange(N)
    idx_export = 3 * N + jnp.arange(N)

    row_grid = jnp.arange(N)

    row_bound_start = N
    row_charge_bound = row_bound_start + 4 * jnp.arange(N)
    row_discharge_bound = row_charge_bound + 1
    row_import_bound = row_charge_bound + 2
    row_export_bound = row_charge_bound + 3

    row_soc = 5 * N + jnp.arange(N + 1)

    # ------------------------------------------------------------
    # Objective: constant quadratic terms
    # ------------------------------------------------------------

    one_way_eff = jnp.sqrt(b_conf.efficiency)
    alpha_charge_soc_per_w = (
        dt_seconds / 3600000.0 * one_way_eff / b_conf.capacity_kwh
    )
    alpha_discharge_soc_per_w = (
        dt_seconds / 3600000.0 / one_way_eff / b_conf.capacity_kwh
    )

    terminal_soc_coeff = jnp.concatenate(
        [
            jnp.full((N,), alpha_charge_soc_per_w),
            jnp.full((N,), -alpha_discharge_soc_per_w),
            jnp.zeros((2 * N,)),
        ]
    )

    weighted_terminal = (
        terminal_soc_weight
        * (b_conf.capacity_kwh ** 2)
        * r_conf.price_weight
    )
    terminal_linear_scale = jnp.asarray(2.0 * weighted_terminal)

    Q = (
        tiny_regularization * jnp.eye(n_vars)
        + terminal_linear_scale
        * jnp.outer(terminal_soc_coeff, terminal_soc_coeff)
    )

    # ------------------------------------------------------------
    # Constraints: fixed A matrix
    # ------------------------------------------------------------

    A = jnp.zeros((n_rows, n_vars))

    # Grid balance:
    # grid_import - grid_export - p_charge + p_discharge = load - pv
    A = A.at[row_grid, idx_import].set(1.0)
    A = A.at[row_grid, idx_export].set(-1.0)
    A = A.at[row_grid, idx_charge].set(-1.0)
    A = A.at[row_grid, idx_discharge].set(1.0)

    # Power and grid variable bounds.
    A = A.at[row_charge_bound, idx_charge].set(1.0)
    A = A.at[row_discharge_bound, idx_discharge].set(1.0)
    A = A.at[row_import_bound, idx_import].set(1.0)
    A = A.at[row_export_bound, idx_export].set(1.0)

    # Cumulative battery energy bounds. Row k includes actions i < k, so row
    # zero intentionally remains all zeros to preserve the old formulation.
    dt_hours = dt_seconds / 3600.0
    charge_wh_per_w = dt_hours * one_way_eff
    discharge_wh_per_w = dt_hours / one_way_eff
    cumulative = jnp.tril(jnp.ones((N + 1, N)), k=-1)
    A_soc = jnp.concatenate(
        [
            cumulative * charge_wh_per_w,
            -cumulative * discharge_wh_per_w,
            jnp.zeros((N + 1, 2 * N)),
        ],
        axis=1,
    )
    A = A.at[row_soc, :].set(A_soc)

    # ------------------------------------------------------------
    # Constraint bounds: constant baseline
    # ------------------------------------------------------------

    l_base = jnp.zeros((n_rows,))
    u_base = jnp.zeros((n_rows,))

    l_base = l_base.at[row_charge_bound].set(0.0)
    u_base = u_base.at[row_charge_bound].set(b_conf.max_power_w)

    l_base = l_base.at[row_discharge_bound].set(0.0)
    u_base = u_base.at[row_discharge_bound].set(b_conf.max_power_w)

    l_base = l_base.at[row_import_bound].set(0.0)
    l_base = l_base.at[row_export_bound].set(0.0)

    energy_factor_kwh_per_w = dt_seconds / 3600000.0

    # Import and export are priced separately, matching f_stage_cost exactly
    # -- see RewardConfig. Import cost is an AFFINE function of wholesale
    # price (retail markup + a flat grid fee), so it splits into a
    # price-proportional scale and a constant per-step offset. Export is
    # either a flat feed-in rate (export_price_eur_per_kwh set) -- a pure
    # offset, no price dependence -- or, if unset, the legacy behaviour of a
    # fixed fraction of wholesale price, which is why export also carries a
    # price-proportional scale term even though only one of the two paths is
    # ever nonzero.
    price_weight = r_conf.price_weight
    import_price_scale = jnp.asarray(
        (1.0 + r_conf.import_tax_rate) * energy_factor_kwh_per_w * price_weight
    )
    # (price + fee) * (1 + tax) = price*(1+tax) + fee*(1+tax), so the fee offset carries
    # the tax factor too. Must track f_stage_cost exactly or MPC optimizes a different
    # objective than it is scored on.
    import_fee_offset = jnp.asarray(
        r_conf.import_grid_fee_eur_per_kwh * (1.0 + r_conf.import_tax_rate)
        * energy_factor_kwh_per_w * price_weight
    )
    if r_conf.export_price_eur_per_kwh is None:
        export_price_scale = jnp.asarray(
            export_price_fraction * energy_factor_kwh_per_w * price_weight
        )
        export_price_offset = jnp.asarray(0.0)
    else:
        export_price_scale = jnp.asarray(0.0)
        export_price_offset = jnp.asarray(
            r_conf.export_price_eur_per_kwh * energy_factor_kwh_per_w * price_weight
        )

    return BatteryQPStaticData(
        Q=Q,
        A=A,
        l_base=l_base,
        u_base=u_base,
        terminal_soc_coeff=terminal_soc_coeff,
        idx_import=idx_import,
        idx_export=idx_export,
        row_grid=row_grid,
        row_import_bound=row_import_bound,
        row_export_bound=row_export_bound,
        row_soc=row_soc,
        capacity_wh=jnp.asarray(b_conf.capacity_kwh * 1000.0),
        max_power_w=jnp.asarray(b_conf.max_power_w),
        import_price_scale=import_price_scale,
        import_fee_offset=import_fee_offset,
        export_price_scale=export_price_scale,
        export_price_offset=export_price_offset,
        terminal_linear_scale=terminal_linear_scale,
        terminal_soc_target=jnp.asarray(terminal_soc_target),
    )


@jit
def update_battery_qp_dynamic_vectors(
    qp_static: BatteryQPStaticData,
    *,
    current_soc,
    load_w,
    pv_w,
    price,
):
    """Update only the hard-MPC QP data that changes at each control step."""

    rhs = load_w - pv_w
    grid_power_upper_w = (
        jnp.maximum(load_w, pv_w)
        + qp_static.max_power_w
        + 1.0
    )

    c = (
        qp_static.terminal_linear_scale
        * (current_soc - qp_static.terminal_soc_target)
        * qp_static.terminal_soc_coeff
    )
    c = c.at[qp_static.idx_import].set(
        price * qp_static.import_price_scale + qp_static.import_fee_offset
    )
    c = c.at[qp_static.idx_export].set(
        -(price * qp_static.export_price_scale + qp_static.export_price_offset)
    )

    current_energy_wh = current_soc * qp_static.capacity_wh

    l = qp_static.l_base
    u = qp_static.u_base

    l = l.at[qp_static.row_grid].set(rhs)
    u = u.at[qp_static.row_grid].set(rhs)

    u = u.at[qp_static.row_import_bound].set(grid_power_upper_w)
    u = u.at[qp_static.row_export_bound].set(grid_power_upper_w)

    l = l.at[qp_static.row_soc].set(-current_energy_wh)
    u = u.at[qp_static.row_soc].set(
        qp_static.capacity_wh - current_energy_wh
    )

    return c, l, u


def build_battery_qp_matrices(
    *,
    N_horizon: int,
    current_soc,
    load_w,
    pv_w,
    price,
    b_conf: BatteryConfig,
    r_conf: RewardConfig,
    dt_seconds: float,
    export_price_fraction: float = EXPORT_PRICE_FRACTION,
    terminal_soc_target: float = TERMINAL_SOC_TARGET,
    terminal_soc_weight: float = TERMINAL_SOC_WEIGHT_EUR_PER_KWH2,
    tiny_regularization: float = TINY_QP_REGULARIZATION,
):
    """
    Build QP matrices for hard-constrained battery MPC.

    This compatibility wrapper preserves the original return signature while
    sharing the optimized static/dynamic construction used by the solver.
    """

    index_data = {
        "idx_charge_start": 0,
        "idx_discharge_start": N_horizon,
        "idx_import_start": 2 * N_horizon,
        "idx_export_start": 3 * N_horizon,
        "n_vars": 4 * N_horizon,
    }

    qp_static = build_battery_qp_static_data(
        N_horizon=N_horizon,
        b_conf=b_conf,
        r_conf=r_conf,
        dt_seconds=dt_seconds,
        export_price_fraction=export_price_fraction,
        terminal_soc_target=terminal_soc_target,
        terminal_soc_weight=terminal_soc_weight,
        tiny_regularization=tiny_regularization,
    )
    c, l, u = update_battery_qp_dynamic_vectors(
        qp_static,
        current_soc=current_soc,
        load_w=load_w,
        pv_w=pv_w,
        price=price,
    )

    return qp_static.Q, c, qp_static.A, l, u, index_data
