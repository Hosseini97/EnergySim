import os

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from energysim.sim.simulator import JAXSimulator
from energysim.control.mpc_solver import JAX_MPC_Solver
from energysim.core.data.dataset import SimulationDataset

from energysim.utils import objectives

from energysim.core.shared.data_structs import (
    BatteryConfig,
    RewardConfig,
    HeatPumpConfig,
    AirConditionerConfig,
    ThermalStorageConfig,
    PVConfig,
    SystemActions,
)

import sample_data_generator
from build_my_house import create_2_room_house


def stack_pytree(list_of_trees):
    return jax.tree.map(lambda *args: jnp.stack(args), *list_of_trees)


def build_simulator(
    dt_seconds: float,
    initial_battery_soc: float = 0.0,
) -> JAXSimulator:
    return JAXSimulator(
        dt_seconds=dt_seconds,
        t_config=create_2_room_house(),
        r_config=RewardConfig(
            price_weight=10.0,
            comfort_weight=50.0,
        ),
        b_config=BatteryConfig(),
        hp_config=HeatPumpConfig(),
        ac_config=AirConditionerConfig(),
        ts_config=ThermalStorageConfig(),
        pv_config=PVConfig(model_type="passthrough"),
        initial_battery_soc=initial_battery_soc,
    )


def build_zero_action(n_rooms: int) -> SystemActions:
    return SystemActions(
        battery_power_w=jnp.array(0.0),
        heat_pump_power_w=jnp.zeros((n_rooms,)),
        ac_power_w=jnp.zeros((n_rooms,)),
        storage_discharge_w=jnp.zeros((n_rooms,)),
    )


def step_cost_eur(
    action: SystemActions,
    outputs,
    exo,
    dt_seconds,
):
    """
    Electricity bill calculation matching objectives_new_clean.f_stage_cost.

    No battery clipping is applied here.
    Battery power is used exactly as returned by the controller.
    """

    uncontrollable_load_w = (
        exo.base_load_w
        + exo.ev_charger_load_w
        + exo.dishwasher_load_w
        + exo.clothes_dryer_load_w
        + exo.water_heater_load_w
        + exo.cooking_load_w
    )

    controllable_electric_power_w = (
        jnp.sum(outputs.hp.electrical_power_w)
        + jnp.sum(outputs.ac.electrical_power_w)
    )

    net_grid_w = (
        uncontrollable_load_w
        + controllable_electric_power_w
        + action.battery_power_w
        - outputs.pv.pv_generation_w
    )

    import_w = jnp.maximum(net_grid_w, 0.0)
    export_w = jnp.maximum(-net_grid_w, 0.0)

    import_kwh = import_w * dt_seconds / 3600000.0
    export_kwh = export_w * dt_seconds / 3600000.0

    buy_price = exo.price
    sell_price = objectives.EXPORT_PRICE_FRACTION * exo.price

    cost_eur = import_kwh * buy_price - export_kwh * sell_price

    return float(cost_eur), float(net_grid_w)


def make_controller(
    case_name: str,
    horizon: int,
    sim_template: JAXSimulator,
    n_rooms: int,
):
    if case_name == "hard_mpc":
        hard_phys_min, hard_phys_max = objectives.build_battery_only_bounds(
            N_horizon=horizon,
            n_rooms=n_rooms,
            b_conf=sim_template.battery.config,
        )

        return JAX_MPC_Solver(
            horizon,
            sim_template,
            stage_cost_fn=objectives.f_stage_cost,
            terminal_cost_fn=objectives.f_terminal_cost,
            trajectory_cost_fn=objectives.f_trajectory_cost,
            phys_min=hard_phys_min,
            phys_max=hard_phys_max,
        )

    if case_name == "no_control":
        return None

    raise ValueError(f"Unknown case_name: {case_name}")


def simulate_case(
    case_name,
    sim_template,
    all_exo,
    horizon,
    eval_steps=None,
):
    sim = sim_template.reset()

    n_rooms = len(sim_template.thermal.config.room_air_indices)
    ac_max_room_w = sim_template.ac.config.max_electrical_power_w / n_rooms
    storage_max_room_w = sim_template.storage.config.max_discharge_w / n_rooms

    controller = make_controller(
        case_name=case_name,
        horizon=horizon,
        sim_template=sim_template,
        n_rooms=n_rooms,
    )

    rows = []

    max_steps = all_exo.ambient_temp.shape[0] - horizon
    rollout_steps = max_steps if eval_steps is None else min(max_steps, int(eval_steps))

    for idx in range(rollout_steps):
        exo_forecast = jax.tree.map(
            lambda arr: jax.lax.dynamic_slice_in_dim(arr, idx, horizon),
            all_exo,
        )

        exo_now = jax.tree.map(
            lambda arr: arr[idx],
            all_exo,
        )

        if controller is None:
            action = build_zero_action(n_rooms)
        else:
            action = controller.solve(
                current_sim=sim,
                exo_forecast=exo_forecast,
                warm_start_norm_actions=None,
            )

        # Clean hard-MPC path:
        # apply the action directly, without battery clipping.
        next_sim, outputs = sim.step(action, exo_now)

        if case_name == "hard_mpc":
            stage_obj = objectives.f_stage_cost(
                current_state=sim.state,
                next_state=next_sim.state,
                actions=action,
                outputs=outputs,
                exogenous=exo_now,
                configs=sim.configs,
                dt_seconds=sim.dt_seconds,
            )
        else:
            stage_obj = objectives.f_stage_cost(
                current_state=sim.state,
                next_state=next_sim.state,
                actions=action,
                outputs=outputs,
                exogenous=exo_now,
                configs=sim.configs,
                dt_seconds=sim.dt_seconds,
            )

        elec_eur, net_grid_w = step_cost_eur(
            action=action,
            outputs=outputs,
            exo=exo_now,
            dt_seconds=sim.dt_seconds,
        )

        room_idx = np.array(sim_template.thermal.config.room_air_indices)
        room_temps = np.array(sim.state.thermal.T_vector)[room_idx]

        hp_arr = np.array(action.heat_pump_power_w, dtype=float)
        ac_arr = np.array(action.ac_power_w, dtype=float)
        stor_arr = np.array(action.storage_discharge_w, dtype=float)

        uncontrollable_load_w = float(
            exo_now.base_load_w
            + exo_now.ev_charger_load_w
            + exo_now.dishwasher_load_w
            + exo_now.clothes_dryer_load_w
            + exo_now.water_heater_load_w
            + exo_now.cooking_load_w
        )

        row = {
            "case": case_name,
            "step": idx,
            "electricity_cost_eur": elec_eur,
            "objective_cost": float(stage_obj),
            "net_grid_w": net_grid_w,
            "uncontrollable_load_w": uncontrollable_load_w,
            "ambient_temp_c": float(exo_now.ambient_temp),
            "price_eur_per_kwh": float(exo_now.price),
            "base_load_w": float(exo_now.base_load_w),
            "pv_generation_w": float(outputs.pv.pv_generation_w),
            "solar_irradiance_w_m2": float(exo_now.solar_irradiance_w_m2),
            "tank_mean_c": float(np.mean(np.array(sim.state.storage.temperatures_c))),
            "battery_soc": float(sim.state.battery.soc),
            "battery_soc_next": float(next_sim.state.battery.soc),
            "battery_w": float(action.battery_power_w),
            "hp_w": float(np.sum(hp_arr)),
            "ac_w": float(np.sum(ac_arr)),
            "stor_w": float(np.sum(stor_arr)),
            "ac_max_room_w": float(ac_max_room_w),
            "storage_max_room_w": float(storage_max_room_w),
        }

        for r in range(n_rooms):
            row[f"t_room_{r}"] = float(room_temps[r])
            row[f"hp_w_{r}"] = float(hp_arr[r])
            row[f"ac_w_{r}"] = float(ac_arr[r])
            row[f"stor_w_{r}"] = float(stor_arr[r])

        rows.append(row)

        sim = next_sim

    return pd.DataFrame(rows)


def summarize(df, dt_seconds):
    dt_h = dt_seconds / 3600.0

    import_kwh = (np.maximum(df["net_grid_w"], 0.0).sum() * dt_h) / 1000.0
    export_kwh = (np.maximum(-df["net_grid_w"], 0.0).sum() * dt_h) / 1000.0

    both_in_band = (
        (
            (df["t_room_0"] >= 20.0)
            & (df["t_room_0"] <= 22.0)
            & (df["t_room_1"] >= 20.0)
            & (df["t_room_1"] <= 22.0)
        ).mean()
    )

    avg_abs_dev = (
        (
            np.abs(df["t_room_0"] - 21.0)
            + np.abs(df["t_room_1"] - 21.0)
        )
        / 2.0
    ).mean()

    return {
        "total_electricity_cost_eur": float(df["electricity_cost_eur"].sum()),
        "total_stage_objective": float(df["objective_cost"].sum()),
        "grid_import_kwh": float(import_kwh),
        "grid_export_kwh": float(export_kwh),
        "peak_grid_import_w": float(np.maximum(df["net_grid_w"], 0.0).max()),
        "min_battery_soc": float(df["battery_soc"].min()),
        "max_battery_soc": float(df["battery_soc"].max()),
        "min_battery_soc_next": float(df["battery_soc_next"].min()),
        "max_battery_soc_next": float(df["battery_soc_next"].max()),
        "avg_abs_room_temp_dev_C": float(avg_abs_dev),
        "both_rooms_within_20_22_frac": float(both_in_band),
    }


def print_violation_diagnostics(
    df,
    case_name,
    n_rooms,
    setpoint=21.0,
    band=1.0,
):
    low = setpoint - band
    high = setpoint + band

    ac_max_room = float(df["ac_max_room_w"].iloc[0])
    storage_max_room = float(df["storage_max_room_w"].iloc[0])

    print(f"\n--- Comfort Violation Diagnostics: {case_name} ---")

    for r in range(n_rooms):
        t = df[f"t_room_{r}"].to_numpy()
        ac = df[f"ac_w_{r}"].to_numpy()
        stor = df[f"stor_w_{r}"].to_numpy()
        tank = df["tank_mean_c"].to_numpy()
        amb = df["ambient_temp_c"].to_numpy()

        hot = t > high
        cold = t < low

        hot_steps = int(hot.sum())
        cold_steps = int(cold.sum())

        if hot_steps == 0 and cold_steps == 0:
            print(f"room_{r}: keine Verletzungen.")
            continue

        hot_no_ac = int(np.sum(hot & (ac <= 1e-3)))
        hot_ac_sat = int(np.sum(hot & (ac >= 0.99 * ac_max_room)))
        hot_partial = hot_steps - hot_no_ac - hot_ac_sat

        cold_no_heat = int(np.sum(cold & (stor <= 1e-3)))
        cold_heat_sat = int(np.sum(cold & (stor >= 0.99 * storage_max_room)))
        cold_partial = cold_steps - cold_no_heat - cold_heat_sat
        cold_low_tank = int(np.sum(cold & (tank < 25.0)))

        print(f"room_{r}:")
        print(
            f"  hot_steps={hot_steps}, "
            f"no_cooling={hot_no_ac}, "
            f"cooling_sat={hot_ac_sat}, "
            f"cooling_partial={hot_partial}"
        )
        print(
            f"  cold_steps={cold_steps}, "
            f"no_heating={cold_no_heat}, "
            f"heating_sat={cold_heat_sat}, "
            f"heating_partial={cold_partial}, "
            f"cold_with_low_tank(<25C)={cold_low_tank}"
        )

        if hot_steps > 0:
            print(
                f"  hot_phase_mean_ambient={float(np.mean(amb[hot])):.2f}C, "
                f"hot_phase_mean_tank={float(np.mean(tank[hot])):.2f}C"
            )

        if cold_steps > 0:
            print(
                f"  cold_phase_mean_ambient={float(np.mean(amb[cold])):.2f}C, "
                f"cold_phase_mean_tank={float(np.mean(tank[cold])):.2f}C"
            )


def plot_battery_vs_price(
    hard_df,
    dt_seconds,
    active_threshold_w=50.0,
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Battery Activity vs Electricity Price")

    price_ax, soc_ax = axes
    time_h = hard_df["step"].to_numpy(dtype=float) * dt_seconds / 3600.0
    price = hard_df["price_eur_per_kwh"].to_numpy(dtype=float)
    battery = hard_df["battery_w"].to_numpy(dtype=float)

    active = np.abs(battery) > active_threshold_w
    switch_on = active & np.r_[True, ~active[:-1]]

    price_ax.plot(
        time_h,
        price,
        color="black",
        linewidth=1.8,
        label="Price",
    )

    price_ax.fill_between(
        time_h,
        0.0,
        1.0,
        where=active,
        step="post",
        color="tab:green",
        alpha=0.10,
        transform=price_ax.get_xaxis_transform(),
        label="Battery active",
    )

    price_ax.scatter(
        time_h[switch_on],
        price[switch_on],
        color="tab:green",
        s=28,
        zorder=4,
        label="Battery switch-on",
    )

    price_ax.set_ylabel("Price [EUR/kWh]")
    price_ax.grid(True, alpha=0.25)
    price_ax.set_title(
        f"Hard-Constrained MPC | active={active.mean():.1%}, "
        f"switch-ons={int(switch_on.sum())}"
    )

    battery_ax = price_ax.twinx()
    battery_ax.step(
        time_h,
        battery,
        where="post",
        color="tab:blue",
        linewidth=1.3,
        label="Battery activity",
    )

    battery_lim = max(
        active_threshold_w,
        float(np.abs(battery).max()),
    ) * 1.05

    battery_ax.axhline(
        0.0,
        color="0.5",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    battery_ax.set_ylim(-battery_lim, battery_lim)
    battery_ax.set_ylabel("Battery [W]")

    h1, l1 = price_ax.get_legend_handles_labels()
    h2, l2 = battery_ax.get_legend_handles_labels()
    price_ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    soc_ax.plot(
        time_h,
        hard_df["battery_soc"],
        color="tab:orange",
        linewidth=1.5,
        label="Hard-Constrained MPC",
    )

    soc_ax.set_title("Battery State of Charge")
    soc_ax.set_ylabel("SoC [-]")
    soc_ax.set_xlabel("Simulated time [h]")
    soc_ax.set_ylim(0.0, 1.0)
    soc_ax.grid(True, alpha=0.25)
    soc_ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("examples/mpc_battery_vs_price.svg", bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_cost(
    no_control_df,
    hard_df,
    dt_seconds,
):
    fig, ax = plt.subplots(figsize=(12, 4))

    for df, label, color in [
        (no_control_df, "No Control", "0.5"),
        (hard_df, "Hard-Constrained MPC", "tab:orange"),
    ]:
        time_h = df["step"].to_numpy(dtype=float) * dt_seconds / 3600.0

        ax.plot(
            time_h,
            df["electricity_cost_eur"].cumsum(),
            linewidth=1.8,
            color=color,
            label=label,
        )

    ax.set_title("Cumulative Electricity Cost")
    ax.set_xlabel("Simulated time [h]")
    ax.set_ylabel("Cumulative cost [EUR]")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig("examples/mpc_cumulative_cost.svg", bbox_inches="tight")
    plt.close(fig)


def plot_comfort_vs_band(
    no_control_df,
    hard_df,
    dt_seconds,
    setpoint=21.0,
    band=1.0,
):
    room_cols = [
        c for c in hard_df.columns
        if c.startswith("t_room_")
    ]

    fig, axes = plt.subplots(
        len(room_cols),
        1,
        figsize=(12, 3 + 2 * len(room_cols)),
        sharex=True,
    )

    axes = np.atleast_1d(axes)

    low = setpoint - band
    high = setpoint + band

    for r, ax in enumerate(axes):
        time_h = hard_df["step"].to_numpy(dtype=float) * dt_seconds / 3600.0
        col = f"t_room_{r}"

        ax.fill_between(
            time_h,
            low,
            high,
            color="tab:green",
            alpha=0.10,
            label="Comfort band",
        )

        ax.plot(
            time_h,
            no_control_df[col],
            color="0.5",
            linewidth=1.2,
            linestyle="--",
            label="No Control",
        )

        ax.plot(
            time_h,
            hard_df[col],
            color="tab:orange",
            linewidth=1.4,
            label="Hard-Constrained MPC",
        )

        ax.axhline(
            setpoint,
            color="0.5",
            linestyle=":",
            linewidth=0.8,
        )

        ax.set_ylabel(f"Room {r} [C]")
        ax.set_title(f"Room {r} Temperature vs Comfort Band")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Simulated time [h]")

    fig.tight_layout()
    fig.savefig("examples/mpc_comfort_vs_band.svg", bbox_inches="tight")
    plt.close(fig)


def plot_no_control_reason(
    no_control_df,
    hard_df,
    dt_seconds,
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    time_h = no_control_df["step"].to_numpy(dtype=float) * dt_seconds / 3600.0

    load = no_control_df["uncontrollable_load_w"].to_numpy(dtype=float)
    pv = no_control_df["pv_generation_w"].to_numpy(dtype=float)
    surplus = pv > load

    axes[0].plot(
        time_h,
        load,
        color="tab:red",
        linewidth=1.5,
        label="No-control load",
    )

    axes[0].plot(
        time_h,
        pv,
        color="tab:green",
        linewidth=1.5,
        label="PV generation",
    )

    axes[0].fill_between(
        time_h,
        load,
        pv,
        where=surplus,
        color="tab:green",
        alpha=0.12,
        label="PV surplus sold",
    )

    axes[0].set_title("No Control: PV Can Exceed Uncontrollable Load")
    axes[0].set_ylabel("Power [W]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=9)

    for df, label, color in [
        (no_control_df, "No Control", "0.5"),
        (hard_df, "Hard-Constrained MPC", "tab:orange"),
    ]:
        control_effect_w = (
            df["net_grid_w"].to_numpy(dtype=float)
            - df["uncontrollable_load_w"].to_numpy(dtype=float)
            + df["pv_generation_w"].to_numpy(dtype=float)
        )

        axes[1].plot(
            time_h,
            control_effect_w,
            color=color,
            linewidth=1.5,
            label=label,
        )

    axes[1].axhline(
        0.0,
        color="0.5",
        linestyle="--",
        linewidth=0.8,
    )

    axes[1].set_title("Grid Impact of Controllable Actions")
    axes[1].set_xlabel("Simulated time [h]")
    axes[1].set_ylabel("HP + AC + battery [W]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig("examples/mpc_no_control_reason.svg", bbox_inches="tight")
    plt.close(fig)


def plot_pv_load_battery_grid(hard_df, dt_seconds):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.suptitle("PV, Load, Battery and Grid Power")

    time_h = hard_df["step"].to_numpy(dtype=float) * dt_seconds / 3600.0

    load = hard_df["uncontrollable_load_w"].to_numpy(dtype=float)
    pv = hard_df["pv_generation_w"].to_numpy(dtype=float)
    battery = hard_df["battery_w"].to_numpy(dtype=float)
    net_grid = hard_df["net_grid_w"].to_numpy(dtype=float)

    ax.plot(time_h, load, label="Uncontrollable load [W]", linewidth=1.4)
    ax.plot(time_h, pv, label="PV generation [W]", linewidth=1.4)
    ax.step(time_h, battery, where="post", label="Battery power [W]", linewidth=1.2)
    ax.plot(time_h, net_grid, label="Net grid power [W]", linewidth=1.2)

    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_title("Hard-Constrained MPC")
    ax.set_ylabel("Power [W]")
    ax.set_xlabel("Simulated time [h]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("examples/mpc_pv_load_battery_grid.svg", bbox_inches="tight")
    plt.close(fig)


def run_comparison():
    seed = 42
    n_days = 7
    eval_steps = None
    horizon = 96  # 24h horizon with 15min steps

    os.makedirs("examples", exist_ok=True)

    np.random.seed(seed)

    sample_data_generator.create_sample_data(n_days=n_days)

    dt = sample_data_generator.DT_SECONDS
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, dt)

    all_exo = stack_pytree(
        [dataset[i] for i in range(len(dataset))]
    )

    sim_template = build_simulator(dt)

    no_control_df = simulate_case(
        "no_control",
        sim_template,
        all_exo,
        horizon,
        eval_steps=eval_steps,
    )

    hard_df = simulate_case(
        "hard_mpc",
        sim_template,
        all_exo,
        horizon,
        eval_steps=eval_steps,
    )

    all_df = pd.concat(
        [no_control_df, hard_df],
        ignore_index=True,
    )

    all_df.to_csv(
        "examples/mpc_comparison_clean_no_clipping.csv",
        index=False,
    )

    stats = {
        "no_control": summarize(no_control_df, dt),
        "hard_mpc": summarize(hard_df, dt),
    }

    print("=== Comparison On Same Perfect-Forecast Trace ===")

    keys = list(stats["no_control"].keys())

    for key in keys:
        n = stats["no_control"][key]
        h = stats["hard_mpc"][key]

        print(f"{key}:")
        print(f"  no_control   = {n:.6f}")
        print(f"  hard_mpc     = {h:.6f} (delta vs no_control = {h - n:+.6f})")

    n_rooms = len(sim_template.thermal.config.room_air_indices)

    print_violation_diagnostics(
        hard_df,
        "hard_mpc",
        n_rooms=n_rooms,
    )

    plot_battery_vs_price(
        hard_df,
        dt,
    )

    plot_cumulative_cost(
        no_control_df,
        hard_df,
        dt,
    )

    plot_comfort_vs_band(
        no_control_df,
        hard_df,
        dt,
    )

    plot_no_control_reason(
        no_control_df,
        hard_df,
        dt,
    )

    plot_pv_load_battery_grid(hard_df, dt)

    print("\nSaved detailed step-by-step results to:")
    print("examples/mpc_comparison_clean_no_clipping.csv")
    print("Saved battery/price plot to examples/mpc_battery_vs_price.svg")
    print("Saved cumulative cost plot to examples/mpc_cumulative_cost.svg")
    print("Saved comfort plot to examples/mpc_comfort_vs_band.svg")
    print("Saved no-control explanation plot to examples/mpc_no_control_reason.svg")


if __name__ == "__main__":
    run_comparison()
