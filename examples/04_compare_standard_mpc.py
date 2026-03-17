import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from energysim.sim.simulator import JAXSimulator
from energysim.control.mpc_solver import JAX_MPC_Solver, MPC_solver_NewMPC
from energysim.core.data.dataset import SimulationDataset
from energysim.utils.objectives import f_cost_step, _effective_battery_grid_power_w
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig,
    ThermalStorageConfig, PVConfig, SystemActions
)
import sample_data_generator
from build_my_house import create_2_room_house


def stack_pytree(list_of_trees):
    return jax.tree.map(lambda *args: jnp.stack(args), *list_of_trees)


def build_simulator(dt_seconds: float) -> JAXSimulator:
    return JAXSimulator(
        dt_seconds=dt_seconds,
        t_config=create_2_room_house(),
        r_config=RewardConfig(price_weight=10.0, comfort_weight=50.0),
        b_config=BatteryConfig(),
        hp_config=HeatPumpConfig(),
        ac_config=AirConditionerConfig(),
        ts_config=ThermalStorageConfig(),
        pv_config=PVConfig(),
    )


def step_cost_eur(next_state, action, outputs, exo, b_conf, dt_seconds):
    uncontrollable_load_w = (
        exo.base_load_w + exo.ev_charger_load_w + exo.dishwasher_load_w +
        exo.clothes_dryer_load_w + exo.water_heater_load_w + exo.cooking_load_w
    )
    effective_battery = _effective_battery_grid_power_w(
        state=next_state,
        requested_battery_power_w=action.battery_power_w,
        b_conf=b_conf,
        dt_seconds=dt_seconds,
    )
    net_grid_w = (
        uncontrollable_load_w
        + effective_battery
        + jnp.sum(outputs.hp.electrical_power_w)
        + jnp.sum(outputs.ac.electrical_power_w)
        - outputs.pv.pv_generation_w
    )
    net_grid_kwh = (net_grid_w * dt_seconds) / 3600000.0
    cost_eur = jnp.where(
        net_grid_kwh > 0.0,
        net_grid_kwh * exo.price,
        net_grid_kwh * (exo.price * 0.20),
    )
    return float(cost_eur), float(net_grid_w)


def simulate_case(case_name, sim_template, all_exo, horizon, eval_steps=None):
    sim = sim_template.reset()
    n_rooms = len(sim_template.thermal.config.room_air_indices)
    ac_max_room_w = sim_template.ac.config.max_electrical_power_w / n_rooms
    storage_max_room_w = sim_template.storage.config.max_discharge_w / n_rooms

    if case_name == "standard_mpc":
        controller = MPC_solver_NewMPC(horizon, sim_template)
    elif case_name == "hard_mpc":
        controller = JAX_MPC_Solver(horizon, sim_template)
    else:
        controller = None

    rows = []
    max_steps = all_exo.ambient_temp.shape[0] - horizon
    rollout_steps = max_steps if eval_steps is None else min(max_steps, int(eval_steps))
    for idx in range(rollout_steps):
        exo_forecast = jax.tree.map(
            lambda arr: jax.lax.dynamic_slice_in_dim(arr, idx, horizon), all_exo
        )
        exo_now = jax.tree.map(lambda arr: arr[idx], all_exo)

        if controller is None:
            action = SystemActions(
                battery_power_w=jnp.array(0.0),
                heat_pump_power_w=jnp.zeros((n_rooms,)),
                ac_power_w=jnp.zeros((n_rooms,)),
                storage_discharge_w=jnp.zeros((n_rooms,)),
            )
        else:
            action = controller.solve(
                current_sim=sim,
                exo_forecast=exo_forecast,
                warm_start_norm_actions=None
            )

        next_sim, outputs = sim.step(action, exo_now)

        stage_obj = f_cost_step(
            state=next_sim.state,
            actions=action,
            outputs=outputs,
            exogenous=exo_now,
            configs=sim.configs,
            dt_seconds=sim.dt_seconds,
        )
        elec_eur, net_grid_w = step_cost_eur(
            next_state=next_sim.state,
            action=action,
            outputs=outputs,
            exo=exo_now,
            b_conf=sim.battery.config,
            dt_seconds=sim.dt_seconds,
        )

        room_idx = np.array(sim_template.thermal.config.room_air_indices)
        room_temps = np.array(sim.state.thermal.T_vector)[room_idx]
        hp_arr = np.array(action.heat_pump_power_w, dtype=float)
        ac_arr = np.array(action.ac_power_w, dtype=float)
        stor_arr = np.array(action.storage_discharge_w, dtype=float)
        row = {
            "case": case_name,
            "step": idx,
            "electricity_cost_eur": elec_eur,
            "objective_cost": float(stage_obj),
            "net_grid_w": net_grid_w,
            "ambient_temp_c": float(exo_now.ambient_temp),
            "price_eur_per_kwh": float(exo_now.price),
            "base_load_w": float(exo_now.base_load_w),
            "solar_irradiance_w_m2": float(exo_now.solar_irradiance_w_m2),
            "tank_mean_c": float(np.mean(np.array(sim.state.storage.temperatures_c))),
            "battery_soc": float(sim.state.battery.soc),
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

        rows.append(
            row
        )

        sim = next_sim

    return pd.DataFrame(rows)


def summarize(df, dt_seconds):
    dt_h = dt_seconds / 3600.0
    import_kwh = (np.maximum(df["net_grid_w"], 0.0).sum() * dt_h) / 1000.0
    export_kwh = (np.maximum(-df["net_grid_w"], 0.0).sum() * dt_h) / 1000.0
    both_in_band = (
        (
            (df["t_room_0"] >= 20.0) & (df["t_room_0"] <= 22.0) &
            (df["t_room_1"] >= 20.0) & (df["t_room_1"] <= 22.0)
        ).mean()
    )
    avg_abs_dev = ((np.abs(df["t_room_0"] - 21.0) + np.abs(df["t_room_1"] - 21.0)) / 2.0).mean()
    return {
        "total_electricity_cost_eur": float(df["electricity_cost_eur"].sum()),
        "total_stage_objective": float(df["objective_cost"].sum()),
        "grid_import_kwh": float(import_kwh),
        "grid_export_kwh": float(export_kwh),
        "peak_grid_import_w": float(np.maximum(df["net_grid_w"], 0.0).max()),
        "avg_abs_room_temp_dev_C": float(avg_abs_dev),
        "both_rooms_within_20_22_frac": float(both_in_band),
    }


def print_violation_diagnostics(df, case_name, n_rooms, setpoint=21.0, band=1.0):
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
            f"  hot_steps={hot_steps}, no_cooling={hot_no_ac}, cooling_sat={hot_ac_sat}, cooling_partial={hot_partial}"
        )
        print(
            f"  cold_steps={cold_steps}, no_heating={cold_no_heat}, heating_sat={cold_heat_sat}, heating_partial={cold_partial}, cold_with_low_tank(<25C)={cold_low_tank}"
        )
        if hot_steps > 0:
            print(
                f"  hot_phase_mean_ambient={float(np.mean(amb[hot])):.2f}C, hot_phase_mean_tank={float(np.mean(tank[hot])):.2f}C"
            )
        if cold_steps > 0:
            print(
                f"  cold_phase_mean_ambient={float(np.mean(amb[cold])):.2f}C, cold_phase_mean_tank={float(np.mean(tank[cold])):.2f}C"
            )


def run_comparison():
    # --- 1. Setup ---
    seed = 42
    n_days = 7
    eval_steps = None
    horizon = 24

    np.random.seed(seed)
    sample_data_generator.create_sample_data(n_days=n_days)
    dt = sample_data_generator.DT_SECONDS
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, dt)
    all_exo = stack_pytree([dataset[i] for i in range(len(dataset))])

    sim_template = build_simulator(dt)

    no_control_df = simulate_case("no_control", sim_template, all_exo, horizon, eval_steps=eval_steps)
    standard_df = simulate_case("standard_mpc", sim_template, all_exo, horizon, eval_steps=eval_steps)
    hard_df = simulate_case("hard_mpc", sim_template, all_exo, horizon, eval_steps=eval_steps)

    all_df = pd.concat([no_control_df, standard_df, hard_df], ignore_index=True)
    all_df.to_csv("examples/mpc_comparison_same_data.csv", index=False)

    stats = {
        "no_control": summarize(no_control_df, dt),
        "standard_mpc": summarize(standard_df, dt),
        "hard_mpc": summarize(hard_df, dt),
    }

    print("=== Comparison On Same Perfect-Forecast Trace ===")
    keys = list(stats["no_control"].keys())
    for key in keys:
        n = stats["no_control"][key]
        s = stats["standard_mpc"][key]
        h = stats["hard_mpc"][key]
        print(f"{key}:")
        print(f"  no_control   = {n:.6f}")
        print(f"  standard_mpc = {s:.6f} (delta vs no_control = {s - n:+.6f})")
        print(f"  hard_mpc     = {h:.6f} (delta vs no_control = {h - n:+.6f})")

    n_rooms = len(sim_template.thermal.config.room_air_indices)
    print_violation_diagnostics(standard_df, "standard_mpc", n_rooms=n_rooms)
    print_violation_diagnostics(hard_df, "hard_mpc", n_rooms=n_rooms)

    print("\nSaved detailed step-by-step results to examples/mpc_comparison_same_data.csv")


if __name__ == "__main__":
    run_comparison()
