import jax
import jax.numpy as jnp
from jax import jit, lax
from typing import Optional
import jaxopt

# Import the simulator and objective functions
from ..sim.simulator import JAXSimulator
from ..utils.objectives import f_cost_step, f_terminal_cost

from ..core.shared.data_structs import (
    SystemActions, ExogenousData
)

class JAX_MPC_Solver:
    """
    A high-level MPC solver that natively utilizes the JAXSimulator 
    for perfect environment modeling and rollout.
    """
    def __init__(
        self,
        N_horizon: int,
        simulator_template: JAXSimulator  # <--- Take the simulator as a template
    ):
        self.N = N_horizon
        self.n_rooms = len(simulator_template.thermal.config.room_air_indices)

        # --- 1. Define the Total Horizon Cost Function ---
        # Note: jaxopt differentiates with respect to the *first* argument (action_sequence)
        @jit
        def f_horizon_cost(
            action_sequence: SystemActions, 
            initial_sim: JAXSimulator,      # The PyTree containing state and config
            exo_sequence: ExogenousData     
        ):
            
            # The scan step now just calls the simulator's native step
            def mpc_scan_step(carry, inputs_k):
                sim_k,  = carry
                action_k, exo_k = inputs_k

                # 1. Step the physics engine
                next_sim, outputs_k = sim_k.step(action_k, exo_k)

                # 2. Evaluate the economic/comfort cost
                cost_k = f_cost_step(
                    state=next_sim.state,
                    actions=action_k,
                    outputs=outputs_k,
                    exogenous=exo_k,
                    configs=sim_k.configs,
                    dt_seconds=sim_k.dt_seconds
                )

                # Return the updated simulator as the next carry
                return (next_sim, ), cost_k

            init_carry = (initial_sim, )

            # Run the Scan
            (final_sim, ), cost_sequence = lax.scan(
                mpc_scan_step, init_carry, (action_sequence, exo_sequence)
            )
            
            # Calculate Terminal Cost using the final state from the rolled-out simulator
            last_exo = jax.tree.map(lambda x: x[-1], exo_sequence)
            term_cost = f_terminal_cost(
                final_state=final_sim.state, 
                initial_state=initial_sim.state, 
                configs=initial_sim.configs, 
                exo_forecast_end=last_exo
            )
            
            return jnp.sum(cost_sequence) + term_cost

        self.objective_fn = f_horizon_cost

        # --- 2. Setup the Optimizer Bounds ---
        b_conf = simulator_template.battery.config
        hp_conf = simulator_template.heat_pump.config
        ac_conf = simulator_template.ac.config
        ts_conf = simulator_template.storage.config

        scalar_shape = (N_horizon,)
        zonal_shape = (N_horizon, self.n_rooms)

        # Store true physical bounds for un-normalization
        self.phys_min = SystemActions(
            battery_power_w=jnp.full(scalar_shape, -b_conf.max_power_w),
            heat_pump_power_w=jnp.full(zonal_shape, 0.0),
            ac_power_w=jnp.full(zonal_shape, 0.0),
            storage_discharge_w=jnp.full(zonal_shape, 0.0)
        )
        self.phys_max = SystemActions(
            battery_power_w=jnp.full(scalar_shape, b_conf.max_power_w),
            heat_pump_power_w=jnp.full(zonal_shape, hp_conf.max_electrical_power_w / self.n_rooms),
            ac_power_w=jnp.full(zonal_shape, ac_conf.max_electrical_power_w / self.n_rooms),
            storage_discharge_w=jnp.full(zonal_shape, ts_conf.max_discharge_w / self.n_rooms)
        )

        def unnormalize(norm_actions: SystemActions) -> SystemActions:
            """Maps [0, 1] back to physical Watts."""
            return jax.tree.map(
                lambda n, p_min, p_max: p_min + n * (p_max - p_min),
                norm_actions, self.phys_min, self.phys_max
            )

        # Wrap the objective function so the optimizer only sees normalized [0, 1] inputs
        @jit
        def normalized_objective(norm_action_seq, initial_sim, exo_seq):
            physical_actions = unnormalize(norm_action_seq)
            return self.objective_fn(physical_actions, initial_sim, exo_seq)

        # The optimizer now operates strictly inside a [0, 1] hypercube
        self.norm_bounds = (
            jax.tree.map(lambda x: jnp.zeros_like(x), self.phys_min),
            jax.tree.map(lambda x: jnp.ones_like(x), self.phys_max)
        )

        self.optimizer = jaxopt.ProjectedGradient(
            fun=normalized_objective,
            projection=jaxopt.projection.projection_box,
            maxiter=200,         # More iterations for tighter convergence
            stepsize=0.02,       # Smaller steps reduce corner-chasing/saturation artifacts
            tol=1e-4,
        )

        # Start warm start right in the middle (0.5 = 0 Watts for battery)
        self.norm_warm_start = jax.tree.map(
            lambda x: jnp.full_like(x, 0.5), self.phys_min
        )

    def solve(
        self,
        current_sim: JAXSimulator,   
        exo_forecast: ExogenousData,
        warm_start_norm_actions: Optional[SystemActions] = None
    ) -> SystemActions:

        if warm_start_norm_actions is None:
            warm_start_norm_actions = self.norm_warm_start

        optim_result = self.optimizer.run(
            init_params=warm_start_norm_actions,
            hyperparams_proj=self.norm_bounds,
            initial_sim=current_sim,
            exo_seq=exo_forecast
        )

        # Map the optimal sequence back to physical Watts before extracting the first action
        physical_optimal_sequence = jax.tree.map(
            lambda n, p_min, p_max: p_min + n * (p_max - p_min),
            optim_result.params, self.phys_min, self.phys_max
        )

        # Receding-horizon warm start: shift one step forward for the next solve call.
        self.norm_warm_start = jax.tree.map(
            lambda x: jnp.concatenate([x[1:], x[-1:]], axis=0),
            optim_result.params
        )
        
        first_action = jax.tree.map(lambda x: x[0], physical_optimal_sequence)

        # Safety projection on first control move to avoid contradictory HVAC commands.
        room_idx = jnp.array(current_sim.thermal.config.room_air_indices)
        room_temps = current_sim.state.thermal.T_vector[room_idx]
        setpoint = current_sim.thermal.config.setpoint
        band = current_sim.thermal.config.comfort_band
        deviation = room_temps - setpoint

        # If already too cold, do not cool. If too hot, do not heat via storage.
        ac_power = jnp.where(room_temps <= (setpoint - band), 0.0, first_action.ac_power_w)
        storage_power = jnp.where(room_temps >= (setpoint + band), 0.0, first_action.storage_discharge_w)

        # If both are active in one zone, keep only the thermodynamically sensible one.
        both_active = (ac_power > 0.0) & (storage_power > 0.0)
        keep_cooling = both_active & (deviation >= 0.0)
        keep_heating = both_active & (deviation < 0.0)
        storage_power = jnp.where(keep_cooling, 0.0, storage_power)
        ac_power = jnp.where(keep_heating, 0.0, ac_power)

        # Battery edge clamps for cleaner actionable power near SOC limits.
        batt_power = first_action.battery_power_w
        soc = current_sim.state.battery.soc
        batt_power = jnp.where((soc >= 0.98) & (batt_power > 0.0), 0.0, batt_power)
        batt_power = jnp.where((soc <= 0.02) & (batt_power < 0.0), 0.0, batt_power)

        return SystemActions(
            battery_power_w=batt_power,
            heat_pump_power_w=first_action.heat_pump_power_w,
            ac_power_w=ac_power,
            storage_discharge_w=storage_power,
        )


class StandardMPCSolver:
    """Baseline MPC using projected gradient with first-action safety repair."""

    def __init__(
        self,
        N_horizon: int,
        simulator_template: JAXSimulator,
    ):
        self.N = N_horizon
        self.n_rooms = len(simulator_template.thermal.config.room_air_indices)
        self.dt_seconds = simulator_template.dt_seconds
        self.room_idx = jnp.array(simulator_template.thermal.config.room_air_indices)
        self.setpoint = simulator_template.thermal.config.setpoint
        self.band = simulator_template.thermal.config.comfort_band
        self.temp_low = self.setpoint - self.band
        self.temp_high = self.setpoint + self.band
        self.safety_iters = 6

        @jit
        def horizon_cost(
            action_sequence: SystemActions,
            initial_sim: JAXSimulator,
            exo_sequence: ExogenousData,
        ):
            def mpc_scan_step(carry, inputs_k):
                sim_k, = carry
                action_k, exo_k = inputs_k
                next_sim, outputs_k = sim_k.step(action_k, exo_k)
                cost_k = f_cost_step(
                    state=next_sim.state,
                    actions=action_k,
                    outputs=outputs_k,
                    exogenous=exo_k,
                    configs=sim_k.configs,
                    dt_seconds=sim_k.dt_seconds,
                )
                return (next_sim,), cost_k

            (final_sim,), stage_costs = lax.scan(
                mpc_scan_step, (initial_sim,), (action_sequence, exo_sequence)
            )
            last_exo = jax.tree.map(lambda x: x[-1], exo_sequence)
            terminal = f_terminal_cost(
                final_state=final_sim.state,
                initial_state=initial_sim.state,
                configs=initial_sim.configs,
                exo_forecast_end=last_exo,
            )
            return jnp.sum(stage_costs) + terminal

        self.objective_fn = horizon_cost

        b_conf = simulator_template.battery.config
        hp_conf = simulator_template.heat_pump.config
        ac_conf = simulator_template.ac.config
        ts_conf = simulator_template.storage.config
        self.b_conf = b_conf
        self.one_way_eff = jnp.sqrt(self.b_conf.efficiency)
        self.hp_cop_heating = hp_conf.cop_heating
        self.hp_max_room_w = hp_conf.max_electrical_power_w / self.n_rooms
        self.hp_max_total_w = hp_conf.max_electrical_power_w
        self.ac_max_room_w = ac_conf.max_electrical_power_w / self.n_rooms
        self.storage_max_room_w = ts_conf.max_discharge_w / self.n_rooms
        self.storage_max_total_w = ts_conf.max_discharge_w
        self.tank_reserve_c = 26.0
        self.tank_emergency_reserve_c = 23.0
        self.tank_recharge_on_c = 30.0
        self.hp_recharge_floor_frac = 0.05
        self.hp_recharge_match_frac = 0.50
        node_vol_m3 = ts_conf.volume_m3 / ts_conf.n_nodes
        self.node_heat_capacity_j_k = node_vol_m3 * 4186.0 * 1000.0

        scalar_shape = (N_horizon,)
        zonal_shape = (N_horizon, self.n_rooms)

        self.phys_min = SystemActions(
            battery_power_w=jnp.full(scalar_shape, -b_conf.max_power_w),
            heat_pump_power_w=jnp.zeros(zonal_shape),
            ac_power_w=jnp.zeros(zonal_shape),
            storage_discharge_w=jnp.zeros(zonal_shape),
        )
        self.phys_max = SystemActions(
            battery_power_w=jnp.full(scalar_shape, b_conf.max_power_w),
            heat_pump_power_w=jnp.full(
                zonal_shape, hp_conf.max_electrical_power_w / self.n_rooms
            ),
            ac_power_w=jnp.full(
                zonal_shape, ac_conf.max_electrical_power_w / self.n_rooms
            ),
            storage_discharge_w=jnp.full(
                zonal_shape, ts_conf.max_discharge_w / self.n_rooms
            ),
        )

        def unnormalize(norm_actions: SystemActions) -> SystemActions:
            return jax.tree.map(
                lambda n, p_min, p_max: p_min + n * (p_max - p_min),
                norm_actions,
                self.phys_min,
                self.phys_max,
            )

        @jit
        def normalized_objective(norm_action_seq, initial_sim, exo_seq):
            physical_actions = unnormalize(norm_action_seq)
            return self.objective_fn(physical_actions, initial_sim, exo_seq)

        self.norm_bounds = (
            jax.tree.map(lambda x: jnp.zeros_like(x), self.phys_min),
            jax.tree.map(lambda x: jnp.ones_like(x), self.phys_max),
        )

        self.optimizer = jaxopt.ProjectedGradient(
            fun=normalized_objective,
            projection=jaxopt.projection.projection_box,
            maxiter=200,
            stepsize=0.001,
            tol=1e-4,
        )

        self.norm_warm_start = SystemActions(
            battery_power_w=jnp.full(scalar_shape, 0.5),
            heat_pump_power_w=jnp.zeros(zonal_shape),
            ac_power_w=jnp.zeros(zonal_shape),
            storage_discharge_w=jnp.zeros(zonal_shape),
        )

    def _battery_power_bounds_from_soc(self, soc):
        soc_clip = jnp.clip(soc, 0.0, 1.0)
        max_discharge_soc_w = (
            soc_clip * self.b_conf.capacity_j * self.one_way_eff
        ) / self.dt_seconds
        max_charge_soc_w = (
            (1.0 - soc_clip) * self.b_conf.capacity_j / self.one_way_eff
        ) / self.dt_seconds
        lower = -jnp.minimum(self.b_conf.max_power_w, max_discharge_soc_w)
        upper = jnp.minimum(self.b_conf.max_power_w, max_charge_soc_w)
        return lower, upper

    def _cap_storage_by_tank_reserve(
        self,
        storage_per_room_w,
        tank_temps_c,
        reserve_temp_c,
    ):
        extractable_energy_j = jnp.sum(
            jnp.clip(tank_temps_c - reserve_temp_c, 0.0, jnp.inf)
            * self.node_heat_capacity_j_k
        )
        max_discharge_from_reserve_w = extractable_energy_j / self.dt_seconds
        max_total_w = jnp.minimum(self.storage_max_total_w, max_discharge_from_reserve_w)
        req_total_w = jnp.sum(storage_per_room_w)
        scale = jnp.where(
            req_total_w > max_total_w,
            max_total_w / jnp.maximum(req_total_w, 1e-6),
            1.0,
        )
        return storage_per_room_w * scale

    def _project_first_action_to_comfort_band(
        self,
        current_sim: JAXSimulator,
        exo_first: ExogenousData,
        action: SystemActions,
    ) -> SystemActions:
        batt_low, batt_high = self._battery_power_bounds_from_soc(
            current_sim.state.battery.soc
        )
        batt = jnp.clip(action.battery_power_w, batt_low, batt_high)
        hp = jnp.clip(action.heat_pump_power_w, 0.0, self.hp_max_room_w)
        ac = jnp.clip(action.ac_power_w, 0.0, self.ac_max_room_w)
        stor = jnp.clip(action.storage_discharge_w, 0.0, self.storage_max_room_w)
        current_room_temps = current_sim.state.thermal.T_vector[self.room_idx]
        tank_temps = current_sim.state.storage.temperatures_c
        tank_mean_c = jnp.mean(tank_temps)
        current_too_cold = current_room_temps < self.temp_low
        reserve_now = jnp.where(
            jnp.any(current_too_cold),
            self.tank_emergency_reserve_c,
            self.tank_reserve_c,
        )
        stor = self._cap_storage_by_tank_reserve(stor, tank_temps, reserve_now)

        projected = SystemActions(
            battery_power_w=batt,
            heat_pump_power_w=hp,
            ac_power_w=ac,
            storage_discharge_w=stor,
        )

        def safety_iter(_, a_prev):
            sim_next, _ = current_sim.step(a_prev, exo_first)
            next_room_temps = sim_next.state.thermal.T_vector[self.room_idx]

            too_hot = next_room_temps > self.temp_high
            too_cold = next_room_temps < self.temp_low

            ac_new = jnp.where(too_hot, self.ac_max_room_w, a_prev.ac_power_w)
            stor_new = jnp.where(too_hot, 0.0, a_prev.storage_discharge_w)

            stor_new = jnp.where(too_cold, self.storage_max_room_w, stor_new)
            ac_new = jnp.where(too_cold, 0.0, ac_new)

            both_active = (ac_new > 0.0) & (stor_new > 0.0)
            cool_preferred = next_room_temps >= self.setpoint
            stor_new = jnp.where(both_active & cool_preferred, 0.0, stor_new)
            ac_new = jnp.where(
                both_active & jnp.logical_not(cool_preferred),
                0.0,
                ac_new,
            )

            comfort_emergency = jnp.any(current_too_cold) | jnp.any(too_cold)
            reserve_temp = jnp.where(
                comfort_emergency,
                self.tank_emergency_reserve_c,
                self.tank_reserve_c,
            )
            stor_new = self._cap_storage_by_tank_reserve(
                stor_new,
                tank_temps,
                reserve_temp,
            )

            stor_total = jnp.sum(stor_new)
            hp_floor_from_discharge = (
                self.hp_recharge_match_frac
                * stor_total
                / jnp.maximum(self.hp_cop_heating, 1.0)
            )
            hp_floor_low_tank = jnp.where(
                (tank_mean_c < self.tank_recharge_on_c) | jnp.any(too_cold),
                self.hp_recharge_floor_frac * self.hp_max_total_w,
                0.0,
            )
            hp_required_total = jnp.minimum(
                self.hp_max_total_w,
                jnp.maximum(hp_floor_from_discharge, hp_floor_low_tank),
            )
            hp_floor_per_room = jnp.full((self.n_rooms,), hp_required_total / self.n_rooms)
            hp_new = jnp.clip(
                jnp.maximum(a_prev.heat_pump_power_w, hp_floor_per_room),
                0.0,
                self.hp_max_room_w,
            )

            return SystemActions(
                battery_power_w=a_prev.battery_power_w,
                heat_pump_power_w=hp_new,
                ac_power_w=ac_new,
                storage_discharge_w=stor_new,
            )

        repaired = lax.fori_loop(0, self.safety_iters, safety_iter, projected)

        ac_final = jnp.where(current_room_temps < self.setpoint, 0.0, repaired.ac_power_w)
        stor_final = jnp.where(
            current_room_temps > self.setpoint,
            0.0,
            repaired.storage_discharge_w,
        )
        both_final = (ac_final > 0.0) & (stor_final > 0.0)
        stor_final = jnp.where(both_final, 0.0, stor_final)
        stor_final = self._cap_storage_by_tank_reserve(
            stor_final,
            tank_temps,
            reserve_now,
        )

        return SystemActions(
            battery_power_w=repaired.battery_power_w,
            heat_pump_power_w=repaired.heat_pump_power_w,
            ac_power_w=ac_final,
            storage_discharge_w=stor_final,
        )

    def solve(
        self,
        current_sim: JAXSimulator,
        exo_forecast: ExogenousData,
        warm_start_norm_actions: Optional[SystemActions] = None,
    ) -> SystemActions:
        if warm_start_norm_actions is None:
            warm_start_norm_actions = self.norm_warm_start

        optim_result = self.optimizer.run(
            init_params=warm_start_norm_actions,
            hyperparams_proj=self.norm_bounds,
            initial_sim=current_sim,
            exo_seq=exo_forecast,
        )

        physical_optimal_sequence = jax.tree.map(
            lambda n, p_min, p_max: p_min + n * (p_max - p_min),
            optim_result.params,
            self.phys_min,
            self.phys_max,
        )

        self.norm_warm_start = jax.tree.map(
            lambda x: jnp.concatenate([x[1:], x[-1:]], axis=0),
            optim_result.params,
        )
        first_action = jax.tree.map(lambda x: x[0], physical_optimal_sequence)
        exo_first = jax.tree.map(lambda x: x[0], exo_forecast)
        return self._project_first_action_to_comfort_band(
            current_sim,
            exo_first,
            first_action,
        )


class MPC_solver_NewMPC(StandardMPCSolver):
    """Compatibility wrapper for the extracted baseline MPC solver."""

    pass
