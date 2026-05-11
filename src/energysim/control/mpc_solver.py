import jax
import jax.numpy as jnp
from jax import jit, lax
from typing import Optional
import jaxopt

# Import the simulator and objective functions
from ..sim.simulator import JAXSimulator
from ..utils.objectives import f_terminal_cost

from ..core.shared.data_structs import (
    SystemActions, ExogenousData
)


from ..utils.objectives import (
    extract_battery_only_forecast,
    build_battery_qp_static_data,
    update_battery_qp_dynamic_vectors,
)


class JAX_MPC_Solver:
    """
    Hard-constrained battery-only MPC using JAXopt BoxOSQP.

    This solver:
        - optimizes charge/discharge power
        - enforces cumulative battery energy bounds as hard constraints
        - enforces battery power bounds as hard constraints
        - enforces grid balance as a hard equality
        - returns the first physical battery action

    Sign convention returned to simulator:
        battery_power_w > 0: charging
        battery_power_w < 0: discharging

    Important:
        This is no longer a generic projected-gradient MPC.
        It is a dedicated battery-only QP MPC.
    """

    def __init__(
        self,
        N_horizon: int,
        simulator_template: JAXSimulator,
        maxiter: int = 6000,
        tol: float = 1e-4,
        stage_cost_fn=None,
        terminal_cost_fn=None,
        trajectory_cost_fn=None,
        phys_min=None,
        phys_max=None,
    ):
        del stage_cost_fn, terminal_cost_fn, trajectory_cost_fn, phys_min, phys_max

        self.N = N_horizon
        self.n_rooms = len(simulator_template.thermal.config.room_air_indices)

        self.simulator_template = simulator_template

        self.b_conf = simulator_template.battery.config
        self.r_conf = simulator_template.configs[2]
        self.pv_conf = simulator_template.pv.config
        self.dt_seconds = simulator_template.dt_seconds
        self.qp_static = build_battery_qp_static_data(
            N_horizon=self.N,
            b_conf=self.b_conf,
            r_conf=self.r_conf,
            dt_seconds=self.dt_seconds,
        )

        self.optimizer = jaxopt.BoxOSQP(
            maxiter=maxiter,
            tol=tol,
            eq_qp_solve="lu",
            jit=True,
            verbose=0,
        )

        self.warm_start = None

    def solve(
        self,
        current_sim: JAXSimulator,
        exo_forecast: ExogenousData,
        warm_start: Optional[object] = None,
        warm_start_norm_actions: Optional[SystemActions] = None,
    ) -> SystemActions:
        del warm_start_norm_actions

        load_w, pv_w, price = extract_battery_only_forecast(
            exo_forecast=exo_forecast,
            N_horizon=self.N,
            pv_conf=self.pv_conf,
        )

        current_soc = current_sim.state.battery.soc

        c, l, u = update_battery_qp_dynamic_vectors(
            self.qp_static,
            current_soc=current_soc,
            load_w=load_w,
            pv_w=pv_w,
            price=price,
        )

        init_params = warm_start if warm_start is not None else self.warm_start

        result = self.optimizer.run(
            init_params=init_params,
            params_obj=(self.qp_static.Q, c),
            params_eq=self.qp_static.A,
            params_ineq=(l, u),
        )

        self.warm_start = result.params

        primal = result.params.primal
        x = primal[0] if isinstance(primal, tuple) else primal

        p_charge_0 = x[0]
        p_discharge_0 = x[self.N]

        battery_power_w = p_charge_0 - p_discharge_0

        zero_zonal = jnp.zeros((self.n_rooms,))

        first_action = SystemActions(
            battery_power_w=battery_power_w,
            heat_pump_power_w=zero_zonal,
            ac_power_w=zero_zonal,
            storage_discharge_w=zero_zonal,
        )

        return first_action
