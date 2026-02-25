import jax
import jax.numpy as jnp
from jax import jit, lax
from typing import Optional
import jaxopt
import equinox as eqx

# Import the simulator and objective functions
from ..sim.simulator import JAXSimulator
from ..utils.objectives import f_cost_step, f_terminal_cost

from ..core.shared.data_structs import (
    SystemActions, ExogenousData, SystemState
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
                    state=sim_k.state,
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
            maxiter=100,         # Give it slightly more iterations to refine
            stepsize=0.1,        # 0.1 stepsize in normalized space is HUGE (10% of max power per step)
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
        
        first_action = jax.tree.map(lambda x: x[0], physical_optimal_sequence)
        return first_action