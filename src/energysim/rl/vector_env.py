import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Optional, Tuple

from ..sim.simulator import JAXSimulator
from ..core.shared.data_structs import SystemState, SystemActions, ExogenousData, SystemOutputs
from ..core.data.dataset import SimulationDataset
from ..behavior.base import AbstractBehavioralModel
from ..sim.helpers import precalculate_exogenous_data
from ..utils.objectives import f_stage_cost, f_terminal_cost  # <--- decoupled cost functions


def feasible_battery_power_w(soc, requested_w, b_config, dt_seconds):
    """The battery power the grid actually sees, after hardware and SOC limits.

    Mirrors SimpleBatteryModel.step, which clips to +/- max_power_w and then
    saturates SOC into [0, 1] -- power beyond the remaining headroom is never
    delivered. f_stage_cost was written for the MPC path, where the QP already
    guarantees feasibility, so it bills whatever action it is handed. Handing it
    a raw RL action lets a policy be paid for energy the battery never moved.
    """
    hw_w = jnp.clip(requested_w, -b_config.max_power_w, b_config.max_power_w)
    one_way_eff = jnp.sqrt(b_config.efficiency)

    # Charging is derated by efficiency; discharging must overdraw to deliver.
    max_charge_w = (1.0 - soc) * b_config.capacity_j / (dt_seconds * one_way_eff)
    max_discharge_w = -soc * b_config.capacity_j * one_way_eff / dt_seconds

    return jnp.clip(hw_w, max_discharge_w, max_charge_w)


# --- 1. Define the Environment State Wrapper ---
class EnvState(eqx.Module):
    sim: JAXSimulator          # Holds the batched physical state (battery, temps, etc.)
    time_idx: jax.Array        # Current integer time step (batched for consistency)

class VectorizedEnergyEnv(eqx.Module):
    """
    Memory-Optimized Vectorized Environment for RL.
    """
    sim_template: JAXSimulator
    shared_exo_data: ExogenousData
    num_envs: int = eqx.field(static=True)
    episode_length: Optional[int] = eqx.field(static=True)
    terminal_soc_cost: bool = eqx.field(static=True)

    def __init__(
        self,
        simulator: JAXSimulator,
        dataset: SimulationDataset,
        num_envs: int,
        behavioral_models: Optional[Dict[str, AbstractBehavioralModel]] = None,
        episode_length: Optional[int] = None,
        terminal_soc_cost: bool = False
    ):
        self.num_envs = num_envs
        self.sim_template = simulator
        # None -> episodes end when the dataset wraps. Set it to e.g. 96 to cut
        # the run into fixed-length (daily) episodes for training.
        self.episode_length = episode_length
        # Charge an f_terminal_cost penalty for ending an episode away from
        # TERMINAL_SOC_TARGET. Without it, an agent that starts each episode
        # with charge it did not pay for can simply sell that charge and book a
        # profit, which beats learning to arbitrage.
        self.terminal_soc_cost = terminal_soc_cost

        # 1. Pre-calculate Exogenous Data
        n_rooms = len(simulator.thermal.config.room_air_indices)
        dummy_sim = simulator.reset() 
        
        print("Pre-calculating data on CPU...")
        self.shared_exo_data = precalculate_exogenous_data(
            dataset=dataset,
            behavioral_models=behavioral_models or {},
            dt_seconds=simulator.dt_seconds,
            n_rooms=n_rooms,
            dummy_state=dummy_sim.state
        )

    def reset(self, key: jax.Array, randomize: bool = False) -> EnvState:
        """
        Returns the initial EnvState replicated across num_envs.

        With randomize=True every env gets its own starting SOC and its own
        offset into the dataset. That is what decorrelates the batch: otherwise
        all envs sit at the same SOC and the same timestep forever, so whenever
        the battery is at a bound the whole batch is at that bound at once.
        Evaluation should leave it False so runs stay reproducible.
        """
        # 1. Reset single simulator
        single_sim = self.sim_template.reset()

        # 2. Replicate Simulator State and Actions
        def replicate(leaf):
            return jnp.repeat(leaf[None, ...], self.num_envs, axis=0)

        batch_sims = jax.tree.map(replicate, single_sim)

        # 3. Initialize Time
        batch_time = jnp.zeros(self.num_envs, dtype=jnp.int32)

        if randomize:
            soc_key, time_key = jax.random.split(key)
            batch_sims = eqx.tree_at(
                lambda s: s.battery.soc,
                batch_sims,
                jax.random.uniform(soc_key, (self.num_envs,))
            )
            batch_time = jax.random.randint(
                time_key, (self.num_envs,), 0, self.n_steps
            ).astype(jnp.int32)

        return EnvState(
            sim=batch_sims,
            time_idx=batch_time
        )

    @property
    def n_steps(self) -> int:
        """Number of timesteps in the dataset."""
        return self.shared_exo_data.price.shape[0]

    def reset_soc_where(self, state: EnvState, mask: jax.Array, key: jax.Array):
        """Resample SOC for the envs flagged by `mask` (i.e. episode boundaries).

        Returns (new_state, energy_cost). Time keeps running -- only the battery
        is handed a fresh starting charge, so each episode is a new day starting
        from a different level. That diversity is what keeps the batch out of
        the SOC=0 trap, where every sampled action discharges an empty battery,
        every outcome is identical and there is no gradient in any direction.

        energy_cost prices the charge handed over, at what it would have cost to
        import. Without it the reset is a gift: an agent that empties the
        battery before midnight gets refilled for free, so dumping is optimal in
        training and value-destroying in deployment. Charging for the difference
        makes it symmetric -- arriving at the boundary empty costs exactly the
        value of the energy that was dumped, and arriving full costs nothing.
        """
        fresh_soc = jax.random.uniform(key, (self.num_envs,))
        old_soc = state.sim.battery.soc
        new_soc = jnp.where(mask, fresh_soc, old_soc)
        new_state = eqx.tree_at(lambda s: s.sim.battery.soc, state, new_soc)

        t = state.time_idx % self.n_steps
        wholesale = self.shared_exo_data.price[t]
        _, _, r_conf, *_ = self.sim_template.configs
        import_price = (
            wholesale * (1.0 + r_conf.import_tax_rate)
            + r_conf.import_grid_fee_eur_per_kwh
        )
        # Priced against the MEAN of the reset distribution, not the draw that
        # actually came out. Both have the same expectation, so the incentive is
        # identical, but the realised draw is not something the policy can see
        # or influence -- billing it injects pure noise into the reward. That
        # noise is large next to a ~0.007 EUR timestep, and it cost the critic
        # more than half its accuracy (explained variance 0.94 -> 0.60), which
        # in turn made every advantage estimate unreliable.
        mean_reset_soc = 0.5
        delta_kwh = (mean_reset_soc - old_soc) * self.sim_template.battery.config.capacity_kwh
        energy_cost = jnp.where(mask, delta_kwh * import_price * r_conf.price_weight, 0.0)

        return new_state, energy_cost

    @jax.jit
    def step(
        self, 
        state: EnvState, 
        actions: SystemActions
    ) -> Tuple[EnvState, jax.Array, jax.Array, dict]:
        """
        Step function complying with standard JAX RL signatures:
        step(state, action) -> (next_state, reward, done, info)
        """
        
        # 1. Extract Data for the current time step.
        # Indexed per-env, so envs may sit at different points in the dataset.
        # The modulo matters: time_idx used to run past the end of the dataset,
        # where JAX silently clamps out-of-bounds gathers, freezing every env on
        # the final row of exogenous data for the rest of training.
        t = state.time_idx % self.n_steps
        exo_batch = jax.tree.map(lambda x: x[t], self.shared_exo_data)

        # 2. VMAP the Simulator Step
        # Simulator returns: (next_sim, SystemOutputs)
        next_sims, batched_outputs = jax.vmap(JAXSimulator.step, in_axes=(0, 0, 0))(
            state.sim,
            actions,
            exo_batch
        )

        # 3. VMAP the Cost Calculation
        # Bill the power the battery could actually deliver from its current SOC,
        # not the power that was requested -- see feasible_battery_power_w.
        billed_actions = SystemActions(
            battery_power_w=feasible_battery_power_w(
                state.sim.battery.soc,
                actions.battery_power_w,
                self.sim_template.battery.config,
                self.sim_template.dt_seconds
            ),
            heat_pump_power_w=actions.heat_pump_power_w,
            ac_power_w=actions.ac_power_w,
            storage_discharge_w=actions.storage_discharge_w
        )

        # We write a small wrapper to neatly pass the batched elements to f_cost_step
        def calc_reward(sim_k, act_k, out_k, exo_k):
            # current_state/next_state are accepted but unused inside f_stage_cost itself.
            cost = f_stage_cost(
                current_state=sim_k.state,
                next_state=sim_k.state,
                actions=act_k,
                outputs=out_k,
                exogenous=exo_k,
                configs=sim_k.configs,
                dt_seconds=sim_k.dt_seconds
            )
            return -cost  # Reward is negative cost

        # Map over the simulators, actions, outputs and exogenous slices.
        rewards = jax.vmap(calc_reward, in_axes=(0, 0, 0, 0))(
            state.sim, billed_actions, batched_outputs, exo_batch
        )

        # 4. Update State
        next_time_idx = state.time_idx + 1
        next_state = EnvState(
            sim=next_sims,
            time_idx=next_time_idx
        )

        # 5. Check Termination
        # Returns a boolean array of shape (num_envs,)
        boundary = self.episode_length if self.episode_length is not None else self.n_steps
        done = (next_time_idx % boundary) == 0

        # 6. Terminal SOC penalty, charged on the step that ends an episode.
        if self.terminal_soc_cost:
            def calc_terminal(sim_k):
                # initial_state / exo_forecast_end are discarded inside.
                return f_terminal_cost(sim_k.state, sim_k.state, sim_k.configs, None)

            terminal = jax.vmap(calc_terminal)(next_sims)
            rewards = rewards - jnp.where(done, terminal, 0.0)
        
        # 6. Populate Info Dictionary
        # This gives your RL logger access to exactly what happened physically
        info = {
            "outputs": batched_outputs,
            "bat_soc": next_sims.battery.soc,
            "room_temps": next_sims.thermal.T_vector[:, jnp.array(self.sim_template.thermal.config.room_air_indices)]
        }
        
        return next_state, rewards, done, info