import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Optional

from ..sim.simulator import JAXSimulator
from ..core.shared.data_structs import SystemState, SystemActions, ExogenousData
from ..core.data.dataset import SimulationDataset
from ..behavior.base import AbstractBehavioralModel
from ..sim.helpers import precalculate_exogenous_data

# --- 1. Define the Environment State Wrapper ---
class EnvState(eqx.Module):
    sim: JAXSimulator          # Holds the physical state (battery, temps, etc.)
    prev_actions: SystemActions # Holds the history for slew rate cost
    time_idx: int              # Current integer time step

class VectorizedEnergyEnv(eqx.Module):
    """
    Memory-Optimized Vectorized Environment.
    """
    sim_template: JAXSimulator
    shared_exo_data: ExogenousData
    num_envs: int = eqx.field(static=True)
    
    # Helper to construct zero-actions for initialization
    _dummy_action_struct: SystemActions

    def __init__(
        self,
        simulator: JAXSimulator,
        dataset: SimulationDataset,
        num_envs: int,
        behavioral_models: Optional[Dict[str, AbstractBehavioralModel]] = None
    ):
        self.num_envs = num_envs
        self.sim_template = simulator

        # 1. Pre-calculate Exogenous Data
        n_rooms = len(simulator.thermal.config.room_air_indices)
        dummy_state = simulator.reset() 
        
        print("Pre-calculating data on CPU...")
        self.shared_exo_data = precalculate_exogenous_data(
            dataset=dataset,
            behavioral_models=behavioral_models or {},
            dt_seconds=simulator.dt_seconds,
            n_rooms=n_rooms,
            dummy_state=dummy_state.state
        )
        
        # 2. Create a dummy action structure for initialization (all zeros)
        # We only need n_rooms to define the shape; specific config limits aren't needed for zeros.
        self._dummy_action_struct = SystemActions(
            battery_power_w=jnp.array(0.0),
            heat_pump_power_w=jnp.zeros(n_rooms),
            ac_power_w=jnp.zeros(n_rooms),
            storage_discharge_w=jnp.zeros(n_rooms)
        )

    def reset(self, key: jax.Array) -> EnvState:
        """
        Returns the initial EnvState replicated across num_envs.
        """
        # 1. Reset single simulator
        single_sim = self.sim_template.reset()
        
        # 2. Replicate Simulator State
        def replicate(leaf):
            return jnp.repeat(leaf[None, ...], self.num_envs, axis=0)
            
        batch_sims = jax.tree.map(replicate, single_sim)
        
        # 3. Replicate Initial (Zero) Actions
        batch_prev_actions = jax.tree.map(replicate, self._dummy_action_struct)
        
        # 4. Initialize Time
        batch_time = jnp.zeros(self.num_envs, dtype=jnp.int32)
        
        return EnvState(
            sim=batch_sims,
            prev_actions=batch_prev_actions,
            time_idx=batch_time
        )

    @jax.jit
    def step(
        self, 
        state: EnvState, 
        actions: SystemActions
    ) -> tuple[EnvState, jax.Array, dict]:
        """
        Step function complying with standard JAX RL signatures:
        step(state, action) -> (next_state, reward, info)
        """
        
        # 1. Extract Data for the current time step
        # We assume all envs are synchronized in time for memory efficiency.
        # Taking the time from the first env is safe here.
        t = state.time_idx[0] 
        exo_batch = jax.tree.map(lambda x: x[t], self.shared_exo_data)
        
        # Broadcast exo data to batch size (optional depending on vmap logic, 
        # but cleaner to treat exo as a singleton broadcasted inside vmap)
        
        # 2. VMAP the Simulator Step
        # JAXSimulator.step signature: (self, actions, prev_actions, exo)
        
        # in_axes:
        # state.sim: 0 (Batched)
        # actions: 0 (Batched)
        # state.prev_actions: 0 (Batched)
        # exo_batch: None (Broadcasted - same weather for all agents, 
        # or 0 if you wanted different weather per agent)
        
        next_sims, costs = jax.vmap(JAXSimulator.step, in_axes=(0, 0, 0, None))(
            state.sim, 
            actions, 
            state.prev_actions, 
            exo_batch
        )
        
        # 3. Update State
        next_state = EnvState(
            sim=next_sims,
            prev_actions=actions, # Current action becomes next prev_action
            time_idx=state.time_idx + 1
        )
        
        # 4. Calculate Reward
        reward = -costs
        
        # 5. Check Termination (if end of dataset)
        # (Simple fixed horizon check)
        done = state.time_idx >= (len(self.shared_exo_data.price) - 1)
        
        return next_state, reward, done