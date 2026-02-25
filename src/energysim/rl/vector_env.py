import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Optional, Tuple

from ..sim.simulator import JAXSimulator
from ..core.shared.data_structs import SystemState, SystemActions, ExogenousData, SystemOutputs
from ..core.data.dataset import SimulationDataset
from ..behavior.base import AbstractBehavioralModel
from ..sim.helpers import precalculate_exogenous_data
from ..utils.objectives import f_cost_step  # <--- Import the decoupled cost function

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
        dummy_sim = simulator.reset() 
        
        print("Pre-calculating data on CPU...")
        self.shared_exo_data = precalculate_exogenous_data(
            dataset=dataset,
            behavioral_models=behavioral_models or {},
            dt_seconds=simulator.dt_seconds,
            n_rooms=n_rooms,
            dummy_state=dummy_sim.state
        )

    def reset(self, key: jax.Array) -> EnvState:
        """
        Returns the initial EnvState replicated across num_envs.
        """
        # 1. Reset single simulator
        single_sim = self.sim_template.reset()
        
        # 2. Replicate Simulator State and Actions
        def replicate(leaf):
            return jnp.repeat(leaf[None, ...], self.num_envs, axis=0)
            
        batch_sims = jax.tree.map(replicate, single_sim)
        
        # 3. Initialize Time
        batch_time = jnp.zeros(self.num_envs, dtype=jnp.int32)
        
        return EnvState(
            sim=batch_sims,
            time_idx=batch_time
        )

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
        
        # 1. Extract Data for the current time step
        # We assume all envs are synchronized in time for memory efficiency.
        t = state.time_idx[0] 
        exo_batch = jax.tree.map(lambda x: x[t], self.shared_exo_data)
        
        # 2. VMAP the Simulator Step
        # Simulator returns: (next_sim, SystemOutputs)
        next_sims, batched_outputs = jax.vmap(JAXSimulator.step, in_axes=(0, 0, None))(
            state.sim, 
            actions,  
            exo_batch
        )

        # 3. VMAP the Cost Calculation
        # We write a small wrapper to neatly pass the batched elements to f_cost_step
        def calc_reward(sim_k, act_k, out_k):
            cost = f_cost_step(
                state=sim_k.state,
                actions=act_k,
                outputs=out_k,
                exogenous=exo_batch,        # Shared across batch
                configs=sim_k.configs,
                dt_seconds=sim_k.dt_seconds
            )
            return -cost  # Reward is negative cost
            
        # Map over the simulators, actions, and outputs. 
        rewards = jax.vmap(calc_reward, in_axes=(0, 0, 0))(
            state.sim, actions, batched_outputs
        )
        
        # 4. Update State
        next_state = EnvState(
            sim=next_sims, 
            time_idx=state.time_idx + 1
        )
        
        # 5. Check Termination (if end of dataset)
        # Returns a boolean array of shape (num_envs,)
        done = next_state.time_idx >= (len(self.shared_exo_data.price) - 1)
        
        # 6. Populate Info Dictionary
        # This gives your RL logger access to exactly what happened physically
        info = {
            "outputs": batched_outputs,
            "bat_soc": next_sims.battery.soc,
            "room_temps": next_sims.thermal.T_vector[:, jnp.array(self.sim_template.thermal.config.room_air_indices)]
        }
        
        return next_state, rewards, done, info