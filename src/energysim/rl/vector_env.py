import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import Tuple, Dict, Optional

from ..sim.simulator import JAXSimulator
from ..core.shared.data_structs import SystemState, SystemActions, ExogenousData
from ..core.data.dataset import SimulationDataset
from ..behavior.base import AbstractBehavioralModel
from ..sim.helpers import precalculate_exogenous_data

class VectorizedEnergyEnv(eqx.Module):
    """
    Memory-Optimized Vectorized Environment.
    
    Stores a SINGLE copy of exogenous data (weather/prices) and broadcasts it 
    to all environments using JAX vmap, saving massive amounts of VRAM.
    """
    sim: JAXSimulator 
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
        
        # 1. Setup Simulator Template
        self.sim = simulator
        
        # 2. Get dummy state for behavioral models (if they require checking SOC etc)
        # In a vectorized setting, we usually use a "nominal" state for pre-calc
        dummy_state = simulator.reset()
        
        # 3. Determine room count for vector splitting
        n_rooms = len(simulator.thermal.config.room_air_indices)

        print("Pre-calculating data on CPU...")
        
        # 4. Pre-calculate and Upload
        self.shared_exo_data = precalculate_exogenous_data(
            dataset=dataset,
            behavioral_models=behavioral_models or {},
            dt_seconds=simulator.dt_seconds,
            n_rooms=n_rooms,
            dummy_state=dummy_state
        )
        
        print(f"Data is on device: {self.shared_exo_data.ambient_temp.device}")

    def reset(self, key: jax.Array) -> SystemState:
        """Returns the initial state replicated across num_envs."""
        single_sim = self.sim.reset()
        # We physically replicate state because each env diverges quickly
        def replicate(leaf):
            return jnp.repeat(leaf[None, ...], self.num_envs, axis=0)
        return jax.tree.map(replicate, single_sim)

    @jax.jit
    def step(self, sims, actions, t_idx):
        # Slice data
        t = t_idx[0]
        exo = jax.tree.map(lambda x: x[t], self.shared_exo_data)
        
        # Pure Magic: VMAP the Simulator directly
        # This automatically maps over the 'state' leaves inside 'sims'
        # and broadcasts the 'config' leaves inside 'sims'.
        next_sims, costs = jax.vmap(JAXSimulator.step, in_axes=(0, 0, None))(sims, actions, exo)
        
        return next_sims, costs