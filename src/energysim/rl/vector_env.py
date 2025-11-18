import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import Tuple, Dict, Optional

from ..sim.simulator import JAXSimulator
from ..sim.pure_ops import PureSimulationOps
from ..core.shared.data_structs import SystemState, SystemActions, ExogenousData
from ..core.data.dataset import SimulationDataset
from ..behavior.base import AbstractBehavioralModel

class VectorizedEnergyEnv(eqx.Module):
    """
    Memory-Optimized Vectorized Environment.
    
    Stores a SINGLE copy of exogenous data (weather/prices) and broadcasts it 
    to all environments using JAX vmap, saving massive amounts of VRAM.
    """
    # --- JAX Fields (Dynamic / Device Arrays) ---
    ops: PureSimulationOps
    shared_exo_data: ExogenousData # Shape: (Time, ...) NOT (Time, Batch, ...)
    
    # --- Static Configuration ---
    num_envs: int = eqx.field(static=True)
    dt_seconds: float = eqx.field(static=True)
    
    # --- Helper Data ---
    _init_state_template: SystemState 

    def __init__(
        self, 
        simulator: JAXSimulator, 
        dataset: SimulationDataset,
        num_envs: int,
        behavioral_models: Optional[Dict[str, AbstractBehavioralModel]] = None
    ):
        self.num_envs = num_envs
        self.dt_seconds = simulator.dt_seconds
        
        self.ops = PureSimulationOps(simulator)
        self._init_state_template = simulator.reset()
        
        print(f"Pre-calculating logic (Shared Mode)...")
        # We generate ONE trajectory of exogenous data that is shared by all envs
        self.shared_exo_data = self._precalculate_shared_exogenous(
            simulator, dataset, behavioral_models or {}
        )
        
        # Check Device
        device = self.shared_exo_data.ambient_temp.device
        print(f"Data uploaded to: {device}")

    def _precalculate_shared_exogenous(
        self, 
        sim: JAXSimulator, 
        dataset: SimulationDataset, 
        models: Dict
    ) -> ExogenousData:
        """Generates a single (T, ...) trace of data shared by all envs."""
        total_steps = len(dataset)
        base_exo_struct = dataset.get_forecast(0, total_steps) 
        
        # 1. Run Python behavioral models (CPU)
        updates = {}
        internal_gains_acc = np.zeros(total_steps, dtype=np.float32)
        dummy_state = sim.reset()

        for key, model in models.items():
            field_name = f"{key}_load_w"
            profile = np.zeros(total_steps, dtype=np.float32)
            model.reset()
            for t in range(total_steps):
                profile[t] = model.step(t, sim.dt_seconds, dummy_state)
            
            updates[field_name] = profile
            if key in ["dishwasher", "cooking", "clothes_dryer"]:
                internal_gains_acc += profile

        # 2. Handle Zonal Splitting (But NO Batch Replication)
        n_rooms = len(sim.initial_thermal.config.room_air_indices)
        zonal_fields = ["solar_gains_w", "occupancy_gains_w"]
        new_data = {}
        
        for field in vars(base_exo_struct):
            if field.startswith("_"): continue
            original_data = getattr(base_exo_struct, field)
            
            if field in updates:
                # Behavioral override
                final_data = jnp.array(updates[field])
            elif field == "device_gains_w":
                # Split accumulator to zones: (T,) -> (T, n_rooms)
                final_data = jnp.tile(
                    jnp.array(internal_gains_acc)[:, None], (1, n_rooms)
                ) / n_rooms
            elif field in zonal_fields and original_data.ndim == 1:
                 # Split scalar weather to zones: (T,) -> (T, n_rooms)
                 final_data = jnp.tile(
                     original_data[:, None], (1, n_rooms)
                 ) / n_rooms
            else:
                # Already correct shape (T, ...) or (T, n_rooms)
                final_data = original_data

            new_data[field] = final_data

        # Move to GPU immediately
        return jax.device_put(ExogenousData(**new_data))

    def reset(self, key: jax.Array) -> SystemState:
        """Returns the initial state replicated across num_envs."""
        # We physically replicate state because each env diverges quickly
        def replicate(leaf):
            return jnp.repeat(leaf[None, ...], self.num_envs, axis=0)
        return jax.tree.map(replicate, self._init_state_template)

    @jax.jit
    def step(
        self, 
        states: SystemState, 
        actions: SystemActions, 
        time_indices: jax.Array
    ) -> Tuple[SystemState, jax.Array]:
        """
        Vectorized step with Shared Exogenous Data.
        """
        # 1. Slice the Shared Data
        # We assume time is synced: t = time_indices[0]
        # Slice shape: (T, ...) -> (...)
        t = time_indices[0]
        exo_slice = jax.tree.map(lambda x: x[t], self.shared_exo_data)

        # 2. Run VMAP
        # states:  (Batch, ...) -> Map (0)
        # actions: (Batch, ...) -> Map (0)
        # exo:     (...)        -> Broadcast (None)
        next_states, costs = jax.vmap(
            self.ops.step, in_axes=(0, 0, None)
        )(states, actions, exo_slice)
        
        return next_states, costs