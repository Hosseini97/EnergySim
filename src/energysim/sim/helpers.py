import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List

from ..core.data.dataset import SimulationDataset
from ..core.shared.data_structs import ExogenousData
from ..behavior.base import AbstractBehavioralModel

def precalculate_exogenous_data(
    dataset: SimulationDataset, 
    behavioral_models: Dict[str, AbstractBehavioralModel],
    dt_seconds: float,
    n_rooms: int,
    dummy_state: 'SystemState' = None
) -> ExogenousData:
    """
    Generates a single, static trace of exogenous data on the GPU.
    
    1. Runs stateful Python behavioral models (CPU) for the full horizon.
    2. Broadcasts scalar values (e.g. Solar Gain) to zonal vectors (n_rooms).
    3. Uploads the result to VRAM as a JAX DeviceArray.
    
    Result Shape: (Time, ...) or (Time, n_rooms) — NO Batch Dimension.
    """
    total_steps = len(dataset)
    
    # 1. Load Base Data (Weather, Price) from CSV
    # This returns a structure of JAX arrays shape (T, ...)
    base_data_struct = dataset.get_forecast(0, total_steps)
    
    # 2. Run Behavioral Models (Python Loop)
    # We use numpy here for speed in the Python loop
    behavioral_traces = {}
    internal_gains_acc = np.zeros(total_steps, dtype=np.float32)
    
    # Reset models
    for model in behavioral_models.values():
        model.reset()

    # We iterate *once* through time
    for t in range(total_steps):
        # Step every model
        for key, model in behavioral_models.items():
            field_name = f"{key}_load_w"
            
            # If model needs state, pass dummy, otherwise it might be ignored
            power = model.step(t, dt_seconds, dummy_state)
            
            if field_name not in behavioral_traces:
                behavioral_traces[field_name] = np.zeros(total_steps, dtype=np.float32)
            
            behavioral_traces[field_name][t] = power

            # Accumulate heat gains from specific devices
            if key in ["dishwasher", "cooking", "clothes_dryer"]:
                internal_gains_acc[t] += power

    # 3. Formatting & Zonal Splitting
    # The simulator expects vectors for: solar_gains_w, occupancy_gains_w, device_gains_w
    zonal_fields = ["solar_gains_w", "occupancy_gains_w"]
    
    new_data_dict = {}
    
    # Convert base struct to dict to iterate
    # Note: vars() on a dataclass/PyTree might not work directly depending on implementation
    # using getattr is safer for Eqx modules/Flax dataclasses
    # If it's a Flax dataclass:
    field_names = base_data_struct.__dataclass_fields__.keys()
    
    for field in field_names:
        original_data = getattr(base_data_struct, field)
        
        # A. Behavioral Override (e.g. 'ev_charger_load_w')
        if field in behavioral_traces:
            final_data = jnp.array(behavioral_traces[field])
            
        # B. Device Gains (Calculated Accumulator) -> Split to Zones
        elif field == "device_gains_w":
            # Shape (T,) -> (T, n_rooms)
            # Divide total heat equally among rooms
            final_data = jnp.tile(
                jnp.array(internal_gains_acc)[:, None], (1, n_rooms)
            ) / n_rooms
            
        # C. Scalar Environmental Gains -> Split to Zones
        elif field in zonal_fields:
            # If the CSV gave us a scalar (T,), we must broadcast to (T, n_rooms)
            if original_data.ndim == 1:
                final_data = jnp.tile(
                    original_data[:, None], (1, n_rooms)
                ) / n_rooms
            else:
                final_data = original_data
                
        # D. Pass through everything else (Price, Ambient Temp)
        else:
            final_data = original_data
            
        new_data_dict[field] = final_data

    # 4. Construct & Upload
    exo_struct = ExogenousData(**new_data_dict)
    
    # Move to GPU immediately. 
    # This creates the "8MB" structure that is cheap to pass around.
    return jax.device_put(exo_struct)