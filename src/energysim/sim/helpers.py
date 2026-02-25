import numpy as np
import jax
import jax.numpy as jnp

import pandas as pd
import jax
import numpy as np

from typing import Dict, List, TYPE_CHECKING

from ..core.data.dataset import SimulationDataset
from ..core.shared.data_structs import ExogenousData
from ..behavior.base import AbstractBehavioralModel

if TYPE_CHECKING:
    from ..core.shared.data_structs import SystemState

def precalculate_exogenous_data(
    dataset: SimulationDataset, 
    behavioral_models: Dict[str, AbstractBehavioralModel],
    dt_seconds: float,
    n_rooms: int,
    dummy_state: "SystemState"
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
                for key, model in behavioral_models.items():
                    field_name = f"{key}_load_w"
                    power = model.step(t, dt_seconds, dummy_state)
                    
                    if field_name not in behavioral_traces:
                        behavioral_traces[field_name] = np.zeros(total_steps, dtype=np.float32)
                    behavioral_traces[field_name][t] = power

                    # Accumulate heat gains ONLY for devices inside the thermal envelope
                    # EV is typically outside. Water heater might be in a garage (adjust as needed).
                    if key in ["dishwasher", "cooking", "clothes_dryer"]:
                        internal_gains_acc[t] += power

    # 3. Formatting (No Zonal Splitting Here!)
    new_data_dict = {}
    field_names = base_data_struct.__dataclass_fields__.keys()

    for field in field_names:
        original_data = getattr(base_data_struct, field)

        # Special Case: If we simulated occupancy, override the dataset's native gain
        if field == "occupancy_gains_w" and "occupancy_load_w" in behavioral_traces:
            final_data = jnp.array(behavioral_traces["occupancy_load_w"])
        
        # Standard electrical behavioral overrides (EV, cooking, etc.)
        elif field in behavioral_traces:
            final_data = jnp.array(behavioral_traces[field])

        # Accumulator for all simulated device thermal gains
        elif field == "device_gains_w":
            final_data = jnp.array(internal_gains_acc)

        # Keep solar as a 1D scalar trace (Time,)
        elif field == "solar_gains_w":
            final_data = original_data 
        
        else:
            final_data = original_data

        new_data_dict[field] = final_data

    # 4. Construct & Upload
    exo_struct = ExogenousData(**new_data_dict)
    
    # Move to GPU immediately. 
    # This creates the "8MB" structure that is cheap to pass around.
    return jax.device_put(exo_struct)

def scan_history_to_df(history_pytree):
    """
    Automatically flattens a JAX scan history PyTree into a pandas DataFrame.
    """
    # 1. Flatten the nested structure and keep the paths (keys)
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(history_pytree)
    
    data_dict = {}
    for path, leaf in leaves_with_paths:
        # 2. Convert JAX's path object to a readable string (e.g., 'system.power_w')
        # keystr() returns paths like '.system.power_w', so we slice off the leading char if it's a dot.
        col_name = jax.tree_util.keystr(path)
        if col_name.startswith('.'):
            col_name = col_name[1:]
        
        # 3. Convert to NumPy array
        arr = np.asarray(leaf)
        
        # 4. Handle dimensions
        if arr.ndim == 1:
            data_dict[col_name] = arr
        elif arr.ndim == 2:
            # If it's a 2D array (Time x Features), split it into numbered columns
            for i in range(arr.shape[1]):
                data_dict[f"{col_name}[{i}]"] = arr[:, i]
        else:
            # Fallback for 3D+ arrays: store as lists in the DataFrame
            data_dict[col_name] = list(arr)
            
    return pd.DataFrame(data_dict)