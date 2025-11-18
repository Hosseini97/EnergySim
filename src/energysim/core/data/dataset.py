# energysim/core/data/dataset.py
from typing import Callable
import numpy as np
import pandas as pd
from energysim.core.shared.data_structs import ExogenousData
from energysim.core.shared.control_variables import ExoKey
import jax.numpy as jnp

class SimulationDataset:
    """
    Loads time-series data from a file and serves it step-by-step.
    Initializes all behavioral and calculated fields to 0.0.
    """
    def __init__(self, file_path: str, dt_seconds: int, read_fn: Callable[[str], pd.DataFrame] = pd.read_csv):
        df = read_fn(file_path)

        self.dt_seconds = dt_seconds

        # Assume total_steps is based on a required column
        self.total_steps = len(df)

        # --- Helper function to safely load columns ---
        def load_col_or_zeros(key: ExoKey) -> np.ndarray:
            if key in df.columns:
                return df[key].to_numpy(dtype=np.float32)
            else:
                print(f"Warning: Column '{key}' not found in data. Defaulting to 0.0.")
                return np.zeros(self.total_steps, dtype=np.float32)

        # Store data as lightweight NumPy arrays
        
        # --- Weather ---
        self.ambient_temp = load_col_or_zeros(ExoKey.AMBIENT_TEMP)
        self.solar_irradiance_w_m2 = load_col_or_zeros(ExoKey.SOLAR_IRRADIANCE) # <--- RENAMED
        self.wind_speed_m_s = load_col_or_zeros(ExoKey.WIND_SPEED_M_S)
        
        # --- Price ---
        self.price = load_col_or_zeros(ExoKey.PRICE)
        
        # --- Loads ---
        self.base_load_w = load_col_or_zeros(ExoKey.LOAD) # <--- RENAMED
        
        # --- Thermal Gains ---
        self.occupancy_gains_w = load_col_or_zeros(ExoKey.INTERNAL_GAINS_W) # <--- RENAMED
        self.solar_gains_w = load_col_or_zeros(ExoKey.SOLAR_GAINS_W)
        
        # Note: 'pv' is removed as it's now 'solar_irradiance_w_m2'

    def __len__(self) -> int:
        return self.total_steps

    def __getitem__(self, idx: int) -> ExogenousData:
        """Returns data for a single step, converting to JAX arrays."""
        
        # All behavioral/calculated fields are initialized to 0.0
        # The environment (e.g., EnergySimEnv) is responsible for filling them.
        return ExogenousData(
            # --- Weather ---
            ambient_temp=jnp.array(self.ambient_temp[idx]),
            solar_irradiance_w_m2=jnp.array(self.solar_irradiance_w_m2[idx]),
            wind_speed_m_s=jnp.array(self.wind_speed_m_s[idx]),
            # --- Price ---
            price=jnp.array(self.price[idx]),
            # --- Loads ---
            base_load_w=jnp.array(self.base_load_w[idx]),
            ev_charger_load_w=jnp.array(0.0),
            dishwasher_load_w=jnp.array(0.0),
            clothes_dryer_load_w=jnp.array(0.0),
            water_heater_load_w=jnp.array(0.0),
            cooking_load_w=jnp.array(0.0),
            # --- Thermal Gains ---
            occupancy_gains_w=jnp.array(self.occupancy_gains_w[idx]),
            solar_gains_w=jnp.array(self.solar_gains_w[idx]),
            device_gains_w=jnp.array(0.0)
        )

    def get_forecast(self, start_idx: int, horizon: int) -> ExogenousData:
        """Returns a slice of data for MPC forecasts."""
        s = slice(start_idx, start_idx + horizon)
        
        # Create zero arrays for all behavioral/calculated fields
        zeros = jnp.zeros(horizon)
        
        return ExogenousData(
            # --- Weather ---
            ambient_temp=jnp.array(self.ambient_temp[s]),
            solar_irradiance_w_m2=jnp.array(self.solar_irradiance_w_m2[s]),
            wind_speed_m_s=jnp.array(self.wind_speed_m_s[s]),
            # --- Price ---
            price=jnp.array(self.price[s]),
            # --- Loads ---
            base_load_w=jnp.array(self.base_load_w[s]),
            ev_charger_load_w=zeros,
            dishwasher_load_w=zeros,
            clothes_dryer_load_w=zeros,
            water_heater_load_w=zeros,
            cooking_load_w=zeros,
            # --- Thermal Gains ---
            occupancy_gains_w=jnp.array(self.occupancy_gains_w[s]),
            solar_gains_w=jnp.array(self.solar_gains_w[s]),
            device_gains_w=zeros
        )