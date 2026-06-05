# data_loader.py
import pandas as pd
import numpy as np

class RealWorldDataLoader:
    def __init__(self, csv_path: str, steps_per_day: int = 96):
        print(f"Loading real-world dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.steps_per_day = steps_per_day
        
        # Ensure we have enough data to sample a full day
        self.max_start_idx = len(self.df) - self.steps_per_day
        
        # Extract features as numpy arrays for fast slicing during training
        # Adjust column names if your final CSV differs slightly
        self.price = self.df['price'].values # Assuming EUR/kWh
        self.t_out = self.df['ambient_temperature'].values
        self.irr = self.df['solar_irradiance'].values
        
        # sample_data_generator.py converted load/pv to Watts. Convert back to kW.
        self.load_kw = self.df['load'].values / 1000.0
        self.pv_kw = self.df['solar_gains_w'].values / 1000.0
        
        # Parse hour of day from timestamp
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.hour = self.df['timestamp'].dt.hour.values

    def sample_episode(self) -> dict:
        """Returns a 24-hour (96-step) slice of real-world data."""
        start_idx = np.random.randint(0, self.max_start_idx)
        end_idx = start_idx + self.steps_per_day
        
        return {
            'hour': self.hour[start_idx:end_idx],
            'price': self.price[start_idx:end_idx],
            't_out': self.t_out[start_idx:end_idx],
            'irr': self.irr[start_idx:end_idx],
            'load': self.load_kw[start_idx:end_idx],
            'pv': self.pv_kw[start_idx:end_idx]
        }