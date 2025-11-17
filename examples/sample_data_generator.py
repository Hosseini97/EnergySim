# examples/sample_data_generator.py
import pandas as pd
import numpy as np

FILE_NAME = "examples/sample_data.csv"
N_STEPS = 192 # 48 hours at 15-min intervals (192 / 4 = 48)
DT_MINUTES = 15
DT_SECONDS = DT_MINUTES * 60

def create_sample_data():
    """
    Generates a 48-hour sample dataset and saves it to a CSV file.
    This creates all the columns required by the SimulationDataset.
    """
    print(f"Generating {FILE_NAME}...")
    
    # Time vector
    t = np.linspace(0, 4*np.pi, N_STEPS) # 2 full cycles (48 hours)
    
    # Weather
    ambient_temp = 10 + 10 * np.sin(t)
    solar_irradiance_w_m2 = np.fmax(0, 800 * np.sin(t))
    
    # Price
    price = 0.15 + 0.1 * np.sin(t + np.pi/2) + np.random.rand(N_STEPS) * 0.05
    price = np.fmax(0.05, price)
    
    # Base Loads & Gains
    base_load_w = 500 + 300 * np.sin(t + np.pi*1.5) + np.random.rand(N_STEPS) * 100
    base_load_w = np.fmax(200, base_load_w)
    
    # Passive gains
    occupancy_gains_w = np.fmax(0, 150 * np.sin(t + np.pi*1.5))
    solar_gains_w = np.fmax(0, 2000 * np.sin(t))
    
    # Create DataFrame with the exact column names from ExoKey
    df = pd.DataFrame({
        "ambient_temp": ambient_temp,
        "solar_irradiance_w_m2": solar_irradiance_w_m2, # Used by SolarModel
        "price": price,
        "load": base_load_w,                          # Maps to base_load_w
        "internal_gains_w": occupancy_gains_w,      # Maps to occupancy_gains_w
        "solar_gains_w": solar_gains_w,             # Heat from windows
    })
    
    df.to_csv(FILE_NAME, index=False)
    print(f"Successfully created {FILE_NAME}")

if __name__ == "__main__":
    create_sample_data()