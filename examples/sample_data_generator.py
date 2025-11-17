import pandas as pd
import numpy as np

FILE_NAME = "examples/sample_data.csv"
DT_MINUTES = 15
DT_SECONDS = DT_MINUTES * 60
STEPS_PER_DAY = 24 * (60 // DT_MINUTES) # 96 steps

def create_sample_data(n_days: int = 2):
    """
    Generates a sample dataset for the given number of days
    and saves it to a CSV file.
    """
    
    n_steps = n_days * STEPS_PER_DAY
    print(f"Generating {FILE_NAME} with {n_steps} steps ({n_days} days)...")

    # Time vector
    t = np.linspace(0, (n_days * 2) * np.pi, n_steps) # 2-day (48h) cycle
    
    # Weather
    ambient_temp = 10 + 10 * np.sin(t)
    solar_irradiance_w_m2 = np.fmax(0, 800 * np.sin(t))

    # Price
    price = 0.15 + 0.1 * np.sin(t + np.pi/2) + np.random.rand(n_steps) * 0.05
    price = np.fmax(0.05, price)

    # Base Loads & Gains
    base_load_w = 500 + 300 * np.sin(t + np.pi*1.5) + np.random.rand(n_steps) * 100
    base_load_w = np.fmax(200, base_load_w)
    
    # Passive gains
    occupancy_gains_w = np.fmax(0, 150 * np.sin(t + np.pi*1.5))
    solar_gains_w = np.fmax(0, 2000 * np.sin(t)) # Heat from windows

    # Create DataFrame with the exact column names from ExoKey
    df = pd.DataFrame({
        "ambient_temp": ambient_temp,
        "solar_irradiance_w_m2": solar_irradiance_w_m2,
        "price": price,
        "load": base_load_w,
        "internal_gains_w": occupancy_gains_w,
        "solar_gains_w": solar_gains_w,
    })

    df.to_csv(FILE_NAME, index=False)
    print(f"Successfully created {FILE_NAME}")

if __name__ == "__main__":
    create_sample_data(n_days=14) # Default to 2 weeks for benchmarks