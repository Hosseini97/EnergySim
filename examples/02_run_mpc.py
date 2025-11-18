import jax.numpy as jnp
import equinox as eqx  # <--- Required for tree_at
from energysim.sim.simulator import JAXSimulator
from energysim.control.mpc_solver import JAX_MPC_Solver
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, 
    ThermalStorageConfig, SolarConfig, SystemActions
)
import sample_data_generator
from build_my_house import create_2_room_house

def run_mpc():
    # 1. Setup
    sample_data_generator.create_sample_data(n_days=2)
    dt = sample_data_generator.DT_SECONDS
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, dt)
    t_config = create_2_room_house()
    n_rooms = int(len(t_config.room_air_indices)) # 2
    
    # Common configs
    configs = {
        "dt_seconds": dt,
        "t_config": t_config,
        "r_config": RewardConfig(price_weight=10.0, comfort_weight=50.0),
        "b_config": BatteryConfig(),
        "hp_config": HeatPumpConfig(),
        "ac_config": AirConditionerConfig(),
        "ts_config": ThermalStorageConfig(),
        "s_config": SolarConfig()
    }

    sim = JAXSimulator(**configs)
    
    # Initialize MPC with 4-hour horizon (16 steps)
    HORIZON = 16
    mpc = JAX_MPC_Solver(N_horizon=HORIZON, **configs)
    
    state = sim.state
    # Initialize previous action for slew rate calculation
    prev_action = mpc.zonal_warm_start # Use zero start

    print(f"Starting MPC Simulation (Horizon={HORIZON})...")
    
    # Define distribution logic (e.g. 60% heat to Living Room, 40% to Bedroom)
    split_factors = jnp.array([0.6, 0.4])

    for i in range(len(dataset) - HORIZON):
        # --- 1. Prepare FORECAST Data (Horizon) ---
        # Dataset returns scalars: shape (HORIZON,)
        raw_forecast = dataset.get_forecast(i, HORIZON)

        # Broadcast scalars to zonal vectors: shape (HORIZON, n_rooms)
        # We use jnp.outer(A, B) -> if A is (H,), B is (N,), Result is (H, N)
        exo_forecast = eqx.tree_at(
            lambda e: (e.solar_gains_w, e.occupancy_gains_w, e.device_gains_w),
            raw_forecast,
            (
                jnp.outer(raw_forecast.solar_gains_w, split_factors),     # (H, 2)
                jnp.outer(raw_forecast.occupancy_gains_w, split_factors), # (H, 2)
                jnp.zeros((HORIZON, n_rooms))                             # (H, 2)
            )
        )

        # --- 2. Solve Optimal Control Problem ---
        # Returns the optimal action for the *first* step of the horizon
        action = mpc.solve(state, exo_forecast, warm_start_actions=None)
        
        # --- 3. Prepare CURRENT Data (Single Step) ---
        # Dataset returns scalars: shape ()
        raw_current = dataset[i]

        # Broadcast scalars to zonal vectors: shape (n_rooms,)
        exo_current = eqx.tree_at(
            lambda e: (e.solar_gains_w, e.occupancy_gains_w, e.device_gains_w),
            raw_current,
            (
                raw_current.solar_gains_w * split_factors,     # (2,)
                raw_current.occupancy_gains_w * split_factors, # (2,)
                jnp.zeros(n_rooms)                             # (2,)
            )
        )
        
        # --- 4. Apply to Reality (Simulator) ---
        new_sim, cost = sim.step(action, prev_action, exo_current)
        state = new_sim.state
        prev_action = action

        if i % 10 == 0:
            temps = state.thermal.T_vector[jnp.array(t_config.room_air_indices)]
            print(f"Step {i}: Cost={cost:.4f} | T_Rooms={temps} | BatSOC={state.battery.soc:.2f}")

if __name__ == "__main__":
    run_mpc()