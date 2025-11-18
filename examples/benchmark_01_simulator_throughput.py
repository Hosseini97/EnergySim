import time
import jax
import jax.numpy as jnp
import equinox as eqx
from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, 
    ThermalStorageConfig, SolarConfig, SystemActions
)
import sample_data_generator
from build_my_house import create_2_room_house

def benchmark():
    print("--- Benchmark: Raw JAX Simulator Throughput ---")
    
    # 1. Massive Dataset (1 Year)
    N_DAYS = 365
    sample_data_generator.create_sample_data(n_days=N_DAYS)
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, 900)
    
    # Load all data into GPU memory
    all_exo = dataset.get_forecast(0, len(dataset))
    
    # 2. Init Simulator
    t_config = create_2_room_house()
    
    # --- Broadcast Exogenous Data to Zones ---
    # The thermal model expects inputs for *every* room.
    # The dataset provides 1 scalar per timestep. We must tile it to (T, n_rooms).
    n_rooms = len(t_config.room_air_indices) # 2

    def broadcast_to_rooms(arr):
        # If array is shape (Time,), expand to (Time, n_rooms)
        if arr.ndim == 1:
            return jnp.tile(arr[:, None], (1, n_rooms))
        return arr

    # Use tree_at to update the specific fields in the immutable structure
    all_exo = eqx.tree_at(
        lambda d: (d.solar_gains_w, d.occupancy_gains_w, d.device_gains_w),
        all_exo,
        (
            broadcast_to_rooms(all_exo.solar_gains_w),
            broadcast_to_rooms(all_exo.occupancy_gains_w),
            broadcast_to_rooms(all_exo.device_gains_w)
        )
    )
    # -------------------------------------------------------

    sim = JAXSimulator(
        dt_seconds=900, t_config=t_config, r_config=RewardConfig(),
        b_config=BatteryConfig(), hp_config=HeatPumpConfig(), 
        ac_config=AirConditionerConfig(), ts_config=ThermalStorageConfig(), 
        s_config=SolarConfig()
    )
    
    # 3. Create Scan Function
    dummy_action_template = SystemActions(
        battery_power_w=jnp.array(0.0),
        heat_pump_power_w=jnp.zeros(n_rooms), # Ensure this matches n_rooms
        ac_power_w=jnp.zeros(n_rooms),
        storage_discharge_w=jnp.zeros(n_rooms)
    )

    @jax.jit
    def run_full_year(init_state, exo_data):
        init_carry = (init_state, dummy_action_template)
        
        def step_fn(carry, exo):
            curr_state, prev_act = carry
            
            # Update action using tree_at (Equinox style)
            current_action = eqx.tree_at(
                lambda a: a.battery_power_w,
                dummy_action_template,
                jnp.array(0.0) # Dummy dynamic value
            )
            
            new_sim, cost = sim.step(current_action, prev_act, exo)
            return (new_sim.state, current_action), cost

        (final_state, _), costs = jax.lax.scan(step_fn, init_carry, exo_data)
        return costs

    # 4. Warmup
    print("Compiling JAX Kernel...")
    start = time.perf_counter()
    run_full_year(sim.state, all_exo).block_until_ready()
    print(f"Compilation finished in {time.perf_counter() - start:.4f}s")

    # 5. Benchmark
    print(f"Running {len(dataset)} steps...")
    start = time.perf_counter()
    run_full_year(sim.state, all_exo).block_until_ready()
    end = time.perf_counter()
    
    duration = end - start
    sps = len(dataset) / duration
    print(f"\nRESULTS:")
    print(f"Time: {duration:.4f} seconds")
    print(f"Speed: {sps:,.0f} steps/second")

if __name__ == "__main__":
    benchmark()