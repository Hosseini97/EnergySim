import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np # Ensure numpy is imported for the callback formatting
import pandas as pd

from energysim.sim.simulator import JAXSimulator
from energysim.control.mpc_solver import JAX_MPC_Solver
from energysim.core.data.dataset import SimulationDataset
from energysim.utils.objectives import f_cost_step  # <--- Imported external cost function
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, 
    ThermalStorageConfig, PVConfig, SystemActions, SystemState, ExogenousData
)
import sample_data_generator
from build_my_house import create_2_room_house

def stack_pytree(list_of_trees):
    """Helper: Stacks a list of Pytrees into a single Pytree along a new time dimension."""
    return jax.tree.map(lambda *args: jnp.stack(args), *list_of_trees)

def run_mpc():
    # --- 1. Setup ---
    sample_data_generator.create_sample_data(n_days=2)
    dt = sample_data_generator.DT_SECONDS
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, dt)
    t_config = create_2_room_house()
    
    # Initialize the Simulator template
    sim = JAXSimulator(
        dt_seconds=dt,
        t_config=t_config,
        r_config=RewardConfig(price_weight=10.0, comfort_weight=50.0),
        b_config=BatteryConfig(),
        hp_config=HeatPumpConfig(),
        ac_config=AirConditionerConfig(),
        ts_config=ThermalStorageConfig(),
        pv_config=PVConfig()
    )
    
    HORIZON = 16
    # The MPC now takes the simulator directly as its perfect internal model
    mpc = JAX_MPC_Solver(N_horizon=HORIZON, simulator_template=sim)
    
    print(f"--- 2. Preparing JAX Data (Horizon={HORIZON}) ---")
    
    # Initialize prev_action with a SINGLE step shape to match mpc.solve's output
    initial_sim = sim.reset()
    
    # Pre-fetch ALL exogenous data into a single PyTree of JAX arrays
    # This prevents expensive Python-to-C++ memory transfers during the loop
    history_exo_list = [dataset[i] for i in range(len(dataset))]
    all_exo = stack_pytree(history_exo_list)
    
    # We stop the loop early so the final step still has a full HORIZON of future data
    step_indices = jnp.arange(len(dataset) - HORIZON)

    # --- 3. Define Debug Callback ---
    def debug_print(i, state: SystemState, action: SystemActions, cost, exo: ExogenousData):
        if i % 10 == 0:
            i = int(i)
            # 1. State Information
            t_vec = np.array(state.thermal.T_vector)
            idx = np.array(t_config.room_air_indices)
            temps = t_vec[idx]
            temps_str = ", ".join([f"{t:.2f}" for t in temps])
            
            tank_temps = np.array(state.storage.temperatures_c)
            tank_mean = np.mean(tank_temps)
            
            # 2. Action Information (What the optimizer chose)
            hp_w = np.array(action.heat_pump_power_w)
            ac_w = np.array(action.ac_power_w)
            bat_w = float(action.battery_power_w)
            stor_w = np.array(action.storage_discharge_w)
            
            # Format action arrays for clean printing
            hp_str = ", ".join([f"{w:.1f}" for w in hp_w])
            ac_str = ", ".join([f"{w:.1f}" for w in ac_w])
            stor_str = ", ".join([f"{w:.1f}" for w in stor_w])
            
            # 3. Exogenous Information
            amb_t = float(exo.ambient_temp)
            irr = float(exo.solar_irradiance_w_m2)
            
            print(f"\n=== Step {i:04d} ===")
            print(f"  Environment : T_Amb={amb_t:.1f}°C | Solar={irr:.1f} W/m2")
            print(f"  System State: T_Rooms=[{temps_str}]°C | Tank_Avg={tank_mean:.1f}°C | BatSOC={float(state.battery.soc):.2f}")
            print(f"  MPC Actions : HP_W=[{hp_str}] | AC_W=[{ac_str}] | Stor_W=[{stor_str}] | Bat_W={bat_w:.1f}")
            print(f"  Step Cost   : {float(cost):.4f}")

    # --- 4. Define the Compiled Scan Step ---
    def scan_step(carry, idx):
        current_sim, = carry
        
        # 1. Carve out the FORECAST horizon natively in XLA
        exo_forecast = jax.tree.map(
            lambda arr: jax.lax.dynamic_slice_in_dim(arr, idx, HORIZON), 
            all_exo
        )
        
        # 2. Carve out the CURRENT step data natively in XLA
        exo_current = jax.tree.map(lambda arr: arr[idx], all_exo)
        
        # 3. Solve Optimal Control Problem (Passing the FULL simulator)
        action = mpc.solve(
            current_sim=current_sim,
            exo_forecast=exo_forecast, 
            warm_start_norm_actions=None
        )
        
        # 4. Step Simulator (Returns SystemOutputs struct instead of cost)
        new_sim, outputs = current_sim.step(action, exo_current)
        
        # 5. Calculate Cost externally
        cost = f_cost_step(
            state=current_sim.state,
            actions=action,
            outputs=outputs,
            exogenous=exo_current,
            configs=current_sim.configs,
            dt_seconds=current_sim.dt_seconds
        )
        
        # 6. Trigger terminal print
        jax.debug.callback(debug_print, idx, current_sim.state, action, cost, exo_current)
        
        # Pack carry for next loop
        next_carry = (new_sim, )
        
        # Flat dict for history (Easier to pass straight to Pandas later)
        output_metrics = {
            "step": idx,
            "cost": cost,
            "pv_gen_w": outputs.pv.pv_generation_w,
            "bat_soc": new_sim.battery.soc
        }
        
        return next_carry, output_metrics

    print("--- 3. Running Compiled MPC Simulation ---")
    initial_carry = (initial_sim, )
    
    # Run the entire simulation in a single compiled XLA call
    final_carry, history_dict = jax.lax.scan(
        scan_step, 
        initial_carry, 
        step_indices
    )

    print("--- Simulation Complete ---")
    
    # Convert history directly to DataFrame (optional)
    df = pd.DataFrame(history_dict)
    print("\nSummary of first 5 steps:")
    print(df.head())

if __name__ == "__main__":
    run_mpc()