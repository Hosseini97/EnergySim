# examples/04_run_calibration.py
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import jax.tree

from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig,
    ThermalStorageConfig, SolarConfig, SystemActions
)
from energysim.calibration.sysid import SystemIdentifier
import sample_data_generator
from build_my_house import create_2_room_house

def run_calibration():
    print("========================================================")
    print("   EnergySim System Identification (Calibration) Demo   ")
    print("========================================================")

    # --- Setup ---
    # Generate 1 day of sample data for setup
    sample_data_generator.create_sample_data(n_days=1) 
    dt = sample_data_generator.DT_SECONDS
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, dt)
    t_config = create_2_room_house()
    n_rooms = len(t_config.room_air_indices)

    # Prepare Data
    horizon = 96 // 2  # 12 hours (48 steps)
    split = jnp.array([0.5, 0.5])
    
    # Helper to prepare batch data
    raw_exo_seq = dataset.get_forecast(0, horizon)
    exo_seq = eqx.tree_at(
        lambda e: (e.solar_gains_w, e.occupancy_gains_w, e.device_gains_w),
        raw_exo_seq,
        (
            jnp.outer(raw_exo_seq.solar_gains_w, split),
            jnp.outer(raw_exo_seq.occupancy_gains_w, split),
            jnp.zeros((horizon, n_rooms))
        )
    )

    # ==========================================
    # Phase 1: Generate Ground Truth Data
    # ==========================================
    print("\n[1] Generating Ground Truth Data (The 'Real World')...")
    
    # --- Secret Physics ---
    # We ISOLATE the efficiency factor to ensure identifiability.
    TRUE_EFFICIENCY_FACTOR = 0.65 
    secret_t_config = t_config # Use nominal thermal config
    # ----------------------

    gt_sim = JAXSimulator(
        dt, secret_t_config, RewardConfig(), BatteryConfig(),
        HeatPumpConfig(), AirConditionerConfig(), ThermalStorageConfig(), SolarConfig()
    )

    # --- Stable Action Sequence (Max 1000W) ---
    key = jax.random.PRNGKey(42)
    MAX_POWER = 1000.0 # Reduced for stability
    
    # Heating for the first 6 hours (0 to 24 steps)
    hp_rand = jax.random.uniform(key, (horizon//2, n_rooms), minval=0, maxval=MAX_POWER)
    heat_phase = jnp.where(hp_rand > MAX_POWER / 2, MAX_POWER, 0.0) 
    
    # Cooling for the second 6 hours (24 to 48 steps)
    ac_rand = jax.random.uniform(key, (horizon//2, n_rooms), minval=0, maxval=MAX_POWER)
    cool_phase = jnp.where(ac_rand > MAX_POWER / 2, MAX_POWER, 0.0)
    
    actions_seq = SystemActions(
        battery_power_w=jnp.zeros(horizon),
        heat_pump_power_w=jnp.concatenate([heat_phase, jnp.zeros_like(cool_phase)]),
        ac_power_w=jnp.concatenate([jnp.zeros_like(heat_phase), cool_phase]),
        # Discharge storage to aid heating phase only (scaled down to 1x power)
        storage_discharge_w=jnp.concatenate([heat_phase * 1.0, jnp.zeros_like(cool_phase)])
    )
    # ------------------------------------------

    # Run the "Ground Truth" simulation manually to inject the efficiency factor
    def gt_step(carry, inputs):
        sim, prev_act = carry
        act, exo = inputs
        
        # Scale the heat source thermal power inputs (HP and Storage)
        scaled_act = eqx.tree_at(
            lambda a: (a.heat_pump_power_w, a.storage_discharge_w),
            act,
            (
                act.heat_pump_power_w * TRUE_EFFICIENCY_FACTOR, 
                act.storage_discharge_w * TRUE_EFFICIENCY_FACTOR
            )
        )
        
        real_sim, _ = sim.step(scaled_act, prev_act, exo)
        
        # Return the resulting state and the full action for the next step's prev_act
        return (real_sim, act), real_sim.state.thermal.T_vector

    # Initialize state
    init_sim = gt_sim.reset()
    # Use jax.tree.map to create a zeroed-out initial action with the correct structure
    prev_action_dummy = jax.tree.map(lambda x: jnp.zeros_like(x[0]), actions_seq)
    
    _, true_all_temps = jax.lax.scan(gt_step, (init_sim, prev_action_dummy), (actions_seq, exo_seq))
    
    # Extract only room temperatures (Observed)
    true_room_temps = true_all_temps[:, jnp.array(t_config.room_air_indices)]
    
    print(f"Ground Truth Generated. (Horizon: {horizon} steps)")
    print(f"True Efficiency: {TRUE_EFFICIENCY_FACTOR}")
    print(f"Initial Room Temp: {true_room_temps[0]}")
    print(f"Final Room Temp:   {true_room_temps[-1]}") # Should now be realistic

    # ==========================================
    # Phase 2: Setup Uncalibrated Model
    # ==========================================
    print("\n[2] Setting up Uncalibrated Model (The 'Guess')...")
    
    # The guess uses the nominal config (Eff=1.0) and nominal thermal properties.
    guess_sim = JAXSimulator(
        dt, t_config, RewardConfig(), BatteryConfig(),
        HeatPumpConfig(), AirConditionerConfig(), ThermalStorageConfig(), SolarConfig()
    )
    
    # ==========================================
    # Phase 3: Run Calibration
    # ==========================================
    print("\n[3] Running Gradient Descent Calibration...")
    
    sys_id = SystemIdentifier(guess_sim)
    
    # Run optimization
    calibrated_config, stats = sys_id.calibrate(
        true_room_temps=true_room_temps,
        actions=actions_seq, # Full, unscaled actions are input to the model
        exo_data=exo_seq,
        learning_rate=0.05,
        steps=400
    )

    # ==========================================
    # Phase 4: Results
    # ==========================================
    print("\n[4] Results Analysis")
    
    est_eff = stats["hvac_efficiency_factor"]
    error_eff = abs(est_eff - TRUE_EFFICIENCY_FACTOR) / TRUE_EFFICIENCY_FACTOR * 100
    
    print(f"------------------------------------------")
    print(f"Parameter          | Truth | Estimated | Error")
    print(f"-------------------|-------|-----------|------")
    print(f"HVAC Efficiency    | {TRUE_EFFICIENCY_FACTOR:.2f}  | {est_eff:.2f}      | {error_eff:.1f}%")
    print(f"------------------------------------------")
    
    print("\nInterpretation:")
    if error_eff < 5.0:
        print("SUCCESS: The system correctly identified the hidden HVAC inefficiency.")
    else:
        print(f"PARTIAL: Optimization converged to {est_eff:.2f} (Error: {error_eff:.1f}%).")
        print("The remaining error likely stems from numerical limitations or slight coupling with the conductance matrix.")

if __name__ == "__main__":
    run_calibration()