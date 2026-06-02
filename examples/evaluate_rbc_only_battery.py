import jax
import jax.numpy as jnp
import equinox as eqx
import time
import matplotlib.pyplot as plt

from energysim.control.baselines import BangBangThermostat, TimeOfUseBattery, CompositeBaseline
from energysim.core.shared.data_structs import (
    ThermalConfig, HeatPumpConfig, AirConditionerConfig, BatteryConfig, RewardConfig, SystemActions
)

import energysim.utils.objectives as obj
if hasattr(obj, 'f_stage_cost'):
    _original_f_stage_cost = obj.f_stage_cost
    
    def _cost_translator(*args, **kwargs):
        # 1. Translate the state keyword
        if 'state' in kwargs:
            kwargs['current_state'] = kwargs.pop('state')
            
        # 2. Translate other common old keywords
        if 'action' in kwargs:
            kwargs['actions'] = kwargs.pop('action')
        if 'dt' in kwargs:
            kwargs['dt_seconds'] = kwargs.pop('dt')
            
        # 3. SATISFY THE MISSING REQUIREMENT
        # vector_env doesn't pass next_state, but the new function requires it.
        # Since f_stage_cost immediately does `del next_state`, we can safely pass None.
        if 'next_state' not in kwargs and len(args) < 2:
            kwargs['next_state'] = None
            
        return _original_f_stage_cost(*args, **kwargs)

    # Tell Python to use our translator
    obj.f_cost_step = _cost_translator

    
from energysim.rl.vector_env import VectorizedEnergyEnv
from energysim.core.data.dataset import SimulationDataset
from energysim.sim.simulator import JAXSimulator
from build_my_house import create_2_room_house

def main():
    print("--- 🔬 Energysim RBC Baseline Evaluation (Battery Only) ---")
    
    dt_seconds = 900 
    dataset_path = "examples/sample_data.csv"
    dataset = SimulationDataset(file_path=dataset_path, dt_seconds=dt_seconds)
    
    t_config = create_2_room_house()
    r_config = RewardConfig(price_weight=1.0, comfort_weight=5.0)
    b_config = BatteryConfig()
    
    # We define components but will override the HP/AC outputs to zero
    simulator = JAXSimulator(
        dt_seconds=dt_seconds,
        t_config=t_config,
        r_config=r_config,
        b_config=b_config,
        hp_config=HeatPumpConfig(), # Will force to 0W
        ac_config=AirConditionerConfig() # Will force to 0W
    )
    
    num_envs = 2048
    env = VectorizedEnergyEnv(simulator=simulator, dataset=dataset, num_envs=num_envs)
    
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    # ==========================================
    # Battery-Only Controller Logic
    # ==========================================
    # Only use TimeOfUseBattery. 
    # We will wrap this to ensure HP/AC actions are always 0.
    battery_rbc = TimeOfUseBattery(b_config, price_low_threshold=0.015, price_high_threshold=0.035)
    
    def battery_only_controller(sim_state, exo, dt):
        # 1. Get baseline result for ONE building
        raw_result = battery_rbc(sim_state, exo, dt)
        
        # 2. Extract battery power
        if isinstance(raw_result, tuple):
            action_obj = next((x for x in raw_result if hasattr(x, 'battery_power_w')), None)
            bat_power = action_obj.battery_power_w
        else:
            bat_power = raw_result.battery_power_w
        
        # 3. CRITICAL FIX: Ensure battery power is a 0-D scalar () 
        # By squeezing it, we prevent the SOC from becoming a 2D array.
        bat_power = jnp.asarray(bat_power, dtype=jnp.float32).squeeze()
        
        # 4. PERFECT PADDING for a 2-Room House (1 + 2 + 2 + 2 = 7 total inputs)
        return SystemActions(
            battery_power_w=bat_power,                             # Shape ()
            heat_pump_power_w=jnp.zeros((2,), dtype=jnp.float32),  # Shape (2,)
            ac_power_w=jnp.zeros((2,), dtype=jnp.float32),         # Shape (2,)
            storage_discharge_w=jnp.zeros((2,), dtype=jnp.float32) # Shape (2,)
        )
    vmap_controller = jax.vmap(battery_only_controller, in_axes=(0, None, None))

    @eqx.filter_jit
    def run_full_dataset(init_state):
        def step_fn(curr_state, _):
            t = curr_state.time_idx[0]
            exo_batch = jax.tree.map(lambda x: x[t], env.shared_exo_data)
            
            # 1. Run the controller
            actions = vmap_controller(curr_state.sim.state, exo_batch, dt_seconds)
            
            # 2. Step the environment
            next_state, rewards, done, info = env.step(curr_state, actions)
            
            # 3. TRACK THE DATA FOR BUILDING 0
            # Ensure you use the correct attributes for your specific state structure!
            history = {
                "reward": rewards[0],
                "battery_power": actions.battery_power_w[0], 
                "outdoor_temp": exo_batch.ambient_temp, 
                "indoor_temp": curr_state.sim.state.thermal.T_vector[0], # <--- T_vector applied!
                "price": exo_batch.price,
                "soc": curr_state.sim.battery.soc[0]  
            }
            
            return next_state, history
        
        _, full_history = jax.lax.scan(step_fn, init_state, None, length=672)
        return full_history

    print("Starting Simulation...")
    # 1. Run the simulation (it now returns a dictionary)
    history = run_full_dataset(state)
    
    # 2. Extract the reward array from the dictionary
    rewards_array = history["reward"]
    
    # 3. Calculate your baseline metrics
    total_reward = jnp.sum(rewards_array, axis=0)
    avg_reward = jnp.mean(total_reward)
    
    prices = jnp.asarray(history["price"])
    battery_powers = jnp.asarray(history["battery_power"])
    if battery_powers.ndim > 1: battery_powers = battery_powers[:, 0]
    if prices.ndim > 1: prices = prices[:, 0]
    TAX_RATE = 0.19 
    GRID_FEE = 0.05 
    
    # Calculate the "Real" Retail Price the agent "feels"
    retail_prices = (prices * (1 + TAX_RATE)) + GRID_FEE
    total_cost_rbc = jnp.sum(retail_prices * battery_powers * (dt_seconds / 3600.0) / 100.0)
    print(f"💰 RBC Total Financial Cost: ${total_cost_rbc:.2f}")
    

    # ==========================================
    # 📊 Graph Generation (3-Panel Layout)
    # ==========================================
    # 1. Extract arrays
    battery_powers = jnp.asarray(history["battery_power"])
    outdoor_temps = jnp.asarray(history["outdoor_temp"])
    indoor_temps = jnp.asarray(history["indoor_temp"])
    prices = jnp.asarray(history["price"])
    socs = jnp.asarray(history["soc"])  
    
    # 2. Force 1D shapes for Building 0
    if battery_powers.ndim > 1: battery_powers = battery_powers[:, 0]
    if outdoor_temps.ndim > 1: outdoor_temps = outdoor_temps[:, 0]
    if prices.ndim > 1: prices = prices[:, 0]
    if socs.ndim > 1: socs = socs[:, 0]
        
    if indoor_temps.ndim > 1:
        if indoor_temps.shape[1] == 2048: indoor_temps = indoor_temps[:, 0]
        if indoor_temps.ndim > 1: indoor_temps = jnp.mean(indoor_temps, axis=-1)
            
    time_steps = jnp.arange(len(battery_powers))
    
    import matplotlib.pyplot as plt

    # Create a 3-panel figure
    fig, ( ax_price, ax_batt) = plt.subplots(2, 1, figsize=(12, 12), dpi=150, sharex=True)

    # --- PANEL 1: Thermal Dynamics ---
    '''
    ax_temp.set_ylabel('Temp (°C)', fontweight='bold')
    ax_temp.axhspan(20, 22, color='mediumseagreen', alpha=0.2, label='Comfort (20-22°C)')
    ax_temp.plot(time_steps, outdoor_temps, color='gray', linestyle='--', label='Outdoor')
    ax_temp.plot(time_steps, indoor_temps, color='navy', linewidth=2, label='Indoor')
    ax_temp.legend(loc='upper right')
    ax_temp.grid(True, linestyle=':', alpha=0.6)
    ax_temp.set_title('1. House Thermal Dynamics', fontweight='bold')
    '''
    # --- PANEL 2: Financial Logic (Price) ---
    ax_price.set_ylabel('Price ($/kWh)', color='darkorange', fontweight='bold')
    ax_price.plot(time_steps, prices, color='darkorange', linewidth=2, label='Grid Price')
    # Update these thresholds to whatever you used in TimeOfUseBattery!
    ax_price.axhline(0.04, color='red', linestyle=':', label='Discharge Trigger (0.04)')
    ax_price.axhline(0.00, color='blue', linestyle=':', label='Charge Trigger (0.00)')
    ax_price.legend(loc='upper right')
    ax_price.grid(True, linestyle=':', alpha=0.6)
    ax_price.set_title('1. RBC Financial Triggers', fontweight='bold')

    # --- PANEL 3: Physical Battery (Power & SOC) ---
    ax_batt.set_xlabel('Time Steps (15-min intervals)', fontweight='bold')
    ax_batt.set_ylabel('Power (W)', color='firebrick', fontweight='bold')
    
    # Power Dispatch
    ax_batt.plot(time_steps, battery_powers, color='firebrick', label='Dispatch (W)')
    ax_batt.fill_between(time_steps, 0, battery_powers, color='firebrick', alpha=0.2)
    ax_batt.tick_params(axis='y', labelcolor='firebrick')
    
    # SOC on secondary Y-axis
    ax_soc = ax_batt.twinx()
    ax_soc.set_ylabel('State of Charge (SOC)', color='purple', fontweight='bold')
    ax_soc.plot(time_steps, socs, color='purple', linewidth=2.5, label='SOC Level')
    ax_soc.set_ylim(-0.05, 1.05) # Keep SOC between 0% and 100%
    ax_soc.tick_params(axis='y', labelcolor='purple')
    
    # Combine legends for panel 3
    l1, lab1 = ax_batt.get_legend_handles_labels()
    l2, lab2 = ax_soc.get_legend_handles_labels()
    ax_batt.legend(l1 + l2, lab1 + lab2, loc='upper right')
    ax_batt.grid(True, linestyle=':', alpha=0.6)
    ax_batt.set_title('2. Physical Response', fontweight='bold')

    plt.tight_layout()
    plt.savefig("rbc_complete_analysis.png")
    plt.show()

if __name__ == "__main__":
    main()