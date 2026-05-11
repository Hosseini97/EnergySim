import jax
import jax.numpy as jnp
import equinox as eqx
import time

# Import the core Energysim modules based on your architecture
from energysim.control.baselines import BangBangThermostat, TimeOfUseBattery, CompositeBaseline
from energysim.core.shared.data_structs import (
    ThermalConfig, HeatPumpConfig, AirConditionerConfig, BatteryConfig, RewardConfig
)
# Assuming you have a way to initialize the VectorEnv in energysim.rl.vector_env
# Replace this import with however Energysim normally initializes the environment
from energysim.rl.vector_env import VectorizedEnergyEnv
from energysim.core.data.dataset import SimulationDataset
from energysim.sim.simulator import JAXSimulator

# If you have a factory/builder for the simulator, import it here too.
# For example, earlier you mentioned a factory.py file:
# from energysim.core.models.factory import build_default_simulator

from build_my_house import create_2_room_house

import sample_data_generator

def main():
    print("--- 🔬 Energysim RBC Baseline Evaluation ---")
    
   # ==========================================
    # 1. Initialize the Environment
    # ==========================================
    print("Loading Dataset and Physics Configs...")
    dt_seconds = 900 
    
    # Use the SAME master file as the PPO agent
    dataset_path = "/home/emmanuel-gendy/Documents/ppo_jax/heeten_training_master.csv"
    dataset = SimulationDataset(
        file_path=dataset_path, 
        dt_seconds=dt_seconds
    )
    
    print("Building Simulator...")
    # 3. Load the EXACT physics used by your RL agent
    t_config = create_2_room_house()
    
    r_config = RewardConfig(price_weight=1.0, comfort_weight=5.0)
    b_config = BatteryConfig()
    hp_config = HeatPumpConfig()
    ac_config = AirConditionerConfig()
    
    # 4. Instantiate the Simulator
    simulator = JAXSimulator(
        dt_seconds=dt_seconds,
        t_config=t_config,
        r_config=r_config,
        b_config=b_config,
        hp_config=hp_config,
        ac_config=ac_config
    )
    
    print("Pre-calculating exogenous data and building Vectorized Env...")
    num_envs = 2048
    env = VectorizedEnergyEnv(
        simulator=simulator,
        dataset=dataset,
        num_envs=num_envs
    )
    
    # Get initial state
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    # ==========================================
    # 2. Instantiate the RBC Controllers
    # ==========================================
    print("Instantiating Composite RBC (BangBang HVAC + TOU Battery)...")
    
    hvac_rbc = BangBangThermostat(t_config, hp_config, ac_config)
    
    # We set TOU thresholds: Charge if price in bottom 20%, Discharge if top 20%
    # (You may need to pass actual absolute prices depending on the dataset scale)
    battery_rbc = TimeOfUseBattery(b_config, price_low_threshold=0.10, price_high_threshold=0.30)
    
    # Combine them into the master baseline
    master_rbc = CompositeBaseline(hvac_rbc, battery_rbc)

    # We must VMAP the controller so it can handle `num_envs` simultaneously
    vmap_controller = jax.vmap(master_rbc, in_axes=(0, 0, None))

    # ==========================================
    # 3. The JIT-Compiled Execution Loop
    # ==========================================
    
    # CORRECTED VMAP: state is batched (0), exo is shared (None), dt is shared (None)
    vmap_controller = jax.vmap(master_rbc, in_axes=(0, None, None))
    
    # Move this OUTSIDE the function so the print statements at the bottom can use it!
    dataset_length = 960 # len(env.shared_exo_data.price)

    @eqx.filter_jit
    def run_full_dataset(init_state):
        def step_fn(curr_state, _):
            t = curr_state.time_idx[0]
            exo_batch = jax.tree.map(lambda x: x[t], env.shared_exo_data)
            actions, _ = vmap_controller(curr_state.sim.state, exo_batch, dt_seconds)
            next_state, rewards, done, info = env.step(curr_state, actions)
            return next_state, rewards
        
        # Ensure we only scan for 960 steps to match the PPO test
        _, all_rewards = jax.lax.scan(step_fn, init_state, None, length=960)
        return all_rewards

    # ==========================================
    # 4. Execute and Measure
    # ==========================================
    print("Starting JAX Compilation and Simulation...")
    start_time = time.time()
    
    # Run the simulation (Much cleaner now!)
    all_rewards = run_full_dataset(state)
    
    wall_time = time.time() - start_time
    
    # Calculate Metrics
    total_reward_per_env = jnp.sum(all_rewards, axis=0)
    mean_daily_cost = -jnp.mean(total_reward_per_env) / (dataset_length / (86400 / dt_seconds))
    
    print("-" * 40)
    print("✅ Baseline Evaluation Complete")
    print(f"Time Elapsed: {wall_time:.2f} seconds")
    print(f"Simulated {num_envs} buildings for {dataset_length} steps.")
    print(f"💰 RBC Average Daily Cost per Building: € {mean_daily_cost:.2f}")
    print("-" * 40)

if __name__ == "__main__":
    main()