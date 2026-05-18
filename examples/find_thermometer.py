import jax
import jax.numpy as jnp
from energysim.sim.simulator import JAXSimulator
from energysim.rl.vector_env import VectorizedEnergyEnv
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import RewardConfig, BatteryConfig, HeatPumpConfig, AirConditionerConfig, SystemActions
from rl_house import create_2_room_house

print("--- 🗺️ THE THERMOMETER MAP ---")

dataset = SimulationDataset(file_path="fixed_master.csv", dt_seconds=900)
t_config = create_2_room_house()
sim = JAXSimulator(
    dt_seconds=900, t_config=t_config, r_config=RewardConfig(), 
    b_config=BatteryConfig(), hp_config=HeatPumpConfig(), ac_config=AirConditionerConfig()
)
env = VectorizedEnergyEnv(sim, dataset, num_envs=1)
state = env.reset(jax.random.PRNGKey(0))

# Extract the actual names of the nodes from the house builder
try:
    names = t_config.node_names
except AttributeError:
    names = [f"Node {i}" for i in range(state.sim.thermal.T_vector.shape[1])]

print("\n🌡️ Initial Temperatures (Heater OFF):")
for i, name in enumerate(names):
    print(f"Index {i} ({name}): {state.sim.thermal.T_vector[0, i]:.4f}°C")

# Blast 5000W into the house
actions = SystemActions(
    battery_power_w=jnp.array([0.0]),
    heat_pump_power_w=jnp.array([[5000.0, 5000.0]]), 
    ac_power_w=jnp.array([[0.0, 0.0]]),
    storage_discharge_w=jnp.array([[0.0, 0.0]])
)
state, _, _, _ = env.step(state, actions)

print("\n🔥 Temperatures (After 5000W Heater ON):")
for i, name in enumerate(names):
    print(f"Index {i} ({name}): {state.sim.thermal.T_vector[0, i]:.4f}°C")