import jax
import jax.numpy as jnp
from energysim.sim.simulator import JAXSimulator
from energysim.core.shared.data_structs import (
    RewardConfig, BatteryConfig, HeatPumpConfig, AirConditionerConfig, SystemActions
)
from energysim.core.data.dataset import ExogenousData
from build_my_house import create_2_room_house

# 1. Setup a manual simulator
t_config = create_2_room_house()
sim = JAXSimulator(
    dt_seconds=900, t_config=t_config,
    r_config=RewardConfig(), b_config=BatteryConfig(),
    hp_config=HeatPumpConfig(), ac_config=AirConditionerConfig()
)

# 2. Create a "Max Heat" action
# Note: For 2 rooms, the shape must be (1, 2)
n_rooms = len(t_config.room_air_indices)
actions = SystemActions(
    battery_power_w=jnp.array([0.0]),
    heat_pump_power_w=jnp.array([[2000.0, 2000.0]]), 
    ac_power_w=jnp.zeros((1, n_rooms)),
    storage_discharge_w=jnp.zeros((1, n_rooms))
)

# 3. Step the physics manually
# REMOVED the jax.random.PRNGKey(0) as per your error message
state = sim.reset() 
print(f"Initial Temps: {state.thermal.T_vector}")

print("\n--- Testing 5 Steps of Maximum Heating ---")
for i in range(5):
    # Dummy cold weather (0°C)
    exo = ExogenousData(
        timestamp=0, ambient_temp=0.0, solar_irradiance_w_m2=0.0, 
        price=0.1, load=0.0, pv=0.0
    )
    
    # Run one physics step
    state, reward, info = sim.step(state, actions, exo)
    
    # We look at the actual electrical draw and the resulting room temp
    hp_draw = info['hp_electrical_w'][0] 
    room_temp = state.thermal.T_vector[1] # Usually index 1 is the first room
    
    print(f"Step {i+1} | Room Temp: {room_temp:.4f}°C | HP Draw: {hp_draw}W")