
import jax
from energysim.core.shared.data_structs import SystemState
from energysim.core.data.dataset import ExogenousData
import jax.numpy as jnp

def extract_obs(state: SystemState, exo: ExogenousData, room_indices: jax.Array) -> jax.Array:
    """Flattens and NORMALIZES the observation vector."""
    
    # Internal State Normalization
    room_temps = state.thermal.T_vector[room_indices]
    norm_temps = (room_temps - 21.0) / 10.0           # Centered at setpoint, scaled 0.1/deg
    norm_soc = (state.battery.soc - 0.5) * 2.0        # [0, 1] -> [-1, 1]
    
    avg_tank = jnp.mean(state.storage.temperatures_c)
    norm_tank = (avg_tank - 45.0) / 30.0             # Centered at nominal tank temp
    
    # Exogenous Data Normalization
    norm_amb = (exo.ambient_temp - 15.0) / 20.0      # Approx seasonal range
    norm_solar = exo.solar_irradiance_w_m2 / 1000.0   # Scale max sun to ~1.0
    norm_price = (exo.price - 0.25) / 0.25           # Centered at typical energy price
    
    obs = jnp.concatenate([
        norm_temps, 
        jnp.atleast_1d(norm_soc), 
        jnp.atleast_1d(norm_tank),
        jnp.atleast_1d(norm_amb), 
        jnp.atleast_1d(norm_solar), 
        jnp.atleast_1d(norm_price)
    ])
    return obs