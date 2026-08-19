
import jax
from energysim.core.shared.data_structs import SystemState
from energysim.core.data.dataset import ExogenousData
import jax.numpy as jnp

# Price normalization defaults.
#
# These were originally 0.25 / 0.25, i.e. centred on a German *retail* tariff. The
# datasets actually used for training are *wholesale* day-ahead prices, which have a
# mean around 0.037 and a standard deviation around 0.018 EUR/kWh. Normalizing those
# against a retail-scale constant shrank every price feature to roughly a twentieth of
# the amplitude of the other observations, so the policy networks were effectively
# price-blind and learned to track PV instead (see hems_project/scripts/7_diagnose_obs.py).
#
# Callers that know their dataset should pass the *training split's* statistics
# explicitly rather than relying on these defaults, and must reuse the training
# statistics when evaluating so no test-split information leaks into normalization.
DEFAULT_PRICE_CENTER = 0.037
DEFAULT_PRICE_SCALE = 0.018


def extract_obs(
    state: SystemState,
    exo: ExogenousData,
    room_indices: jax.Array,
    *,
    price_center: float = DEFAULT_PRICE_CENTER,
    price_scale: float = DEFAULT_PRICE_SCALE,
) -> jax.Array:
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
    norm_price = (exo.price - price_center) / price_scale

    obs = jnp.concatenate([
        norm_temps,
        jnp.atleast_1d(norm_soc),
        jnp.atleast_1d(norm_tank),
        jnp.atleast_1d(norm_amb),
        jnp.atleast_1d(norm_solar),
        jnp.atleast_1d(norm_price)
    ])
    return obs