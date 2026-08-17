"""Free-floating stability of the RC thermal network.

With every HVAC actuator commanded to zero the house is driven only by weather
and by strictly non-negative solar/internal gains. It must therefore settle a
bounded distance above ambient. The regression these tests guard is an envelope
whose conductance to ambient is orders of magnitude too small: the network stays
formally stable but its global time constant runs to years, so room temperature
ramps monotonically for the whole of any simulation and never equilibrates.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from build_my_house import create_2_room_house  # noqa: E402
from energysim.core.data.dataset import SimulationDataset  # noqa: E402
from energysim.core.network_builder import RCNetworkBuilder  # noqa: E402
from energysim.core.shared.data_structs import RewardConfig, SystemActions  # noqa: E402
from energysim.rl.vector_env import VectorizedEnergyEnv  # noqa: E402
from energysim.sim.simulator import JAXSimulator  # noqa: E402

DT_SECONDS = 900.0
STEPS_PER_DAY = int(24 * 3600 / DT_SECONDS)
SIM_DAYS = 30

AMBIENT_MEAN_C = 10.0
AMBIENT_SWING_C = 10.0
SOLAR_PEAK_W = 1000.0
INTERNAL_PEAK_W = 150.0


def _weather_frame(n_days=7, constant=False):
    """Reproduces examples/sample_data_generator.py without touching the disk.

    Both gain channels are half-wave rectified sinusoids, so they have a
    strictly positive mean -- that is what the envelope has to shed.
    """
    n_steps = n_days * STEPS_PER_DAY
    t = np.linspace(0, n_days * 2 * np.pi, n_steps)
    if constant:
        ambient = np.full(n_steps, AMBIENT_MEAN_C)
        solar = np.full(n_steps, SOLAR_PEAK_W / np.pi)      # same mean as the sine
        internal = np.full(n_steps, INTERNAL_PEAK_W / np.pi)
        wind = np.full(n_steps, 2.0 / np.pi)
    else:
        ambient = AMBIENT_MEAN_C + AMBIENT_SWING_C * np.sin(t)
        solar = np.maximum(0.0, SOLAR_PEAK_W * np.sin(t))
        internal = np.maximum(0.0, INTERNAL_PEAK_W * np.sin(t + np.pi * 1.5))
        wind = np.abs(2.0 * np.sin(t))

    return pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=n_steps, freq="15min"),
        "ambient_temp": ambient,
        "solar_irradiance_w_m2": np.zeros(n_steps),
        "price": np.zeros(n_steps),
        "load": np.zeros(n_steps),
        "internal_gains_w": internal,
        "solar_gains_w": solar,
        "wind_speed_m_s": wind,
    })


def _run_free_floating(df, days=SIM_DAYS):
    """Run the 2-room house with all HVAC actions pinned to zero.

    Returns (room_temps[T, n_rooms], ambient[T]).
    """
    dataset = SimulationDataset("<in-memory>", dt_seconds=int(DT_SECONDS), read_fn=lambda _: df)
    simulator = JAXSimulator(
        dt_seconds=DT_SECONDS,
        t_config=create_2_room_house(),
        r_config=RewardConfig(),
    )
    env = VectorizedEnergyEnv(simulator, dataset, num_envs=1)
    state = env.reset(jax.random.PRNGKey(0))

    n_rooms = len(simulator.thermal.config.room_air_indices)
    zero_rooms = jnp.zeros((1, n_rooms))
    idle = SystemActions(
        battery_power_w=jnp.zeros((1,)),
        heat_pump_power_w=zero_rooms,
        ac_power_w=zero_rooms,
        storage_discharge_w=zero_rooms,
    )

    temps, ambient = [], []
    for _ in range(days * STEPS_PER_DAY):
        ambient.append(env.shared_exo_data.ambient_temp[state.time_idx[0] % env.n_steps])
        state, _, _, info = env.step(state, idle)
        temps.append(info["room_temps"][0])

    return np.asarray(jnp.stack(temps)), np.asarray(jnp.stack(ambient))


@pytest.fixture(scope="module")
def free_float():
    return _run_free_floating(_weather_frame())


def test_stays_within_plausible_band_of_ambient(free_float):
    """Indoor temperature never leaves a band a real unconditioned house occupies."""
    temps, ambient = free_float

    # Passive solar can lift a house well above ambient, but not beyond this,
    # and with no cooling source it cannot fall below the coldest ambient.
    lower = ambient.min() - 2.0
    upper = ambient.max() + 20.0

    assert temps.min() >= lower, (
        f"indoor fell to {temps.min():.1f} C, below ambient minimum {ambient.min():.1f} C"
    )
    assert temps.max() <= upper, (
        f"indoor reached {temps.max():.1f} C after {SIM_DAYS} days with zero HVAC; "
        f"ambient never exceeded {ambient.max():.1f} C"
    )


def test_does_not_ramp_monotonically(free_float):
    """The daily mean must stop climbing -- the actual regression.

    Under the broken envelope (UA = 1.5 W/K) the daily mean rose past 31 C by
    day 7 and 45 C by day 20, gaining ~1 K/day and still accelerating away.
    """
    temps, _ = free_float
    daily_mean = temps.reshape(SIM_DAYS, STEPS_PER_DAY, -1).mean(axis=(1, 2))

    # Second half of the run: transient is gone, only drift is left.
    drift_per_day = (daily_mean[-1] - daily_mean[SIM_DAYS // 2]) / (SIM_DAYS - SIM_DAYS // 2)
    assert abs(drift_per_day) < 0.1, (
        f"daily mean still moving at {drift_per_day:+.3f} K/day between day "
        f"{SIM_DAYS // 2} ({daily_mean[SIM_DAYS // 2]:.2f} C) and day {SIM_DAYS} "
        f"({daily_mean[-1]:.2f} C); the house is not reaching equilibrium"
    )


def test_settles_under_constant_weather():
    """With time-invariant forcing the network must reach a true fixed point."""
    temps, ambient = _run_free_floating(_weather_frame(constant=True))
    daily_mean = temps.reshape(SIM_DAYS, STEPS_PER_DAY, -1).mean(axis=(1, 2))

    assert abs(daily_mean[-1] - daily_mean[-2]) < 0.01, (
        f"no fixed point after {SIM_DAYS} days: day {SIM_DAYS - 1} "
        f"{daily_mean[-2]:.3f} C -> day {SIM_DAYS} {daily_mean[-1]:.3f} C"
    )
    # Steady-state lift above ambient is set by (total gains) / (total UA).
    lift = daily_mean[-1] - ambient.mean()
    assert 0.0 < lift < 15.0, f"steady-state lift above ambient is {lift:.1f} K"


def test_builder_warns_when_envelope_cannot_shed_gains():
    """The original 2-room house passed area-normalised R-values straight to add_resistor."""
    builder = RCNetworkBuilder(n_rooms=1)
    builder.add_node("room_air_0", capacity_j_k=5.0e6)
    builder.add_node("wall_0", capacity_j_k=1.0e8)
    builder.add_resistor("room_air_0", "wall_0", R_k_w=1.0)
    builder.add_resistor("wall_0", "ambient", R_k_w=2.0)     # 0.5 W/K for a whole wall
    builder.add_resistor("room_air_0", "ambient", R_k_w=4.0)
    builder.add_input_mapping("solar_gains_w", "wall_0", room_index=0)

    with pytest.warns(UserWarning, match="time constant"):
        builder.compile()


def test_builder_rejects_node_with_no_path_to_ambient():
    """A node isolated from ambient makes any gain on it accumulate forever."""
    builder = RCNetworkBuilder(n_rooms=1)
    builder.add_node("room_air_0", capacity_j_k=5.0e5)
    builder.add_node("wall_0", capacity_j_k=2.0e7)
    builder.add_resistor("room_air_0", "wall_0", R_k_w=0.03)  # nothing reaches ambient
    builder.add_input_mapping("solar_gains_w", "room_air_0", room_index=0)

    with pytest.raises(ValueError, match="no conductance path to 'ambient'"):
        builder.compile()
