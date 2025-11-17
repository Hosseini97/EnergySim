# energysim/core/shared/data_structs.py
from dataclasses import dataclass, field
# --- MODIFIED: Added Tuple ---
from typing import Literal, List, Tuple
import jax.numpy as jnp
from flax.struct import dataclass as flax_dataclass

# Define array type for clarity
Array = jnp.ndarray

# --- Configuration Structs ---
# These are static and passed as parameters

@dataclass(frozen=True)
class ThermalConfig:
    """Parameters for the thermal model for N rooms."""
    # This remains a scalar
    model_type: Literal["1R1C", "2R2C", "3R2C", "4R3C", "passthrough"] = "1R1C"
    
    # Shared setpoints can be scalars
    setpoint: float = 21.0
    comfort_band: float = 1.0
    
    # --- Room-specific parameters become VECTORS ---
    # We can't use @property anymore, so we pre-calculate capacities.
    air_capacity_j_k: Array       # J/K, shape (N_rooms,)
    wall_capacity_j_k: Array      # J/K, shape (N_rooms,)
    mass_capacity_j_k: Array      # J/K, shape (N_rooms,)
    
    # Resistances/Coefficients also become vectors
    air_to_ambient_coeff_w_k: Array # W/K, shape (N_rooms,)
    wall_to_ambient_r_k_w: Array  # K/W, shape (N_rooms,)
    air_to_wall_r_k_w: Array      # K/W, shape (N_rooms,)
    air_to_ambient_r_k_w: Array   # K/W, shape (N_rooms,)
    air_to_mass_r_k_w: Array      # K/W, shape (N_rooms,)
    
    # Fractions can also be room-specific
    solar_gain_to_wall_frac: Array # shape (N_rooms,)
    solar_gain_to_mass_frac: Array # shape (N_rooms,)

    @property
    def air_capacity_j_k(self) -> float:
        """Thermal capacity of the internal air mass."""
        return self.air_volume_m3 * self.air_density_kg_m3 * self.air_specific_heat_j_kgk

@dataclass(frozen=True)
class BatteryConfig:
    """Parameters for the battery model."""
    model_type: Literal["simple", "degradation"] = "simple"
    capacity_kwh: float = 10.0
    max_power_kw: float = 5.0
    efficiency: float = 0.90 # Round-trip
    degradation_rate_per_cycle: float = 0.0001 # Fractional SOH loss per full cycle
    @property
    def capacity_j(self) -> float:
        return self.capacity_kwh * 3.6e6
    @property
    def max_power_w(self) -> float:
        return self.max_power_kw * 1000.0

@dataclass(frozen=True)
class RewardConfig:
    """Weights for the cost/reward function."""
    price_weight: float = 1.0
    comfort_weight: float = 5.0 # Penalize discomfort highly

@dataclass(frozen=True)
class AirConditionerConfig:
    model_type: Literal["stateless", "ramping", "variable_cop"] = "stateless"
    max_electrical_power_w: float = 5000.0 # 5kW max electricity draw
    cop_cooling: float = 3.0 # Fixed COP (for stateless/ramping)
    ramp_rate_w_per_sec: float = 1000.0 # W of *electrical* power change per sec
    cop_ambient_temps_c: Tuple[float, ...] = field(default_factory=lambda: (20.0, 25.0, 30.0, 35.0, 40.0))
    cop_values_cooling: Tuple[float, ...] = field(default_factory=lambda: (4.5, 4.0, 3.5, 3.0, 2.5))

@dataclass(frozen=True)
class HeatPumpConfig:
    model_type: Literal["stateless", "ramping", "variable_cop"] = "stateless"
    max_electrical_power_w: float = 5000.0 # 5kW max electricity draw
    cop_heating: float = 3.5             # Fixed COP (for stateless/ramping)
    ramp_rate_w_per_sec: float = 1000.0 # W of *electrical* power change per sec
    cop_ambient_temps_c: Tuple[float, ...] = field(default_factory=lambda: (-10.0, 0.0, 10.0, 20.0))
    cop_values_heating: Tuple[float, ...] = field(default_factory=lambda: (2.5, 3.0, 3.5, 4.0))

@dataclass(frozen=True)
class ThermalStorageConfig:
    capacity_kwh: float = 50.0
    max_charge_kw: float = 15.0
    max_discharge_kw: float = 15.0
    standing_loss_rate: float = 0.01
    @property
    def capacity_j(self) -> float:
        return self.capacity_kwh * 3.6e6
    @property
    def max_charge_w(self) -> float:
        return self.max_charge_kw * 1000.0
    @property
    def max_discharge_w(self) -> float:
        return self.max_discharge_kw * 1000.0
    @property
    def standing_loss_w_per_soc(self) -> float:
        return (self.capacity_kwh * 1000.0 * self.standing_loss_rate)

@dataclass(frozen=True)
class SolarConfig:
    model_type: Literal["simple", "passthrough"] = "simple"
    panel_area_m2: float = 20.0
    efficiency: float = 0.20
    temp_coefficient: float = -0.004
    reference_temp_c: float = 25.0

# --- Dynamic State Structs ---

@flax_dataclass
class ThermalState:
    """State of the thermal model."""
    room_temp: Array  # °C, shape (N_rooms,)
    wall_temp: Array  # °C, shape (N_rooms,)
    mass_temp: Array  # °C, shape (N_rooms,)

@flax_dataclass
class BatteryState:
    """State of the battery model."""
    soc: Array  # [0.0 - 1.0]
    soh: Array  # [0.0 - 1.0], only for degradation model

@flax_dataclass
class ThermalStorageState:
    """Data-only view of thermal storage state."""
    soc: Array  # [0.0 - 1.0]

@flax_dataclass
class HeatPumpState:
    """Data-only view of heat pump state."""
    current_electrical_w: Array # W

@flax_dataclass
class AirConditionerState:
    """Data-only view of AC state."""
    current_electrical_w: Array # W

@flax_dataclass
class SystemState:
    """The complete internal state of the simulation."""
    thermal: ThermalState
    battery: BatteryState
    storage: ThermalStorageState
    heat_pump: HeatPumpState
    air_conditioner: AirConditionerState

@flax_dataclass  # <--- HEAVILY UPDATED
class ExogenousData:
    """All external data for a single timestep."""
    # --- Weather ---
    ambient_temp: Array        # °C
    solar_irradiance_w_m2: Array # W/m^2 (replaces 'pv')

    # --- Price ---
    price: Array               # €/kWh

    # --- Loads (W) ---
    base_load_w: Array       # Non-controllable, non-device load (replaces 'load')
    ev_charger_load_w: Array     # from behavioral model
    dishwasher_load_w: Array     # from behavioral model
    clothes_dryer_load_w: Array # from behavioral model
    water_heater_load_w: Array   # from behavioral model
    cooking_load_w: Array        # from behavioral model

    # --- Thermal Gains (W) ---
    # Room-specific thermal gains become vectors
    occupancy_gains_w: Array     # Heat from people, shape (N_rooms,)
    solar_gains_w: Array         # Heat from windows, shape (N_rooms,)
    device_gains_w: Array        # Heat from all electrical devices, shape (N_rooms,)

@flax_dataclass
class SystemActions:
    """All control actions for a single timestep."""
    # Battery is a shared resource, remains scalar
    battery_power_w: Array     # W, shape ()
    
    # HVAC actions are now zonal (per-room)
    heat_pump_power_w: Array     # W (Electrical), shape (N_rooms,)
    ac_power_w: Array            # W (Electrical), shape (N_rooms,)
    storage_discharge_w: Array   # W (Thermal), shape (N_rooms,)

@flax_dataclass
class HeatPumpOutput:
    """The calculated outputs of the Heat Pump model for one step."""
    thermal_power_w: Array     # The *actual* (clipped) thermal power *generated*
    electrical_power_w: Array    # The electrical power *consumed*

@flax_dataclass
class AirConditionerOutput:
    """The calculated outputs of the AC model for one step."""
    thermal_power_w: Array     # The *actual* (clipped) thermal power *removed* (will be negative)
    electrical_power_w: Array    # The electrical power *consumed*

@flax_dataclass
class ThermalStorageOutput:
    """Calculated outputs from the thermal storage step."""
    actual_discharge_w: Array    # Actual thermal power to room
    rejected_heat_w: Array       # Wasted heat (charged when full)

@flax_dataclass # <--- NEW
class SolarOutput:
    """Calculated outputs from the solar model step."""
    pv_generation_w: Array # Actual PV power generated (W)