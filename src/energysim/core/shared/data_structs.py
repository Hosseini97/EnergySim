# energysim/core/shared/data_structs.py
from dataclasses import dataclass, field
# --- MODIFIED: Added Tuple ---
from typing import Literal, List, Tuple
import jax.numpy as jnp
from flax.struct import dataclass as flax_dataclass
import equinox as eqx

# Define array type for clarity
Array = jnp.ndarray

class ThermalConfig(eqx.Module):  # <--- CHANGE
    """
    Configuration for a full RC-Network thermal model.
    Holds the pre-built matrices for the simulation.
    """
    # --- 1. The Matrices (Dynamic) ---
    A_matrix: Array
    C_inv_vector: Array
    B_matrix: Array

    # --- 2. Node Indices (Static) ---
    ambient_air_index: int = eqx.field(static=True)
    room_air_indices: Tuple[int, ...] = eqx.field(static=True)
    wall_indices: Tuple[int, ...] = eqx.field(static=True)
    mass_indices: Tuple[int, ...] = eqx.field(static=True)

    # --- 3. Cost/Control Parameters (Static) ---
    setpoint: float = eqx.field(static=True, default=21.0)
    comfort_band: float = eqx.field(static=True, default=1.0)
    
    # --- 4. Model Type (Static) ---
    # This field is added by network_builder.py
    model_type: str = eqx.field(static=True, default="RCNetwork")


class BatteryConfig(eqx.Module):  # <--- CHANGE
    """Parameters for the battery model."""
    model_type: Literal["simple", "degradation"] = eqx.field(static=True, default="simple")
    capacity_kwh: float = eqx.field(static=True, default=10.0)
    max_power_kw: float = eqx.field(static=True, default=5.0)
    efficiency: float = eqx.field(static=True, default=0.90) # Round-trip
    degradation_rate_per_cycle: float = eqx.field(static=True, default=0.0001)
    
    @property
    def capacity_j(self) -> float:
        return self.capacity_kwh * 3.6e6
    @property
    def max_power_w(self) -> float:
        return self.max_power_kw * 1000.0

class RewardConfig(eqx.Module):  # <--- CHANGE
    """Weights for the cost/reward function."""
    price_weight: float = eqx.field(static=True, default=1.0)
    comfort_weight: float = eqx.field(static=True, default=5.0)

class AirConditionerConfig(eqx.Module):  # <--- CHANGE
    model_type: Literal["stateless", "ramping", "variable_cop"] = eqx.field(static=True, default="stateless")
    max_electrical_power_w: float = eqx.field(static=True, default=5000.0)
    cop_cooling: float = eqx.field(static=True, default=3.0)
    ramp_rate_w_per_sec: float = eqx.field(static=True, default=1000.0)
    cop_ambient_temps_c: Tuple[float, ...] = eqx.field(static=True, default_factory=lambda: (20.0, 25.0, 30.0, 35.0, 40.0))
    cop_values_cooling: Tuple[float, ...] = eqx.field(static=True, default_factory=lambda: (4.5, 4.0, 3.5, 3.0, 2.5))

class HeatPumpConfig(eqx.Module):  # <--- CHANGE
    model_type: Literal["stateless", "ramping", "variable_cop"] = eqx.field(static=True, default="stateless")
    max_electrical_power_w: float = eqx.field(static=True, default=5000.0)
    cop_heating: float = eqx.field(static=True, default=3.5)
    ramp_rate_w_per_sec: float = eqx.field(static=True, default=1000.0)
    cop_ambient_temps_c: Tuple[float, ...] = eqx.field(static=True, default_factory=lambda: (-10.0, 0.0, 10.0, 20.0))
    cop_values_heating: Tuple[float, ...] = eqx.field(static=True, default_factory=lambda: (2.5, 3.0, 3.5, 4.0))

class ThermalStorageConfig(eqx.Module):  # <--- CHANGE
    capacity_kwh: float = eqx.field(static=True, default=50.0)
    max_charge_kw: float = eqx.field(static=True, default=15.0)
    max_discharge_kw: float = eqx.field(static=True, default=15.0)
    standing_loss_rate: float = eqx.field(static=True, default=0.01)

    # ... (property methods remain unchanged) ...
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

class SolarConfig(eqx.Module):  # <--- CHANGE
    model_type: Literal["simple", "passthrough"] = eqx.field(static=True, default="simple")
    panel_area_m2: float = eqx.field(static=True, default=20.0)
    efficiency: float = eqx.field(static=True, default=0.20)
    temp_coefficient: float = eqx.field(static=True, default=-0.004)
    reference_temp_c: float = eqx.field(static=True, default=25.0)

# --- Dynamic State Structs ---

@flax_dataclass
class ThermalState:
    """
    Data-only state for the RC-Network thermal model.
    It is a single, flat vector of all node temperatures.
    """
    # T_vector shape (N_nodes,)
    T_vector: Array

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