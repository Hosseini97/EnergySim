import jax.numpy as jnp
import equinox as eqx
from dataclasses import field
from typing import Literal, Tuple

# Define array type for clarity
Array = jnp.ndarray

# ==========================================
# 1. Configuration Modules (Static/Physics)
# ==========================================

class ThermalConfig(eqx.Module):
    """
    Configuration for a full RC-Network thermal model.
    """
    # --- 1. The Matrices (Dynamic Leaves) ---
    # In Equinox, arrays should generally NOT be static if you want 
    # to differentiate through them or use them in calculations.
    A_matrix: Array
    C_inv_vector: Array
    B_matrix: Array

    # --- 2. Node Indices (Static) ---
    # Integers used for indexing must be static to trace correctly in JIT.
    ambient_air_index: int = eqx.field(static=True)
    room_air_indices: Tuple[int, ...] = eqx.field(static=True)
    wall_indices: Tuple[int, ...] = eqx.field(static=True)
    mass_indices: Tuple[int, ...] = eqx.field(static=True)

    # --- 3. Coupling Indices (Static) ---
    waste_heat_node_index: int = eqx.field(static=True, default=-1)

    # --- 4. Infiltration Parameters (Static/Hyperparams) ---
    use_dynamic_infiltration: bool = eqx.field(static=True, default=False)
    inf_k1: float = eqx.field(static=True, default=0.1)
    inf_k2: float = eqx.field(static=True, default=0.0)
    inf_k3: float = eqx.field(static=True, default=0.0)
    room_vol_m3: float = eqx.field(static=True, default=0.0)

    # --- 5. Cost/Control Parameters (Static) ---
    setpoint: float = eqx.field(static=True, default=21.0)
    comfort_band: float = eqx.field(static=True, default=1.0)
    model_type: str = eqx.field(static=True, default="RCNetwork")


class BatteryConfig(eqx.Module):
    model_type: Literal["simple", "degradation"] = eqx.field(static=True, default="simple")
    capacity_kwh: float = eqx.field(static=True, default=10.0)
    max_power_kw: float = eqx.field(static=True, default=5.0)
    efficiency: float = eqx.field(static=True, default=0.90)
    degradation_rate_per_cycle: float = eqx.field(static=True, default=0.0001)

    @property
    def capacity_j(self) -> float:
        return self.capacity_kwh * 3.6e6

    @property
    def max_power_w(self) -> float:
        return self.max_power_kw * 1000.0


class RewardConfig(eqx.Module):
    price_weight: float = eqx.field(static=True, default=1.0)
    comfort_weight: float = eqx.field(static=True, default=5.0)


class AirConditionerConfig(eqx.Module):
    model_type: Literal["stateless", "ramping", "variable_cop"] = eqx.field(static=True, default="stateless")
    max_electrical_power_w: float = eqx.field(static=True, default=5000.0)
    min_electrical_power_w: float = eqx.field(static=True, default=500.0)
    tau_thermal_seconds: float = eqx.field(static=True, default=60.0)
    
    cop_cooling: float = eqx.field(static=True, default=3.0)
    ramp_rate_w_per_sec: float = eqx.field(static=True, default=1000.0)
    
    # Note: Converted Tuples to Arrays for easier JAX interpolation if needed. 
    # If these are strictly lookup keys, they can remain static tuples, 
    # but arrays are more flexible in JAX.
    cop_ambient_temps_c: Array = eqx.field(default_factory=lambda: jnp.array([20.0, 25.0, 30.0, 35.0, 40.0]))
    cop_values_cooling: Array = eqx.field(default_factory=lambda: jnp.array([4.5, 4.0, 3.5, 3.0, 2.5]))


class HeatPumpConfig(eqx.Module):
    model_type: Literal["stateless", "ramping", "variable_cop"] = eqx.field(static=True, default="stateless")
    max_electrical_power_w: float = eqx.field(static=True, default=5000.0)
    min_electrical_power_w: float = eqx.field(static=True, default=500.0)
    tau_thermal_seconds: float = eqx.field(static=True, default=60.0)
    
    cop_heating: float = eqx.field(static=True, default=3.5)
    ramp_rate_w_per_sec: float = eqx.field(static=True, default=1000.0)
    
    cop_ambient_temps_c: Array = eqx.field(default_factory=lambda: jnp.array([-10.0, 0.0, 10.0, 20.0]))
    cop_values_heating: Array = eqx.field(default_factory=lambda: jnp.array([2.5, 3.0, 3.5, 4.0]))


class ThermalStorageConfig(eqx.Module):
    n_nodes: int = eqx.field(static=True, default=5)
    volume_m3: float = eqx.field(static=True, default=0.3)
    height_m: float = eqx.field(static=True, default=1.5)
    
    max_charge_kw: float = eqx.field(static=True, default=15.0)
    max_discharge_kw: float = eqx.field(static=True, default=15.0)
    
    loss_coeff_w_k: float = eqx.field(static=True, default=2.0)
    ambient_temp_c: float = eqx.field(static=True, default=15.0)
    vertical_conductivity_w_mk: float = eqx.field(static=True, default=0.6)

    @property
    def max_charge_w(self) -> float:
        return self.max_charge_kw * 1000.0
    @property
    def max_discharge_w(self) -> float:
        return self.max_discharge_kw * 1000.0


class GridThermalStorageConfig(eqx.Module):
    grid_shape: Tuple[int, int, int] = eqx.field(static=True, default=(10, 5, 5)) 
    total_volume_m3: float = eqx.field(static=True, default=0.3)
    height_m: float = eqx.field(static=True, default=1.5)
    
    thermal_conductivity_w_mk: float = eqx.field(static=True, default=0.65)
    convection_conductivity_w_mk: float = eqx.field(static=True, default=50.0)
    loss_coeff_to_ambient_w_m2k: float = eqx.field(static=True, default=0.5)
    ambient_temp_c: float = eqx.field(static=True, default=15.0)

    charge_inlet_idx: Tuple[int, int, int] = eqx.field(static=True, default=(0, 2, 2))
    discharge_outlet_idx: Tuple[int, int, int] = eqx.field(static=True, default=(0, 0, 0))
    
    max_charge_kw: float = eqx.field(static=True, default=15.0)
    max_discharge_kw: float = eqx.field(static=True, default=15.0)

    @property
    def voxel_volume_m3(self) -> float:
        z, y, x = self.grid_shape
        return self.total_volume_m3 / (z * y * x)
    
    @property
    def max_charge_w(self) -> float:
        return self.max_charge_kw * 1000.0
    @property
    def max_discharge_w(self) -> float:
        return self.max_discharge_kw * 1000.0


class PVConfig(eqx.Module):
    model_type: Literal["simple", "passthrough", "geometric"] = eqx.field(static=True, default="simple")
    
    panel_area_m2: float = eqx.field(static=True, default=20.0)
    efficiency: float = eqx.field(static=True, default=0.20)
    temp_coefficient: float = eqx.field(static=True, default=-0.004)
    reference_temp_c: float = eqx.field(static=True, default=25.0)
    
    latitude_deg: float = eqx.field(static=True, default=48.13)
    longitude_deg: float = eqx.field(static=True, default=11.58)
    panel_azimuth_deg: float = eqx.field(static=True, default=180.0)
    panel_tilt_deg: float = eqx.field(static=True, default=30.0)

# ==========================================
# 2. Dynamic State Modules
# ==========================================
# We replace @flax_dataclass with eqx.Module.
# These are now Pytrees by definition.

class ThermalState(eqx.Module):
    T_vector: Array

class BatteryState(eqx.Module):
    soc: Array 
    soh: Array

class ThermalStorageState(eqx.Module):
    temperatures_c: Array # (Z, Y, X)
    
    @property
    def soc(self) -> Array:
        # Note: Operations inside properties work, but for JIT-compiled
        # functions, prefer passing the resulting Array, not the property access
        # if it involves heavy logic. Here it is fine.
        avg = jnp.mean(self.temperatures_c)
        return jnp.clip((avg - 30.0) / (60.0 - 30.0), 0.0, 1.0)

class HeatPumpState(eqx.Module):
    current_electrical_w: Array
    current_thermal_w: Array

class AirConditionerState(eqx.Module):
    current_electrical_w: Array
    current_thermal_w: Array

class PVState(eqx.Module):
    current_power_w: Array

class SystemState(eqx.Module):
    thermal: ThermalState
    battery: BatteryState
    storage: ThermalStorageState
    heat_pump: HeatPumpState
    air_conditioner: AirConditionerState

class ExogenousData(eqx.Module):
    # Weather
    ambient_temp: Array
    solar_irradiance_w_m2: Array
    wind_speed_m_s: Array
    time_of_year_seconds: Array 

    # Price
    price: Array

    # Loads (W)
    base_load_w: Array
    ev_charger_load_w: Array
    dishwasher_load_w: Array
    clothes_dryer_load_w: Array
    water_heater_load_w: Array
    cooking_load_w: Array

    # Thermal Gains (W)
    occupancy_gains_w: Array
    solar_gains_w: Array
    device_gains_w: Array

class SystemActions(eqx.Module):
    battery_power_w: Array      
    heat_pump_power_w: Array    
    ac_power_w: Array           
    storage_discharge_w: Array  

class HeatPumpOutput(eqx.Module):
    thermal_power_w: Array     
    electrical_power_w: Array   

class AirConditionerOutput(eqx.Module):
    thermal_power_w: Array     
    electrical_power_w: Array   

class ThermalStorageOutput(eqx.Module):
    actual_discharge_w: Array    
    rejected_heat_w: Array
    standing_loss_w: Array

class PVOutput(eqx.Module):
    pv_generation_w: Array

class SystemOutputs(eqx.Module):
    """
    Captures all instantaneous outputs generated during a single simulation step.
    This is passed to the external cost function / reward calculator.
    """
    pv: PVOutput
    hp: HeatPumpOutput
    ac: AirConditionerOutput
    storage: ThermalStorageOutput
    total_waste_heat_w: Array