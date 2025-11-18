from dataclasses import dataclass, field
from typing import Literal, List, Tuple
import jax.numpy as jnp
from flax.struct import dataclass as flax_dataclass
import equinox as eqx

# Define array type for clarity
Array = jnp.ndarray

class ThermalConfig(eqx.Module):
    """
    Configuration for a full RC-Network thermal model.
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

    # --- 3. Coupling Indices (Static) ---
    # Which node receives the waste heat from Storage/HVAC? (e.g., Utility Room)
    # If -1, waste heat is lost to ambient/void.
    waste_heat_node_index: int = eqx.field(static=True, default=-1)

    # --- 4. Infiltration Parameters (Static) ---
    # If True, calculates dynamic infiltration based on wind/temp
    use_dynamic_infiltration: bool = eqx.field(static=True, default=False)
    # Air Change Per Hour (ACH) coefficients: ACH = K1 + K2*|dT| + K3*Wind
    inf_k1: float = eqx.field(static=True, default=0.1) # Base leakage
    inf_k2: float = eqx.field(static=True, default=0.0) # Temperature driven (stack effect)
    inf_k3: float = eqx.field(static=True, default=0.0) # Wind driven
    room_vol_m3: float = eqx.field(static=True, default=0.0) # Total air volume for infiltration calc

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
    
    # --- NEW: Inverter Constraints ---
    min_electrical_power_w: float = eqx.field(static=True, default=500.0) # Unit shuts off below this
    
    # --- NEW: Thermal Lag ---
    tau_thermal_seconds: float = eqx.field(static=True, default=60.0) # Time constant for cooling delivery
    
    cop_cooling: float = eqx.field(static=True, default=3.0)
    ramp_rate_w_per_sec: float = eqx.field(static=True, default=1000.0)
    cop_ambient_temps_c: Tuple[float, ...] = eqx.field(static=True, default_factory=lambda: (20.0, 25.0, 30.0, 35.0, 40.0))
    cop_values_cooling: Tuple[float, ...] = eqx.field(static=True, default_factory=lambda: (4.5, 4.0, 3.5, 3.0, 2.5))

class HeatPumpConfig(eqx.Module):
    model_type: Literal["stateless", "ramping", "variable_cop"] = eqx.field(static=True, default="stateless")
    max_electrical_power_w: float = eqx.field(static=True, default=5000.0)
    
    # --- NEW: Inverter Constraints ---
    min_electrical_power_w: float = eqx.field(static=True, default=500.0)
    
    # --- NEW: Thermal Lag ---
    tau_thermal_seconds: float = eqx.field(static=True, default=60.0)
    
    cop_heating: float = eqx.field(static=True, default=3.5)
    ramp_rate_w_per_sec: float = eqx.field(static=True, default=1000.0)
    cop_ambient_temps_c: Tuple[float, ...] = eqx.field(static=True, default_factory=lambda: (-10.0, 0.0, 10.0, 20.0))
    cop_values_heating: Tuple[float, ...] = eqx.field(static=True, default_factory=lambda: (2.5, 3.0, 3.5, 4.0))

class ThermalStorageConfig(eqx.Module):
    # --- UPDATED: Stratification Parameters ---
    n_nodes: int = eqx.field(static=True, default=5) # Number of vertical layers
    volume_m3: float = eqx.field(static=True, default=0.3) # ~300 Liters
    height_m: float = eqx.field(static=True, default=1.5) 
    
    # Derived capacity depends on Delta T, generally we define capacity via volume now
    # but for backward compatibility/cost calc, we might keep a nominal capacity reference.
    
    max_charge_kw: float = eqx.field(static=True, default=15.0)
    max_discharge_kw: float = eqx.field(static=True, default=15.0)
    
    # Heat Loss (U-value * Surface Area approximation per node)
    loss_coeff_w_k: float = eqx.field(static=True, default=2.0) # W/K loss to ambient per node
    ambient_temp_c: float = eqx.field(static=True, default=15.0) # Utility room temp
    
    # Vertical Conductivity (Simulation of mixing/conduction between water layers)
    vertical_conductivity_w_mk: float = eqx.field(static=True, default=0.6) # Water conductivity + mixing factor

    @property
    def max_charge_w(self) -> float:
        return self.max_charge_kw * 1000.0
    @property
    def max_discharge_w(self) -> float:
        return self.max_discharge_kw * 1000.0
    
class GridThermalStorageConfig(eqx.Module):
    # --- Geometry ---
    # Shape: (Z_layers, Y_rows, X_cols)
    # For 2D, set X_cols=1. For 1D, set Y=1, X=1.
    grid_shape: Tuple[int, int, int] = eqx.field(static=True, default=(10, 5, 5)) 
    
    total_volume_m3: float = eqx.field(static=True, default=0.3)
    height_m: float = eqx.field(static=True, default=1.5)
    
    # --- Physics ---
    # Thermal conductivity of water (approx 0.6) + enhancement factor for turbulence
    thermal_conductivity_w_mk: float = eqx.field(static=True, default=0.65)
    
    # Buoyancy: Effective conductivity when T_bottom > T_top (unstable)
    # We simulate convection by drastically increasing conductivity in unstable regions
    convection_conductivity_w_mk: float = eqx.field(static=True, default=50.0)
    
    loss_coeff_to_ambient_w_m2k: float = eqx.field(static=True, default=0.5) # Insulation quality
    ambient_temp_c: float = eqx.field(static=True, default=15.0)

    # --- Port Mapping (Where inputs/outputs physically connect) ---
    # Coordinates are (z, y, x). 0 is Top.
    charge_inlet_idx: Tuple[int, int, int] = eqx.field(static=True, default=(0, 2, 2)) # Top Center
    discharge_outlet_idx: Tuple[int, int, int] = eqx.field(static=True, default=(0, 0, 0)) # Top Corner
    
    max_charge_kw: float = eqx.field(static=True, default=15.0)
    max_discharge_kw: float = eqx.field(static=True, default=15.0)

    @property
    def voxel_volume_m3(self) -> float:
        z, y, x = self.grid_shape
        return self.total_volume_m3 / (z * y * x)

    @property
    def voxel_height_m(self) -> float:
        return self.height_m / self.grid_shape[0]
    
    @property
    def max_charge_w(self) -> float:
        return self.max_charge_kw * 1000.0
    @property
    def max_discharge_w(self) -> float:
        return self.max_discharge_kw * 1000.0

class SolarConfig(eqx.Module):
    model_type: Literal["simple", "passthrough"] = eqx.field(static=True, default="simple")
    panel_area_m2: float = eqx.field(static=True, default=20.0)
    efficiency: float = eqx.field(static=True, default=0.20)
    temp_coefficient: float = eqx.field(static=True, default=-0.004)
    reference_temp_c: float = eqx.field(static=True, default=25.0)

# --- Dynamic State Structs ---

@flax_dataclass
class ThermalState:
    T_vector: Array

@flax_dataclass
class BatteryState:
    soc: Array 
    soh: Array

@flax_dataclass
class ThermalStorageState:
    temperatures_c: Array # Now shape (Z, Y, X)
    
    @property
    def soc(self) -> Array:
        # Mean temperature metric for observation
        avg = jnp.mean(self.temperatures_c)
        return jnp.clip((avg - 30.0) / (60.0 - 30.0), 0.0, 1.0)

@flax_dataclass
class HeatPumpState:
    current_electrical_w: Array
    current_thermal_w: Array

@flax_dataclass
class AirConditionerState:
    current_electrical_w: Array
    current_thermal_w: Array

@flax_dataclass
class SystemState:
    thermal: ThermalState
    battery: BatteryState
    storage: ThermalStorageState
    heat_pump: HeatPumpState
    air_conditioner: AirConditionerState

@flax_dataclass
class ExogenousData:
    """All external data for a single timestep."""
    # --- Weather ---
    ambient_temp: Array       
    solar_irradiance_w_m2: Array
    wind_speed_m_s: Array     # <--- ADDED: Required for infiltration model

    # --- Price ---
    price: Array              

    # --- Loads (W) ---
    base_load_w: Array       
    ev_charger_load_w: Array     
    dishwasher_load_w: Array     
    clothes_dryer_load_w: Array 
    water_heater_load_w: Array   
    cooking_load_w: Array        

    # --- Thermal Gains (W) ---
    occupancy_gains_w: Array     
    solar_gains_w: Array         
    device_gains_w: Array        

@flax_dataclass
class SystemActions:
    battery_power_w: Array      
    heat_pump_power_w: Array    
    ac_power_w: Array           
    storage_discharge_w: Array  

@flax_dataclass
class HeatPumpOutput:
    thermal_power_w: Array     
    electrical_power_w: Array   

@flax_dataclass
class AirConditionerOutput:
    thermal_power_w: Array     
    electrical_power_w: Array   

@flax_dataclass
class ThermalStorageOutput:
    actual_discharge_w: Array    
    rejected_heat_w: Array
    standing_loss_w: Array      # <--- ADDED: Loss to be coupled to room

@flax_dataclass
class SolarOutput:
    pv_generation_w: Array