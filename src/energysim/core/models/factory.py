# energysim/core/models/factory.py
import jax.numpy as jnp
import equinox as eqx
from typing import Optional

# Import models
from energysim.core.models.battery_model import (
    AbstractBatteryModel, SimpleBatteryModel,
    DegradationBatteryModel, PassthroughBatteryModel
)
# --- MODIFIED: Import new thermal models ---
from energysim.core.models.thermal_model import (
    AbstractThermalModel, ThermalModel_1R1C,
    ThermalModel_2R2C, ThermalModel_3R2C, ThermalModel_4R3C,
    PassthroughThermalModel
)
from energysim.core.models.heat_pump_model import (
    AbstractHeatPumpModel, PassthroughHeatPumpModel, RampingHeatPumpModel,
    StatelessHeatPumpModel, VariableCOPHeatPumpModel
)
from energysim.core.models.air_conditioner_model import (
    AbstractAirConditionerModel, PassthroughAirConditionerModel, RampingAirConditionerModel,
    StatelessAirConditionerModel, VariableCOPAirConditionerModel
)
from energysim.core.models.thermal_storage_model import (
    AbstractThermalStorage, ThermalStorageModel, ThermalStoragePassthrough
)
from energysim.core.models.solar_model import (
    AbstractSolarModel, SimpleSolarModel, PassthroughSolarModel
)


# Import configs and dummies
from energysim.core.shared.data_structs import (
    BatteryConfig, ThermalConfig, HeatPumpConfig,
    AirConditionerConfig, ThermalStorageConfig,
    SolarConfig
)

# ... (Dummy configs are unchanged) ...
DUMMY_STORAGE_CONFIG = ThermalStorageConfig(
    capacity_kwh=0.0,
    max_charge_kw=0.0,
    max_discharge_kw=0.0,
    standing_loss_rate=0.0
)
DUMMY_BATTERY_CONFIG = BatteryConfig(capacity_kwh=0.0, max_power_kw=0.0, efficiency=1.0)
DUMMY_HP_CONFIG = HeatPumpConfig(max_electrical_power_w=0.0, cop_heating=1.0)
DUMMY_AC_CONFIG = AirConditionerConfig(max_electrical_power_w=0.0, cop_cooling=1.0)
DUMMY_SOLAR_CONFIG = SolarConfig(model_type="passthrough", panel_area_m2=0.0)


# --- Factory Functions ---

# ... (create_battery, create_heat_pump, create_ac, create_storage are unchanged) ...
def create_battery(config: Optional[BatteryConfig]) -> AbstractBatteryModel:
    if config:
        if config.model_type == "simple":
            return SimpleBatteryModel(config, initial_soc=0.5)
        elif config.model_type == "degradation":
            return DegradationBatteryModel(config, initial_soc=0.5, initial_soh=1.0)
        else:
            raise ValueError(f"Unknown battery model_type: {config.model_type}")
    else:
        return PassthroughBatteryModel(DUMMY_BATTERY_CONFIG)

def create_heat_pump(config: Optional[HeatPumpConfig]) -> AbstractHeatPumpModel:
    if config:
        if config.model_type == "stateless":
            return StatelessHeatPumpModel(config)
        elif config.model_type == "ramping":
            return RampingHeatPumpModel(config)
        elif config.model_type == "variable_cop":
            return VariableCOPHeatPumpModel(config)
        else:
            raise ValueError(f"Unknown heat_pump model_type: {config.model_type}")
    else:
        return PassthroughHeatPumpModel(DUMMY_HP_CONFIG)

def create_ac(config: Optional[AirConditionerConfig]) -> AbstractAirConditionerModel:
    if config:
        if config.model_type == "stateless":
            return StatelessAirConditionerModel(config)
        elif config.model_type == "ramping":
            return RampingAirConditionerModel(config)
        elif config.model_type == "variable_cop":
            return VariableCOPAirConditionerModel(config)
        else:
            raise ValueError(f"Unknown ac model_type: {config.model_type}")
    else:
        return PassthroughAirConditionerModel(DUMMY_AC_CONFIG)

def create_storage(config: Optional[ThermalStorageConfig]) -> AbstractThermalStorage:
    if config:
        return ThermalStorageModel(config, initial_soc=0.5)
    else:
        return ThermalStoragePassthrough(DUMMY_STORAGE_CONFIG)

def create_thermal(config: ThermalConfig) -> AbstractThermalModel:
    """Factory function for thermal models."""
    initial_temp = config.setpoint # Start at setpoint

    # --- MODIFIED: Add new model types ---
    if config.model_type == "1R1C":
        return ThermalModel_1R1C(config, initial_temp=initial_temp)
    elif config.model_type == "2R2C":
        return ThermalModel_2R2C(config, initial_temp=initial_temp)
    elif config.model_type == "3R2C":
        return ThermalModel_3R2C(config, initial_temp=initial_temp)
    elif config.model_type == "4R3C":
        return ThermalModel_4R3C(config, initial_temp=initial_temp)
    elif config.model_type == "passthrough":
        return PassthroughThermalModel(config, initial_temp=initial_temp)
    else:
        raise ValueError(f"Unknown thermal model_type: {config.model_type}")

def create_solar(config: Optional[SolarConfig]) -> AbstractSolarModel:
# ... (create_solar is unchanged) ...
    """Factory function for solar PV models."""
    if config:
        if config.model_type == "simple":
            return SimpleSolarModel(config)
        elif config.model_type == "passthrough":
            return PassthroughSolarModel(config)
        else:
            raise ValueError(f"Unknown solar model_type: {config.model_type}")
    else:
        return PassthroughSolarModel(DUMMY_SOLAR_CONFIG)