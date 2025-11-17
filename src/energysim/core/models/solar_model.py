# core/models/solar_model.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import SolarConfig, ExogenousData, SolarOutput

class AbstractSolarModel(eqx.Module):
    config: SolarConfig = eqx.field(static=True)

    @eqx.filter_jit
    def calculate(self, exogenous: ExogenousData) -> SolarOutput:
        """Calculates PV output from exogenous weather."""
        raise NotImplementedError

class SimpleSolarModel(AbstractSolarModel):
    @eqx.filter_jit
    def calculate(self, exogenous: ExogenousData) -> SolarOutput:
        # 1. Get Irradiance (W/m^2)
        irradiance_w_m2 = exogenous.solar_irradiance_w_m2
        
        # 2. Calculate temperature correction
        T_amb = exogenous.ambient_temp
        temp_factor = 1.0 + (T_amb - self.config.reference_temp_c) * self.config.temp_coefficient
        
        # 3. Calculate Power (W)
        power_w = (
            irradiance_w_m2 
            * self.config.panel_area_m2 
            * self.config.efficiency 
            * temp_factor
        )
        
        # Clip at 0 (no negative generation)
        pv_generation_w = jnp.fmax(0.0, power_w)
        return SolarOutput(pv_generation_w=pv_generation_w)

class PassthroughSolarModel(AbstractSolarModel):
    """A dummy model for backward compatibility. 
    Assumes 'solar_irradiance_w_m2' is actually pre-computed PV power.
    """
    @eqx.filter_jit
    def calculate(self, exogenous: ExogenousData) -> SolarOutput:
        # Pass through the value directly
        return SolarOutput(pv_generation_w=exogenous.solar_irradiance_w_m2)