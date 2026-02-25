# core/models/solar_model.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import PVConfig, ExogenousData, PVOutput

class AbstractPVModel(eqx.Module):
    config: PVConfig

    @eqx.filter_jit
    def calculate(self, exogenous: ExogenousData) -> PVOutput:
        """Calculates PV output from exogenous weather."""
        raise NotImplementedError
    
class GeometricPVModel(AbstractPVModel):
    """
    High-Fidelity model considering location, day of year, 
    panel orientation, and temperature coefficients.
    """
    @eqx.filter_jit
    def calculate(self, exo: ExogenousData) -> PVOutput:
        # Constants
        DEG_2_RAD = jnp.pi / 180.0
        SOLAR_CONSTANT = 1361.0 # W/m2 (AM0) - used for theoretical max clamping

        # 1. Time Inputs
        # Assuming time_of_year_seconds is provided in exo data
        day_of_year = (exo.time_of_year_seconds // 86400) + 1
        hour_of_day = (exo.time_of_year_seconds % 86400) / 3600.0

        # 2. Solar Declination (delta) - Cooper's Equation
        delta = 23.45 * jnp.sin(DEG_2_RAD * (360.0 / 365.0) * (day_of_year + 284.0))
        delta_rad = delta * DEG_2_RAD

        # 3. Hour Angle (omega)
        # Simple approximation: Solar Noon is at 12:00
        # (For ultra-fidelity, implement Equation of Time correction here)
        omega = 15.0 * (hour_of_day - 12.0)
        omega_rad = omega * DEG_2_RAD

        # 4. Location & Panel Geometry
        lat_rad = self.config.latitude_deg * DEG_2_RAD
        tilt_rad = self.config.panel_tilt_deg * DEG_2_RAD
        p_azimuth_rad = (self.config.panel_azimuth_deg - 180.0) * DEG_2_RAD 
        # Note: Standard physics uses 0=South for calculation often, 
        # but here we normalize to 0=North input, converted to calculation frame.

        # 5. Solar Zenith (theta_z) & Elevation (alpha)
        sin_alpha = (jnp.sin(lat_rad) * jnp.sin(delta_rad)) + \
                    (jnp.cos(lat_rad) * jnp.cos(delta_rad) * jnp.cos(omega_rad))
        alpha_rad = jnp.arcsin(jnp.clip(sin_alpha, -1.0, 1.0))
        # Zenith is complement of elevation
        theta_z_rad = (jnp.pi / 2.0) - alpha_rad

        # 6. Solar Azimuth (gamma_s)
        cos_gamma_s = (jnp.sin(alpha_rad) * jnp.sin(lat_rad) - jnp.sin(delta_rad)) / \
                      (jnp.cos(alpha_rad) * jnp.cos(lat_rad) + 1e-6)
        gamma_s_rad = jnp.sign(omega_rad) * jnp.arccos(jnp.clip(cos_gamma_s, -1.0, 1.0))

        # 7. Angle of Incidence (theta)
        # cos(theta) = cos(theta_z)cos(tilt) + sin(theta_z)sin(tilt)cos(gamma_s - panel_azimuth)
        cos_theta = (jnp.cos(theta_z_rad) * jnp.cos(tilt_rad)) + \
                    (jnp.sin(theta_z_rad) * jnp.sin(tilt_rad) * jnp.cos(gamma_s_rad - p_azimuth_rad))
        
        cos_theta = jnp.fmax(0.0, cos_theta) # Clip negative incidence (sun behind panel)

        # 8. Plane of Array (POA) Irradiance
        # Ideally, we separate Diffuse and Direct components. 
        # High-Fidelity approx: GHI is mostly direct when sunny.
        # We apply the geometric factor to the GHI provided by weather file.
        poa_irradiance = exo.solar_irradiance_w_m2 * (cos_theta / (jnp.cos(theta_z_rad) + 1e-6))
        
        # Safety clamp: POA cannot physically exceed Solar Constant significantly on Earth
        poa_irradiance = jnp.clip(poa_irradiance, 0.0, SOLAR_CONSTANT)

        # 9. Temperature Correction & Power
        temp_factor = 1.0 + (exo.ambient_temp - self.config.reference_temp_c) * self.config.temp_coefficient
        power_w = poa_irradiance * self.config.panel_area_m2 * self.config.efficiency * temp_factor

        return PVOutput(pv_generation_w=jnp.fmax(0.0, power_w))

class SimplePVModel(AbstractPVModel):
    @eqx.filter_jit
    def calculate(self, exogenous: ExogenousData) -> PVOutput:
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
        return PVOutput(pv_generation_w=pv_generation_w)

class PassthroughPVModel(AbstractPVModel):
    """A dummy model for backward compatibility. 
    Assumes 'solar_irradiance_w_m2' is actually pre-computed PV power.
    """
    @eqx.filter_jit
    def calculate(self, exogenous: ExogenousData) -> PVOutput:
        # Pass through the value directly
        return PVOutput(pv_generation_w=exogenous.solar_irradiance_w_m2)