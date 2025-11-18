import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from typing import Optional

from ..shared.data_structs import ExogenousData

# --- Configuration ---

class NoiseConfig(eqx.Module):
    """
    Parameters for generating forecast errors.
    """
    sigma_base: float = 0.0      # Base standard deviation at t=0
    horizon_growth: float = 0.1  # How much sigma grows per step (linear cone)
    
    # AR1 specific (ignored by Simple Gaussian)
    theta: float = 0.1           # Mean reversion speed (0.0=Random Walk, 1.0=White Noise)

class ForecastConfig(eqx.Module):
    """Global config for all variables."""
    model_type: str = "gaussian" # "gaussian" or "ar1"
    
    temp: NoiseConfig = NoiseConfig(sigma_base=0.5, horizon_growth=0.05)
    irradiance: NoiseConfig = NoiseConfig(sigma_base=0.1, horizon_growth=0.02) # Multiplicative
    load: NoiseConfig = NoiseConfig(sigma_base=100.0, horizon_growth=10.0)
    price: NoiseConfig = NoiseConfig(sigma_base=0.01, horizon_growth=0.0)

# --- Abstract Base Class ---

class AbstractForecaster(eqx.Module):
    config: ForecastConfig = eqx.field(static=True)
    
    def __init__(self, config: ForecastConfig):
        self.config = config

    @abstractmethod
    def calculate(self, key: jax.Array, ground_truth_slice: ExogenousData) -> ExogenousData:
        """Computes the noisy forecast."""
        pass

# --- 1. Simple Gaussian Cone Forecaster (Your Request) ---

class GaussianNoiseForecaster(AbstractForecaster):
    """
    Takes ground truth data and adds uncorrelated white noise 
    that linearly increases in variance over the horizon.
    
    Error[t] ~ N(0, sigma_base + (t * growth))
    """
    
    @jax.jit
    def _generate_cone(self, key: jax.Array, n_steps: int, conf: NoiseConfig) -> jax.Array:
        # 1. Base White Noise: N(0, 1)
        noise_std = jax.random.normal(key, (n_steps,))
        
        # 2. The Cone of Uncertainty
        # Creates a vector [0, 1, 2, ..., N-1]
        time_steps = jnp.arange(n_steps, dtype=jnp.float32)
        
        # Sigma[t] = Base + (Growth * t)
        sigma_profile = conf.sigma_base + (conf.horizon_growth * time_steps)
        
        return noise_std * sigma_profile

    @jax.jit
    def calculate(self, key: jax.Array, gt: ExogenousData) -> ExogenousData:
        keys = jax.random.split(key, 4)
        N = gt.ambient_temp.shape[0]

        # 1. Temperature (Additive)
        noise_t = self._generate_cone(keys[0], N, self.config.temp)
        fc_temp = gt.ambient_temp + noise_t

        # 2. Irradiance (Multiplicative + Physics Check)
        # We center noise around 1.0. Clip to prevent negative sun or >200% sun.
        noise_irr = self._generate_cone(keys[1], N, self.config.irradiance)
        # E.g., if noise is 0.2, factor is 1.2
        irr_factor = jnp.clip(1.0 + noise_irr, 0.0, 3.0) 
        
        fc_irr = gt.solar_irradiance_w_m2 * irr_factor
        fc_gains = gt.solar_gains_w * irr_factor

        # 3. Load (Additive + Positive Clip)
        noise_load = self._generate_cone(keys[2], N, self.config.load)
        fc_load = jnp.fmax(0.0, gt.base_load_w + noise_load)

        # 4. Price (Additive + Positive Clip)
        noise_price = self._generate_cone(keys[3], N, self.config.price)
        fc_price = jnp.fmax(0.0, gt.price + noise_price)

        # Reconstruct using functional update (equinox tree_at)
        return eqx.tree_at(
            lambda d: (d.ambient_temp, d.solar_irradiance_w_m2, d.solar_gains_w, d.base_load_w, d.price),
            gt,
            (fc_temp, fc_irr, fc_gains, fc_load, fc_price)
        )

# --- 2. AR1 (Correlated) Forecaster (Higher Fidelity) ---

class AR1Forecaster(AbstractForecaster):
    """
    Generates noise that is time-correlated (Autoregressive).
    If it over-predicts at t=0, it likely over-predicts at t=1.
    Also scales with a cone.
    """
    
    @jax.jit
    def _generate_ar1_cone(self, key: jax.Array, n_steps: int, conf: NoiseConfig) -> jax.Array:
        # 1. Base White Noise
        white_noise = jax.random.normal(key, (n_steps,)) * conf.sigma_base

        # 2. AR1 Scan Loop: x_t = (1-theta)*x_{t-1} + noise
        def scan_fn(carry, noise_in):
            val = (1.0 - conf.theta) * carry + noise_in
            return val, val
            
        _, correlated_noise = jax.lax.scan(scan_fn, 0.0, white_noise)

        # 3. Apply Cone Scaling (Linear scaling on top of correlation)
        time_steps = jnp.linspace(1.0, 1.0 + (conf.horizon_growth * n_steps), n_steps)
        
        return correlated_noise * time_steps

    @jax.jit
    def calculate(self, key: jax.Array, gt: ExogenousData) -> ExogenousData:
        # Identical structure to Gaussian, just calls _generate_ar1_cone
        keys = jax.random.split(key, 4)
        N = gt.ambient_temp.shape[0]

        noise_t = self._generate_ar1_cone(keys[0], N, self.config.temp)
        fc_temp = gt.ambient_temp + noise_t

        noise_irr = self._generate_ar1_cone(keys[1], N, self.config.irradiance)
        irr_factor = jnp.clip(1.0 + noise_irr, 0.0, 3.0)
        fc_irr = gt.solar_irradiance_w_m2 * irr_factor
        fc_gains = gt.solar_gains_w * irr_factor

        noise_load = self._generate_ar1_cone(keys[2], N, self.config.load)
        fc_load = jnp.fmax(0.0, gt.base_load_w + noise_load)

        noise_price = self._generate_ar1_cone(keys[3], N, self.config.price)
        fc_price = jnp.fmax(0.0, gt.price + noise_price)

        return eqx.tree_at(
            lambda d: (d.ambient_temp, d.solar_irradiance_w_m2, d.solar_gains_w, d.base_load_w, d.price),
            gt,
            (fc_temp, fc_irr, fc_gains, fc_load, fc_price)
        )