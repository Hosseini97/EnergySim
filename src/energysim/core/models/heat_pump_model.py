import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import HeatPumpConfig, HeatPumpOutput, Array, ExogenousData

# --- 1. Abstract Base Class ---
class AbstractHeatPumpModel(eqx.Module):
    """Abstract base class for all heat pump models."""
    current_electrical_w: Array
    config: HeatPumpConfig = eqx.field(static=True)
    n_rooms: int = eqx.field(static=True) # <-- NEW: Store n_rooms

    @eqx.filter_jit
    def step(self,
             requested_electrical_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> tuple['AbstractHeatPumpModel', HeatPumpOutput]:
        raise NotImplementedError


# --- 2. Stateless (Instant) Implementation ---
class StatelessHeatPumpModel(AbstractHeatPumpModel):
    """Ramps instantly to the requested power, clipped by per-room max."""
    def __init__(self, config: HeatPumpConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms), # <-- UPDATED
            config=config,
            n_rooms=n_rooms # <-- UPDATED
        )

    @eqx.filter_jit
    def step(self,
             requested_electrical_w: Array,
             exogenous: ExogenousData, 
             dt_seconds: float
    ) -> tuple['StatelessHeatPumpModel', HeatPumpOutput]:
        
        # --- UPDATED: Clip against per-room power ---
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        actual_electrical_w = jnp.clip(
            requested_electrical_w,
            0.0, 
            max_w_per_room
        )
        
        actual_thermal_w = actual_electrical_w * self.config.cop_heating
        
        output = HeatPumpOutput(
            thermal_power_w=actual_thermal_w,
            electrical_power_w=actual_electrical_w
        )
        
        new_model = eqx.tree_at(
            lambda m: m.current_electrical_w, self, actual_electrical_w
        )
        
        return new_model, output

# --- 3. Ramping (Stateful) Implementation ---
class RampingHeatPumpModel(AbstractHeatPumpModel):
    """A stateful model that limits the rate of change (ramping)."""
    def __init__(self, config: HeatPumpConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms), # <-- UPDATED
            config=config,
            n_rooms=n_rooms # <-- UPDATED
        )

    @eqx.filter_jit
    def step(self,
             requested_electrical_w: Array,
             exogenous: ExogenousData, 
             dt_seconds: float
    ) -> tuple['RampingHeatPumpModel', HeatPumpOutput]:
        
        # --- UPDATED: Clip target against per-room power ---
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        target_electrical_w = jnp.clip(
            requested_electrical_w,
            0.0,
            max_w_per_room
        )
        
        max_delta_w = self.config.ramp_rate_w_per_sec * dt_seconds
        
        lower_ramp_limit = self.current_electrical_w - max_delta_w
        upper_ramp_limit = self.current_electrical_w + max_delta_w
        
        actual_electrical_w = jnp.clip(
            target_electrical_w, lower_ramp_limit, upper_ramp_limit
        )
        
        actual_thermal_w = actual_electrical_w * self.config.cop_heating
        
        output = HeatPumpOutput(
            thermal_power_w=actual_thermal_w,
            electrical_power_w=actual_electrical_w
        )
        
        new_model = eqx.tree_at(
            lambda m: m.current_electrical_w, self, actual_electrical_w
        )
        
        return new_model, output

# --- 4. Variable COP (Advanced) Implementation ---
class VariableCOPHeatPumpModel(AbstractHeatPumpModel):
    """Ramping + Variable COP based on ambient temperature."""
    cop_temps: Array
    cop_values: Array

    def __init__(self, config: HeatPumpConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms), # <-- UPDATED
            config=config,
            n_rooms=n_rooms # <-- UPDATED
        )
        self.cop_temps = jnp.array(config.cop_ambient_temps_c)
        self.cop_values = jnp.array(config.cop_values_heating)

    @eqx.filter_jit
    def step(self,
             requested_electrical_w: Array,
             exogenous: ExogenousData, 
             dt_seconds: float
    ) -> tuple['VariableCOPHeatPumpModel', HeatPumpOutput]:
        
        # --- 1. Ramping Logic ---
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        target_electrical_w = jnp.clip(
            requested_electrical_w, 0.0, max_w_per_room
        )
        max_delta_w = self.config.ramp_rate_w_per_sec * dt_seconds
        lower_ramp_limit = self.current_electrical_w - max_delta_w
        upper_ramp_limit = self.current_electrical_w + max_delta_w
        actual_electrical_w = jnp.clip(
            target_electrical_w, lower_ramp_limit, upper_ramp_limit
        )
        
        # --- 2. Variable COP Logic ---
        T_amb = exogenous.ambient_temp
        current_cop = jnp.interp(T_amb, self.cop_temps, self.cop_values)
        
        # 3. Calculate thermal generation
        actual_thermal_w = actual_electrical_w * current_cop
        
        output = HeatPumpOutput(
            thermal_power_w=actual_thermal_w,
            electrical_power_w=actual_electrical_w
        )
        
        # 4. Update state
        new_model = eqx.tree_at(
            lambda m: m.current_electrical_w, self, actual_electrical_w
        )
        
        return new_model, output

# --- 5. Passthrough (Dummy) Implementation ---
class PassthroughHeatPumpModel(AbstractHeatPumpModel):
    """A dummy model for when no heat pump is present."""
    def __init__(self, config: HeatPumpConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms), # <-- UPDATED
            config=config,
            n_rooms=n_rooms # <-- UPDATED
        )

    @eqx.filter_jit
    def step(self,
             requested_electrical_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> tuple['PassthroughHeatPumpModel', HeatPumpOutput]:
        
        output = HeatPumpOutput(
            thermal_power_w=jnp.zeros_like(self.current_electrical_w), # Return zonal 0
            electrical_power_w=jnp.zeros_like(self.current_electrical_w) # Return zonal 0
        )
        return self, output