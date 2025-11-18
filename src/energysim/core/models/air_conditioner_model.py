import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import AirConditionerConfig, AirConditionerOutput, Array, ExogenousData

# --- 1. Abstract Base Class ---
class AbstractAirConditionerModel(eqx.Module):
    """Abstract base class for all AC models."""
    current_electrical_w: Array
    current_thermal_w: Array
    config: AirConditionerConfig
    n_rooms: int = eqx.field(static=True)

    @eqx.filter_jit
    def step(self,
             requested_electrical_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> tuple['AbstractAirConditionerModel', AirConditionerOutput]:
        raise NotImplementedError


# --- 2. Stateless (Instant) Implementation ---
class StatelessAirConditionerModel(AbstractAirConditionerModel):
    """Ramps instantly to the requested power, clipped by per-room max."""
    def __init__(self, config: AirConditionerConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms),
            current_thermal_w=jnp.zeros(n_rooms), # <-- Added
            config=config,
            n_rooms=n_rooms
        )

    @eqx.filter_jit
    def step(self,
             requested_electrical_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> tuple['StatelessAirConditionerModel', AirConditionerOutput]:
        
        # Clip against per-room power
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        actual_electrical_w = jnp.clip(
            requested_electrical_w,
            0.0,
            max_w_per_room
        )
        
        # Thermal power is NEGATIVE (removing heat)
        actual_thermal_w = - (actual_electrical_w * self.config.cop_cooling)
        
        output = AirConditionerOutput(
            thermal_power_w=actual_thermal_w,
            electrical_power_w=actual_electrical_w
        )
        
        # Update both states
        new_model = eqx.tree_at(
            lambda m: (m.current_electrical_w, m.current_thermal_w), 
            self, 
            (actual_electrical_w, actual_thermal_w)
        )
        return new_model, output


# --- 3. Ramping (Stateful) Implementation ---
class RampingAirConditionerModel(AbstractAirConditionerModel):
    def __init__(self, config: AirConditionerConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms),
            current_thermal_w=jnp.zeros(n_rooms),
            config=config,
            n_rooms=n_rooms
        )

    @eqx.filter_jit
    def step(self, requested_electrical_w: Array, exogenous: ExogenousData, dt_seconds: float) -> tuple['RampingAirConditionerModel', AirConditionerOutput]:
        
        # 1. Min Power
        max_w = self.config.max_electrical_power_w / self.n_rooms
        min_w = self.config.min_electrical_power_w / self.n_rooms
        target_w = jnp.where(
            requested_electrical_w < min_w, 0.0, jnp.clip(requested_electrical_w, 0.0, max_w)
        )

        # 2. Ramping
        max_delta = self.config.ramp_rate_w_per_sec * dt_seconds
        actual_elec = jnp.clip(target_w, self.current_electrical_w - max_delta, self.current_electrical_w + max_delta)

        # 3. Cooling (Negative Thermal Power)
        raw_thermal = -1.0 * actual_elec * self.config.cop_cooling

        # 4. Thermal Lag
        alpha = 1.0 - jnp.exp(-dt_seconds / self.config.tau_thermal_seconds)
        actual_thermal = (alpha * raw_thermal) + ((1.0 - alpha) * self.current_thermal_w)

        output = AirConditionerOutput(thermal_power_w=actual_thermal, electrical_power_w=actual_elec)
        
        new_model = eqx.tree_at(
            lambda m: (m.current_electrical_w, m.current_thermal_w), 
            self, (actual_elec, actual_thermal)
        )
        return new_model, output


# --- 4. Variable COP (Advanced) Implementation ---
class VariableCOPAirConditionerModel(AbstractAirConditionerModel):
    """Ramping + Variable COP based on ambient temperature + Thermal Lag."""
    cop_temps: Array
    cop_values: Array

    def __init__(self, config: AirConditionerConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms),
            current_thermal_w=jnp.zeros(n_rooms), # <-- Added
            config=config,
            n_rooms=n_rooms
        )
        self.cop_temps = jnp.array(config.cop_ambient_temps_c)
        self.cop_values = jnp.array(config.cop_values_cooling)

    @eqx.filter_jit
    def step(self,
             requested_electrical_w: Array,
             exogenous: ExogenousData, 
             dt_seconds: float
    ) -> tuple['VariableCOPAirConditionerModel', AirConditionerOutput]:
        
        # --- 1. Ramping Logic ---
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        # Apply min power threshold
        min_w_per_room = self.config.min_electrical_power_w / self.n_rooms
        
        target_electrical_w = jnp.clip(requested_electrical_w, 0.0, max_w_per_room)
        target_electrical_w = jnp.where(target_electrical_w < min_w_per_room, 0.0, target_electrical_w)

        max_delta_w = self.config.ramp_rate_w_per_sec * dt_seconds
        actual_electrical_w = jnp.clip(
            target_electrical_w, 
            self.current_electrical_w - max_delta_w, 
            self.current_electrical_w + max_delta_w
        )
        
        # --- 2. Variable COP Logic ---
        T_amb = exogenous.ambient_temp
        current_cop = jnp.interp(T_amb, self.cop_temps, self.cop_values)
        
        # Raw thermal generation (negative for cooling)
        raw_thermal_gen = -1.0 * actual_electrical_w * current_cop
        
        # --- 3. Thermal Lag ---
        alpha = 1.0 - jnp.exp(-dt_seconds / self.config.tau_thermal_seconds)
        actual_thermal_w = (alpha * raw_thermal_gen) + ((1.0 - alpha) * self.current_thermal_w)
        
        output = AirConditionerOutput(
            thermal_power_w=actual_thermal_w,
            electrical_power_w=actual_electrical_w
        )
        
        # --- 4. Update state ---
        new_model = eqx.tree_at(
            lambda m: (m.current_electrical_w, m.current_thermal_w), 
            self, 
            (actual_electrical_w, actual_thermal_w)
        )
        
        return new_model, output


# --- 5. Passthrough (Dummy) Implementation ---
class PassthroughAirConditionerModel(AbstractAirConditionerModel):
    """A dummy model for when no AC is present."""
    def __init__(self, config: AirConditionerConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms),
            current_thermal_w=jnp.zeros(n_rooms), # <-- Added
            config=config,
            n_rooms=n_rooms
        )

    @eqx.filter_jit
    def step(self,
             requested_electrical_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> tuple['PassthroughAirConditionerModel', AirConditionerOutput]:
        
        output = AirConditionerOutput(
            thermal_power_w=jnp.zeros_like(self.current_electrical_w),
            electrical_power_w=jnp.zeros_like(self.current_electrical_w)
        )
        return self, output