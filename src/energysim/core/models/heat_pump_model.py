import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import HeatPumpConfig, HeatPumpOutput, Array, ExogenousData

class AbstractHeatPumpModel(eqx.Module):
    current_electrical_w: Array
    current_thermal_w: Array
    config: HeatPumpConfig = eqx.field(static=True)
    n_rooms: int = eqx.field(static=True)

    @eqx.filter_jit
    def step(self, requested_electrical_w: Array, exogenous: ExogenousData, dt_seconds: float) -> tuple['AbstractHeatPumpModel', HeatPumpOutput]:
        raise NotImplementedError
    
    
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
    

class RampingHeatPumpModel(AbstractHeatPumpModel):
    def __init__(self, config: HeatPumpConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms),
            current_thermal_w=jnp.zeros(n_rooms),
            config=config,
            n_rooms=n_rooms
        )

    @eqx.filter_jit
    def step(self, requested_electrical_w: Array, exogenous: ExogenousData, dt_seconds: float) -> tuple['RampingHeatPumpModel', HeatPumpOutput]:
        
        # 1. Minimum Power Constraints (Snap to 0 if below min)
        # If ramping down, we might get stuck > 0. Logic: If target < min, target=0.
        max_w_per_room = self.config.max_electrical_power_w / self.n_rooms
        min_w_per_room = self.config.min_electrical_power_w / self.n_rooms
        
        target_w = jnp.clip(requested_electrical_w, 0.0, max_w_per_room)
        target_w = jnp.where(target_w < min_w_per_room, 0.0, target_w)

        # 2. Ramping
        max_delta = self.config.ramp_rate_w_per_sec * dt_seconds
        actual_elec = jnp.clip(
            target_w, 
            self.current_electrical_w - max_delta, 
            self.current_electrical_w + max_delta
        )
        
        # 3. COP Calculation
        raw_thermal_gen = actual_elec * self.config.cop_heating

        # 4. Thermal Lag (First Order Filter)
        # y[t] = alpha * x[t] + (1-alpha) * y[t-1]
        # alpha = 1 - exp(-dt / tau)
        alpha = 1.0 - jnp.exp(-dt_seconds / self.config.tau_thermal_seconds)
        actual_thermal = (alpha * raw_thermal_gen) + ((1.0 - alpha) * self.current_thermal_w)

        output = HeatPumpOutput(
            thermal_power_w=actual_thermal,
            electrical_power_w=actual_elec
        )

        new_model = eqx.tree_at(
            lambda m: (m.current_electrical_w, m.current_thermal_w), 
            self, 
            (actual_elec, actual_thermal)
        )
        return new_model, output

class VariableCOPHeatPumpModel(AbstractHeatPumpModel):
    cop_temps: Array
    cop_values: Array

    def __init__(self, config: HeatPumpConfig, n_rooms: int):
        super().__init__(
            current_electrical_w=jnp.zeros(n_rooms),
            current_thermal_w=jnp.zeros(n_rooms),
            config=config,
            n_rooms=n_rooms
        )
        self.cop_temps = jnp.array(config.cop_ambient_temps_c)
        self.cop_values = jnp.array(config.cop_values_heating)

    @eqx.filter_jit
    def step(self, requested_electrical_w: Array, exogenous: ExogenousData, dt_seconds: float) -> tuple['VariableCOPHeatPumpModel', HeatPumpOutput]:
        
        # 1. Min Power & Ramping
        max_w = self.config.max_electrical_power_w / self.n_rooms
        min_w = self.config.min_electrical_power_w / self.n_rooms
        
        target_w = jnp.where(
            requested_electrical_w < min_w, 0.0, jnp.clip(requested_electrical_w, 0.0, max_w)
        )
        
        max_delta = self.config.ramp_rate_w_per_sec * dt_seconds
        actual_elec = jnp.clip(target_w, self.current_electrical_w - max_delta, self.current_electrical_w + max_delta)
        
        # 2. Variable COP
        cop = jnp.interp(exogenous.ambient_temp, self.cop_temps, self.cop_values)
        raw_thermal_gen = actual_elec * cop
        
        # 3. Thermal Lag
        alpha = 1.0 - jnp.exp(-dt_seconds / self.config.tau_thermal_seconds)
        actual_thermal = (alpha * raw_thermal_gen) + ((1.0 - alpha) * self.current_thermal_w)
        
        output = HeatPumpOutput(thermal_power_w=actual_thermal, electrical_power_w=actual_elec)
        
        new_model = eqx.tree_at(
            lambda m: (m.current_electrical_w, m.current_thermal_w), 
            self, (actual_elec, actual_thermal)
        )
        return new_model, output

# Passthrough remains similar but must initialize current_thermal_w
class PassthroughHeatPumpModel(AbstractHeatPumpModel):
    def __init__(self, config: HeatPumpConfig, n_rooms: int):
        super().__init__(jnp.zeros(n_rooms), jnp.zeros(n_rooms), config, n_rooms)
    
    @eqx.filter_jit
    def step(self, requested_electrical_w: Array, exo: ExogenousData, dt: float):
        return self, HeatPumpOutput(jnp.zeros_like(self.current_electrical_w), jnp.zeros_like(self.current_electrical_w))