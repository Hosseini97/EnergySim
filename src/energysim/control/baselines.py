import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple

from ..core.shared.data_structs import (
    SystemState, SystemActions, ExogenousData, 
    ThermalConfig, BatteryConfig, HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig
)

# ==========================================
# 1. Abstract Interface
# ==========================================

class AbstractController(eqx.Module):
    """
    Base class for all baseline controllers.
    Must return (Action, NextControllerState).
    """
    def __call__(
        self, 
        state: SystemState, 
        exo: ExogenousData, 
        dt: float
    ) -> Tuple[SystemActions, 'AbstractController']:
        raise NotImplementedError


# ==========================================
# 2. HVAC Baselines (The "Floor")
# ==========================================

class BangBangThermostat(AbstractController):
    """
    The absolute floor. A naive hysteresis controller.
    If T < Setpoint - Deadband: HEAT MAX
    If T > Setpoint + Deadband: COOL MAX
    """
    t_config: ThermalConfig = eqx.field(static=True)
    hp_config: HeatPumpConfig = eqx.field(static=True)
    ac_config: AirConditionerConfig = eqx.field(static=True)
    n_rooms: int = eqx.field(static=True)

    def __init__(self, t_config, hp_config, ac_config):
        self.t_config = t_config
        self.hp_config = hp_config
        self.ac_config = ac_config
        self.n_rooms = len(t_config.room_air_indices)

    @jax.jit
    def __call__(self, state: SystemState, exo: ExogenousData, dt: float):
        # 1. Extract Room Temps
        room_indices = jnp.array(self.t_config.room_air_indices)
        current_temps = state.thermal.T_vector[room_indices]
        
        setpoint = self.t_config.setpoint
        band = self.t_config.comfort_band

        # 2. Logic
        # Heat if below lower bound
        needs_heating = current_temps < (setpoint - band)
        # Cool if above upper bound
        needs_cooling = current_temps > (setpoint + band)

        # 3. Map to Actions (Per Zone)
        hp_max = self.hp_config.max_electrical_power_w / self.n_rooms
        ac_max = self.ac_config.max_electrical_power_w / self.n_rooms

        hp_action = jnp.where(needs_heating, hp_max, 0.0)
        ac_action = jnp.where(needs_cooling, ac_max, 0.0)

        # Prevent simultaneous heating and cooling (safety interlock)
        # If both triggered (unlikely with deadband, but mathematically possible), prioritize cooling
        hp_action = jnp.where(needs_cooling, 0.0, hp_action)

        actions = SystemActions(
            battery_power_w=jnp.array(0.0), # Handled by composite or ignored
            heat_pump_power_w=hp_action,
            ac_power_w=ac_action,
            storage_discharge_w=jnp.zeros(self.n_rooms)
        )
        
        return actions, self


class PIDThermostat(AbstractController):
    """
    Industrial Standard. Continuous modulation based on error.
    Maintains state (integral, prev_error) functionally.
    
    u(t) = Kp*e + Ki*integral + Kd*derivative
    """
    # Configs
    t_config: ThermalConfig = eqx.field(static=True)
    hp_config: HeatPumpConfig = eqx.field(static=True)
    ac_config: AirConditionerConfig = eqx.field(static=True)
    n_rooms: int = eqx.field(static=True)

    # Gains (Tunable)
    kp: float = eqx.field(static=True)
    ki: float = eqx.field(static=True)
    kd: float = eqx.field(static=True)

    # Dynamic State (Learned/Updated over time)
    integral_error: jax.Array
    prev_error: jax.Array

    def __init__(
        self, t_config, hp_config, ac_config, 
        kp=2000.0, ki=10.0, kd=100.0, 
        initial_integral=None, initial_prev_error=None
    ):
        self.t_config = t_config
        self.hp_config = hp_config
        self.ac_config = ac_config
        self.n_rooms = len(t_config.room_air_indices)
        
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Initialize state to zeros if not provided
        if initial_integral is None:
            self.integral_error = jnp.zeros(self.n_rooms)
        else:
            self.integral_error = initial_integral

        if initial_prev_error is None:
            self.prev_error = jnp.zeros(self.n_rooms)
        else:
            self.prev_error = initial_prev_error

    @jax.jit
    def __call__(self, state: SystemState, exo: ExogenousData, dt: float):
        room_indices = jnp.array(self.t_config.room_air_indices)
        current_temps = state.thermal.T_vector[room_indices]
        
        # Error: Positive means room is COLD (needs heating)
        # Negative means room is HOT (needs cooling)
        error = self.t_config.setpoint - current_temps

        # P
        p_term = self.kp * error
        
        # I (Trapezoidal integration)
        new_integral = self.integral_error + error * dt
        # Anti-windup: Clamp integral to prevent runaway limits
        integral_limit = 5000.0 # Watts worth of error accumulation
        new_integral = jnp.clip(new_integral, -integral_limit, integral_limit)
        i_term = self.ki * new_integral

        # D
        d_term = self.kd * (error - self.prev_error) / dt

        # Control Signal (Watts thermal required roughly)
        u = p_term + i_term + d_term

        # Split into Heating vs Cooling actuators
        # u > 0 -> Heating, u < 0 -> Cooling
        
        hp_max = self.hp_config.max_electrical_power_w / self.n_rooms
        ac_max = self.ac_config.max_electrical_power_w / self.n_rooms

        # Heat Pump Logic
        hp_req = jnp.maximum(0.0, u)
        hp_action = jnp.clip(hp_req, 0.0, hp_max)

        # AC Logic
        ac_req = jnp.maximum(0.0, -u)
        ac_action = jnp.clip(ac_req, 0.0, ac_max)

        actions = SystemActions(
            battery_power_w=jnp.array(0.0),
            heat_pump_power_w=hp_action,
            ac_power_w=ac_action,
            storage_discharge_w=jnp.zeros(self.n_rooms)
        )

        # Return actions AND updated controller state
        new_controller = eqx.tree_at(
            lambda c: (c.integral_error, c.prev_error), 
            self, 
            (new_integral, error)
        )

        return actions, new_controller


# ==========================================
# 3. Battery Baselines
# ==========================================

class TimeOfUseBattery(AbstractController):
    """
    Price-Arbitrage Baseline.
    - Charge if Price < Low_Threshold
    - Discharge if Price > High_Threshold
    - Else Idle
    """
    b_config: BatteryConfig = eqx.field(static=True)
    price_low_q: float = eqx.field(static=True) # Quantile for "Cheap"
    price_high_q: float = eqx.field(static=True) # Quantile for "Expensive"
    
    # Note: In a real RBC, these thresholds might be fixed numbers.
    # For a robust baseline, we often assume the agent "knows" what implies cheap/expensive 
    # relative to the dataset average.
    
    def __init__(self, b_config, price_low_threshold=0.20, price_high_threshold=0.35):
        self.b_config = b_config
        self.price_low_q = price_low_threshold
        self.price_high_q = price_high_threshold

    @jax.jit
    def __call__(self, state: SystemState, exo: ExogenousData, dt: float):
        current_price = exo.price
        
        # Logic
        is_cheap = current_price <= self.price_low_q
        is_expensive = current_price >= self.price_high_q
        
        # Power (Watts)
        # Charge (+)
        charge_power = jnp.where(is_cheap, self.b_config.max_power_w, 0.0)
        # Discharge (-)
        discharge_power = jnp.where(is_expensive, -self.b_config.max_power_w, 0.0)
        
        # Combine (Mutual exclusivity implied by thresholds, but sum is safe if threshold logic holds)
        power = charge_power + discharge_power

        # SOC Constraints (Physical Clamps)
        # Don't charge if full
        power = jnp.where((state.battery.soc >= 0.98) & (power > 0), 0.0, power)
        # Don't discharge if empty
        power = jnp.where((state.battery.soc <= 0.02) & (power < 0), 0.0, power)

        # Return dummy struct (other fields zeroed, allows composition)
        actions = SystemActions(
            battery_power_w=power,
            heat_pump_power_w=jnp.array([]), # Shape mismatch handled by composite
            ac_power_w=jnp.array([]),
            storage_discharge_w=jnp.array([])
        )
        
        return actions, self

# ==========================================
# 4. Composite Baseline (The Strategy Mixer)
# ==========================================

class CompositeBaseline(AbstractController):
    """
    Combines an HVAC controller and a Battery controller into one system action.
    """
    hvac_controller: AbstractController
    battery_controller: AbstractController

    def __init__(self, hvac_controller, battery_controller):
        self.hvac_controller = hvac_controller
        self.battery_controller = battery_controller

    @jax.jit
    def __call__(self, state: SystemState, exo: ExogenousData, dt: float):
        # 1. Get Sub-Actions
        hvac_act, new_hvac = self.hvac_controller(state, exo, dt)
        batt_act, new_batt = self.battery_controller(state, exo, dt)

        # 2. Merge
        # We assume hvac_controller handles HP/AC/Storage and Battery handles Battery
        combined_actions = SystemActions(
            battery_power_w=batt_act.battery_power_w,
            heat_pump_power_w=hvac_act.heat_pump_power_w,
            ac_power_w=hvac_act.ac_power_w,
            storage_discharge_w=hvac_act.storage_discharge_w
        )

        # 3. Update self with new sub-controllers
        new_self = eqx.tree_at(
            lambda m: (m.hvac_controller, m.battery_controller),
            self,
            (new_hvac, new_batt)
        )

        return combined_actions, new_self