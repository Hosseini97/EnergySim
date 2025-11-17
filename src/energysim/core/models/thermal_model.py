# energysim/core/models/thermal_model.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import ThermalConfig, ExogenousData, Array

# --- 1. Abstract Base Class ---
class AbstractThermalModel(eqx.Module):
    """Abstract base class for all thermal models."""
    # --- MODIFIED: Added mass_temp ---
    room_temp: Array
    wall_temp: Array # Dummy for 1R1C, real for 2R2C+
    mass_temp: Array # Dummy for 1R1C/2R2C, real for 4R3C

    # --- Static Config ---
    config: ThermalConfig = eqx.field(static=True)

    @eqx.filter_jit
    def step(self,
             storage_discharge_w: Array,
             ac_thermal_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> 'AbstractThermalModel':
        """The 'step' function is the contract all models must fulfill."""
        raise NotImplementedError

    @eqx.filter_jit
    def _get_total_gains(self, exogenous: ExogenousData) -> tuple[Array, Array]:
        """
        Helper to calculate total gains and split solar gains.
        Returns: (total_air_gains, solar_gain_to_wall, solar_gain_to_mass)
        """
        # 1. Non-solar gains all go to the air
        net_passive_w = (
            exogenous.occupancy_gains_w  # Heat from people
            + exogenous.device_gains_w   # Heat from appliances
        )
        
        # 2. Split solar gains
        solar_total_w = exogenous.solar_gains_w
        
        # Get fractions from config
        f_wall = self.config.solar_gain_to_wall_frac
        f_mass = self.config.solar_gain_to_mass_frac
        
        # Ensure fractions are only used by models that support them
        is_4r3c = self.config.model_type == "4R3C"
        
        solar_to_wall_w = jnp.where(is_4r3c, solar_total_w * f_wall, 0.0)
        solar_to_mass_w = jnp.where(is_4r3c, solar_total_w * f_mass, 0.0)
        
        solar_to_air_w = solar_total_w - solar_to_wall_w - solar_to_mass_w
        
        total_air_gains = net_passive_w + solar_to_air_w
        
        return total_air_gains, solar_to_wall_w, solar_to_mass_w


# --- 2. 1R1C Implementation ---
class ThermalModel_1R1C(AbstractThermalModel):
    """
    1-Resistor, 1-Capacitor model.
    Models only the room air, with a single resistance to ambient.
    """
    def __init__(self, config: ThermalConfig, initial_temp: float = 20.0):
        super().__init__(
            room_temp=jnp.array(initial_temp),
            wall_temp=jnp.array(initial_temp), # Dummy state, tracks room temp
            mass_temp=jnp.array(initial_temp)  # Dummy state, tracks room temp
        )

    @eqx.filter_jit
    def step(self,
             storage_discharge_w: Array,
             ac_thermal_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> 'ThermalModel_1R1C':

        # 1. Get model parameters from config
        C_air = self.config.air_capacity_j_k
        R_air_amb = 1.0 / self.config.air_to_ambient_coeff_w_k

        T_room = self.room_temp
        T_amb = exogenous.ambient_temp

        # 2. Total thermal power (W)
        net_controllable_w = storage_discharge_w + ac_thermal_w

        # --- UPDATED GAINS ---
        # For 1R1C, all gains go to the air
        net_passive_w = (
            exogenous.occupancy_gains_w
            + exogenous.device_gains_w
            + exogenous.solar_gains_w # All solar gain hits air
        )
        net_thermal_input_w = net_controllable_w + net_passive_w

        # 3. Calculate temperature change
        # dT/dt = (1/C) * (P_in + (T_amb - T_room) / R)
        dT_gains_k = (net_thermal_input_w / C_air) * dt_seconds
        dT_loss_k = ((T_amb - T_room) / (R_air_amb * C_air)) * dt_seconds
        next_temp = T_room + dT_gains_k + dT_loss_k

        # Return new model with updated state
        # --- MODIFIED: Update all 3 temp states ---
        return eqx.tree_at(
            lambda m: (m.room_temp, m.wall_temp, m.mass_temp),
            self,
            (next_temp, next_temp, next_temp) # All dummies track room_temp
        )

# --- 3. 2R2C Implementation ---
class ThermalModel_2R2C(AbstractThermalModel):
    """
    2-Resistor, 2-Capacitor model.
    Models room air (C_air) and building envelope (C_wall) separately.
    Heat path: Ambient <-> Wall <-> Air <-> InternalGains
    """
    def __init__(self, config: ThermalConfig, initial_temp: float = 20.0):
        super().__init__(
            room_temp=jnp.array(initial_temp),
            wall_temp=jnp.array(initial_temp), # Real state
            mass_temp=jnp.array(initial_temp)  # Dummy state, tracks room temp
        )

    @eqx.filter_jit
    def step(self,
             storage_discharge_w: Array,
             ac_thermal_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> 'ThermalModel_2R2C':

        # 1. Get model parameters
        C_air = self.config.air_capacity_j_k
        C_wall = self.config.wall_capacity_j_k
        R_wall_amb = self.config.wall_to_ambient_r_k_w
        R_air_wall = self.config.air_to_wall_r_k_w

        T_room = self.room_temp
        T_wall = self.wall_temp
        T_amb = exogenous.ambient_temp

        # 2. Get heat inputs (W)
        net_controllable_w = storage_discharge_w + ac_thermal_w

        # --- UPDATED GAINS ---
        # For 2R2C, all gains go to the air
        net_passive_w = (
            exogenous.occupancy_gains_w
            + exogenous.device_gains_w
            + exogenous.solar_gains_w
        )
        net_internal_gains_w = net_controllable_w + net_passive_w

        # 3. Calculate heat flows (W)
        Q_wall_to_amb = (T_wall - T_amb) / R_wall_amb
        Q_air_to_wall = (T_room - T_wall) / R_air_wall

        # 4. Calculate temperature change for Wall
        # dT_wall/dt = (1/C_wall) * (Q_air_to_wall - Q_wall_to_amb)
        dT_wall_k = ((Q_air_to_wall - Q_wall_to_amb) / C_wall) * dt_seconds
        next_wall_temp = T_wall + dT_wall_k

        # 5. Calculate temperature change for Room Air
        # dT_room/dt = (1/C_air) * (Q_internal_gains - Q_air_to_wall)
        dT_room_k = ((net_internal_gains_w - Q_air_to_wall) / C_air) * dt_seconds
        next_room_temp = T_room + dT_room_k

        # 6. Return new model with updated state
        # --- MODIFIED: Update all 3 temp states ---
        return eqx.tree_at(
            lambda m: (m.room_temp, m.wall_temp, m.mass_temp),
            self,
            (next_room_temp, next_wall_temp, next_room_temp) # mass_temp tracks room_temp
        )

# --- 4. NEW: 3R2C Implementation ---
class ThermalModel_3R2C(AbstractThermalModel):
    """
    3-Resistor, 2-Capacitor model.
    Like 2R2C, but adds a direct resistance from air to ambient (ventilation).
    """
    def __init__(self, config: ThermalConfig, initial_temp: float = 20.0):
        super().__init__(
            room_temp=jnp.array(initial_temp),
            wall_temp=jnp.array(initial_temp), # Real state
            mass_temp=jnp.array(initial_temp)  # Dummy state, tracks room temp
        )

    @eqx.filter_jit
    def step(self,
             storage_discharge_w: Array,
             ac_thermal_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> 'ThermalModel_3R2C':

        # 1. Get model parameters
        C_air = self.config.air_capacity_j_k
        C_wall = self.config.wall_capacity_j_k
        R_wall_amb = self.config.wall_to_ambient_r_k_w
        R_air_wall = self.config.air_to_wall_r_k_w
        R_air_amb = self.config.air_to_ambient_r_k_w # New resistor

        T_room = self.room_temp
        T_wall = self.wall_temp
        T_amb = exogenous.ambient_temp

        # 2. Get heat inputs (W)
        net_controllable_w = storage_discharge_w + ac_thermal_w
        net_passive_w = (
            exogenous.occupancy_gains_w
            + exogenous.device_gains_w
            + exogenous.solar_gains_w
        )
        net_internal_gains_w = net_controllable_w + net_passive_w

        # 3. Calculate heat flows (W)
        Q_wall_to_amb = (T_wall - T_amb) / R_wall_amb
        Q_air_to_wall = (T_room - T_wall) / R_air_wall
        Q_air_to_amb = (T_room - T_amb) / R_air_amb # New heat flow

        # 4. Calculate temperature change for Wall
        # dT_wall/dt = (1/C_wall) * (Q_air_to_wall - Q_wall_to_amb)
        dT_wall_k = ((Q_air_to_wall - Q_wall_to_amb) / C_wall) * dt_seconds
        next_wall_temp = T_wall + dT_wall_k

        # 5. Calculate temperature change for Room Air
        # dT_room/dt = (1/C_air) * (Q_gains - Q_air_to_wall - Q_air_to_amb)
        dT_room_k = ((net_internal_gains_w - Q_air_to_wall - Q_air_to_amb) / C_air) * dt_seconds
        next_room_temp = T_room + dT_room_k

        # 6. Return new model
        return eqx.tree_at(
            lambda m: (m.room_temp, m.wall_temp, m.mass_temp),
            self,
            (next_room_temp, next_wall_temp, next_room_temp)
        )

# --- 5. NEW: 4R3C Implementation ---
class ThermalModel_4R3C(AbstractThermalModel):
    """
    4-Resistor, 3-Capacitor model.
    Adds an internal mass node (C_mass) to the 3R2C model.
    Splits solar gains between air, wall, and internal mass.
    """
    def __init__(self, config: ThermalConfig, initial_temp: float = 20.0):
        super().__init__(
            room_temp=jnp.array(initial_temp),
            wall_temp=jnp.array(initial_temp), # Real state (envelope)
            mass_temp=jnp.array(initial_temp)  # Real state (internal mass)
        )

    @eqx.filter_jit
    def step(self,
             storage_discharge_w: Array,
             ac_thermal_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> 'ThermalModel_4R3C':

        # 1. Get model parameters
        C_air = self.config.air_capacity_j_k
        C_wall = self.config.wall_capacity_j_k  # Envelope
        C_mass = self.config.mass_capacity_j_k  # Internal mass
        
        R_wall_amb = self.config.wall_to_ambient_r_k_w
        R_air_wall = self.config.air_to_wall_r_k_w
        R_air_amb = self.config.air_to_ambient_r_k_w
        R_air_mass = self.config.air_to_mass_r_k_w

        T_room = self.room_temp
        T_wall = self.wall_temp
        T_mass = self.mass_temp
        T_amb = exogenous.ambient_temp

        # 2. Get heat inputs (W)
        net_controllable_w = storage_discharge_w + ac_thermal_w
        
        # Use helper to get split gains
        (air_gains, solar_to_wall, solar_to_mass) = self._get_total_gains(exogenous)
        
        net_air_gains_w = net_controllable_w + air_gains

        # 3. Calculate heat flows (W)
        Q_wall_to_amb = (T_wall - T_amb) / R_wall_amb
        Q_air_to_wall = (T_room - T_wall) / R_air_wall
        Q_air_to_amb = (T_room - T_amb) / R_air_amb
        Q_air_to_mass = (T_room - T_mass) / R_air_mass

        # 4. Calculate temperature change for Wall (Envelope)
        # dT_wall/dt = (1/C_wall) * (Q_air_to_wall - Q_wall_to_amb + Q_solar_to_wall)
        dT_wall_k = ((Q_air_to_wall - Q_wall_to_amb + solar_to_wall) / C_wall) * dt_seconds
        next_wall_temp = T_wall + dT_wall_k

        # 5. Calculate temperature change for Mass (Internal)
        # dT_mass/dt = (1/C_mass) * (Q_air_to_mass + Q_solar_to_mass)
        dT_mass_k = ((Q_air_to_mass + solar_to_mass) / C_mass) * dt_seconds
        next_mass_temp = T_mass + dT_mass_k

        # 6. Calculate temperature change for Room Air
        # dT_room/dt = (1/C_air) * (Q_gains - Q_air_to_wall - Q_air_to_amb - Q_air_to_mass)
        dT_room_k = ((net_air_gains_w - Q_air_to_wall - Q_air_to_amb - Q_air_to_mass) / C_air) * dt_seconds
        next_room_temp = T_room + dT_room_k

        # 7. Return new model
        return eqx.tree_at(
            lambda m: (m.room_temp, m.wall_temp, m.mass_temp),
            self,
            (next_room_temp, next_wall_temp, next_mass_temp)
        )

# --- 6. Passthrough (Dummy) Implementation ---
class PassthroughThermalModel(AbstractThermalModel):
    """
    A dummy model that clamps the room temp to the setpoint.
    """
    def __init__(self, config: ThermalConfig, initial_temp: float = 20.0):
        super().__init__(
            room_temp=jnp.array(initial_temp),
            wall_temp=jnp.array(initial_temp), # Dummy state
            mass_temp=jnp.array(initial_temp)  # Dummy state
        )

    @eqx.filter_jit
    def step(self,
             storage_discharge_w: Array,
             ac_thermal_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> 'PassthroughThermalModel':

        # Room temp is always the setpoint
        next_room_temp = jnp.array(self.config.setpoint)
        # Wall temp just tracks ambient
        next_wall_temp = exogenous.ambient_temp
        # Mass temp tracks room
        next_mass_temp = next_room_temp

        # --- MODIFIED: Update all 3 temp states ---
        return eqx.tree_at(
            lambda m: (m.room_temp, m.wall_temp, m.mass_temp),
            self,
            (next_room_temp, next_wall_temp, next_mass_temp)
        )