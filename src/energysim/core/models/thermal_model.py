import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import ThermalConfig, ExogenousData, Array, SystemActions

class AbstractThermalModel(eqx.Module):
    T_vector: Array
    config: ThermalConfig

    @eqx.filter_jit
    def step(self, heating_w: Array, cooling_w: Array, waste_heat_w: float, exogenous: ExogenousData, dt_seconds: float) -> 'AbstractThermalModel':
        raise NotImplementedError

class RCNetworkModel(AbstractThermalModel):
    """
    Solving C * dT/dt = A*T + B*U + Q_inf + Q_waste
    """
    def __init__(self, config: ThermalConfig, initial_T_vector: Array):
        super().__init__(
            T_vector=initial_T_vector,
            config=config
        )

    @eqx.filter_jit
    def _build_input_vector(self, heating_w: Array, cooling_w: Array, exo: ExogenousData) -> Array:
        inputs_flat = jnp.concatenate([
            heating_w,
            cooling_w,
            exo.solar_gains_w,
            exo.occupancy_gains_w,
            exo.device_gains_w
        ])
        U_vector = self.config.B_matrix @ inputs_flat
        return U_vector

    @eqx.filter_jit
    def _calculate_dynamic_infiltration(self, T_k: Array, T_amb: float, wind_speed: float) -> Array:
        """
        Calculates heat flow due to air infiltration/ventilation.
        Q_inf = G_inf * (T_amb - T_room)
        where G_inf is dynamic based on wind and temp diff.
        """
        if not self.config.use_dynamic_infiltration:
            return jnp.zeros_like(T_k)

        # Get average room temperature for stack effect calculation
        room_temps = T_k[jnp.array(self.config.room_air_indices)]
        avg_room_temp = jnp.mean(room_temps)
        delta_T = jnp.abs(T_amb - avg_room_temp)

        # Calculate Air Changes Per Hour (ACH)
        # ACH = K1 + K2*|dT| + K3*Wind
        ach = self.config.inf_k1 + (self.config.inf_k2 * delta_T) + (self.config.inf_k3 * wind_speed)
        
        # Convert ACH to mass flow rate approx or Conductance (G)
        # G = (ACH * Vol * AirDensity * SpecificHeat) / 3600
        # Approx: rho*Cp ~= 1200 J/(m3 K)
        conductance = (ach * self.config.room_vol_m3 * 1200.0) / 3600.0
        
        # We assume infiltration affects all air nodes proportionally or split evenly
        # Simple approach: split conductance among all room nodes
        n_rooms = len(self.config.room_air_indices)
        g_per_room = conductance / n_rooms

        # Build the heat flow vector
        Q_inf_vector = jnp.zeros_like(T_k)
        
        # Calculate flow for each room: Q = G * (T_amb - T_room)
        q_rooms = g_per_room * (T_amb - room_temps)
        
        # Apply to indices
        Q_inf_vector = Q_inf_vector.at[jnp.array(self.config.room_air_indices)].set(q_rooms)
        
        return Q_inf_vector

    @eqx.filter_jit
    def step(self, heating_w: Array, cooling_w: Array, waste_heat_w: float, exogenous: ExogenousData, dt_seconds: float) -> 'RCNetworkModel':
        
        T_k = self.T_vector
        
        # 1. Set Ambient Node
        T_k = T_k.at[self.config.ambient_air_index].set(exogenous.ambient_temp)

        # 2. Base Dynamics: A*T + B*U
        U_vector = self._build_input_vector(heating_w, cooling_w, exogenous)
        A_T = self.config.A_matrix @ T_k
        
        # 3. Dynamic Infiltration
        Q_inf = self._calculate_dynamic_infiltration(T_k, exogenous.ambient_temp, exogenous.wind_speed_m_s)

        # 4. Waste Heat Coupling (from Storage/HVAC)
        Q_waste = jnp.zeros_like(T_k)
        # Only apply if a valid node index is set in config
        valid_node = self.config.waste_heat_node_index >= 0
        waste_node_idx = jnp.where(valid_node, self.config.waste_heat_node_index, 0) # Default to 0 if invalid to avoid index error
        # If valid, add heat; if not, add 0
        added_heat = jnp.where(valid_node, waste_heat_w, 0.0)
        Q_waste = Q_waste.at[waste_node_idx].add(added_heat)

        # 5. Integration
        # dT/dt = C_inv * (Sum of Heat Flows)
        total_heat_flow = A_T + U_vector + Q_inf + Q_waste
        dT_dt_vector = self.config.C_inv_vector * total_heat_flow

        # Zero out derivative for ambient node (infinite capacity)
        dT_dt_vector = dT_dt_vector.at[self.config.ambient_air_index].set(0.0)

        T_k_plus_1 = T_k + dT_dt_vector * dt_seconds

        return eqx.tree_at(lambda m: m.T_vector, self, T_k_plus_1)