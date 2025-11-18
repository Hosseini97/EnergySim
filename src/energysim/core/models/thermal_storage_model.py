import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import ThermalStorageConfig, ThermalStorageOutput, Array

class AbstractThermalStorage(eqx.Module):
    temperatures_c: Array
    config: ThermalStorageConfig = eqx.field(static=True)

    @property
    def soc(self):
        # Simple mapping of mean temp to 0-1 for external observers
        # Assuming useful range 30C -> 60C
        return jnp.clip((jnp.mean(self.temperatures_c) - 30.0) / 30.0, 0.0, 1.0)

    @eqx.filter_jit
    def step(self, action_discharge_w: Array, hvac_charge_w: Array, dt_seconds: float) -> tuple['AbstractThermalStorage', ThermalStorageOutput]:
        raise NotImplementedError

class StratifiedThermalStorageModel(AbstractThermalStorage):
    """
    N-Node Stratified Water Tank.
    Node 0 = Top (Hot), Node N-1 = Bottom (Cold).
    """
    def __init__(self, config: ThermalStorageConfig, initial_temp_c: float = 45.0):
        super().__init__(
            temperatures_c=jnp.full((config.n_nodes,), initial_temp_c),
            config=config
        )

    @eqx.filter_jit
    def step(self, action_discharge_w: Array, hvac_charge_w: Array, dt_seconds: float) -> tuple["StratifiedThermalStorageModel", ThermalStorageOutput]:
        
        # Constants
        WATER_HEAT_CAPACITY = 4186.0 * 1000.0 # J/(m^3 K) approx volumetric heat capacity
        
        # Geometry
        node_vol_m3 = self.config.volume_m3 / self.config.n_nodes
        node_mass_capacity_j_k = node_vol_m3 * WATER_HEAT_CAPACITY
        node_height = self.config.height_m / self.config.n_nodes
        cross_section_area = self.config.volume_m3 / self.config.height_m

        # 1. Aggregate inputs (Zonal -> Scalar for the tank)
        total_charge_w = jnp.sum(hvac_charge_w) # Heat Pump input
        total_discharge_w = jnp.sum(action_discharge_w) # Load extraction

        # Clip inputs to hardware limits
        actual_charge_w = jnp.clip(total_charge_w, 0.0, self.config.max_charge_w)
        rejected_charge_w = total_charge_w - actual_charge_w
        
        actual_discharge_w = jnp.clip(total_discharge_w, 0.0, self.config.max_discharge_w)

        # 2. Calculate Heat Fluxes per Node (Watts)
        # Initialize flux vector
        Q_net = jnp.zeros(self.config.n_nodes)

        # A. Active Charging (Source) -> Injected into Top Node (0)
        Q_net = Q_net.at[0].add(actual_charge_w)

        # B. Active Discharging (Load) -> Extracted from Top Node (0)
        # Note: Real physical tanks replace top water with bottom cold water. 
        # Simplified: We extract energy from top. If top is empty (cold), we extract less 
        # (but here we assume idealized control extracts requested power).
        Q_net = Q_net.at[0].add(-actual_discharge_w)

        # C. Vertical Conduction (Heat transfer between adjacent nodes)
        # Q_cond = k * A * (T_next - T_curr) / dx
        k_eff = self.config.vertical_conductivity_w_mk
        conductance = (k_eff * cross_section_area) / node_height
        
        T = self.temperatures_c
        # Flux from i+1 to i (Bottom to Top)
        # We shift T arrays to calc diffs
        T_up = jnp.roll(T, -1) # T[i] becomes T[i+1]
        # Ignore the wrap-around at the boundary (last node) using mask
        valid_boundary = jnp.ones(self.config.n_nodes)
        valid_boundary = valid_boundary.at[-1].set(0.0) # No conduction from below bottom node
        
        flux_up = (T_up - T) * conductance * valid_boundary
        
        # Apply Fluxes
        # Node i gains flux from i+1
        Q_net = Q_net + flux_up
        # Node i+1 loses flux to i (shift flux_up back down)
        Q_net = Q_net - jnp.roll(flux_up, 1).at[0].set(0.0)

        # D. Losses to Ambient
        # Q_loss = U * (T_node - T_amb)
        # Assuming loss_coeff in config is per node W/K
        loss_flux = (T - self.config.ambient_temp_c) * self.config.loss_coeff_w_k
        Q_net = Q_net - loss_flux

        # 3. Update Temperatures (Euler integration)
        dT = (Q_net * dt_seconds) / node_mass_capacity_j_k
        new_temperatures = T + dT
        
        # Physical constraints (water doesn't freeze or boil easily in this simplified model, 
        # but clipping 10C to 90C keeps numerics sane)
        new_temperatures = jnp.clip(new_temperatures, 10.0, 90.0)

        # 4. Distribute outputs back to Zones
        # We need to report per-zone values for the simulator/cost function
        total_req_safe = jnp.where(total_discharge_w > 1e-6, total_discharge_w, 1.0)
        discharge_fractions = action_discharge_w / total_req_safe
        zonal_discharge = discharge_fractions * actual_discharge_w
        
        total_chg_safe = jnp.where(total_charge_w > 1e-6, total_charge_w, 1.0)
        zonal_rejected = (hvac_charge_w / total_chg_safe) * rejected_charge_w
        
        # Total losses (scalar)
        total_loss_w = jnp.sum(loss_flux)

        output = ThermalStorageOutput(
            actual_discharge_w=zonal_discharge,
            rejected_heat_w=zonal_rejected,
            standing_loss_w=total_loss_w
        )

        new_model = eqx.tree_at(lambda m: m.temperatures_c, self, new_temperatures)
        return new_model, output

class ThermalStoragePassthrough(AbstractThermalStorage):
    def __init__(self, dummy_config: ThermalStorageConfig):
        # Dummy 1-node
        super().__init__(temperatures_c=jnp.array([20.0]), config=dummy_config)

    @eqx.filter_jit
    def step(self, action_discharge_w: Array, hvac_charge_w: Array, dt_seconds: float) -> tuple['ThermalStoragePassthrough', ThermalStorageOutput]:
        return self, ThermalStorageOutput(hvac_charge_w, jnp.zeros_like(hvac_charge_w), 0.0)