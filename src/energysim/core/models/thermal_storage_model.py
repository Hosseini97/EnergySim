import jax.numpy as jnp
import jax
import equinox as eqx
from ..shared.data_structs import ThermalStorageConfig, ThermalStorageOutput, Array, GridThermalStorageConfig

class AbstractThermalStorage(eqx.Module):
    temperatures_c: Array
    config: ThermalStorageConfig 

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
    

class GridThermalStorageModel(AbstractThermalStorage):
    """
    3D Finite Volume Thermal Storage Model.
    Simulates diffusion, boundary losses, and buoyancy-driven mixing.
    """
    config: GridThermalStorageConfig
    
    def __init__(self, config: GridThermalStorageConfig, initial_temp_c: float = 45.0):
        self.config = config
        # Initialize 3D temperature grid
        self.temperatures_c = jnp.full(config.grid_shape, initial_temp_c)

    @eqx.filter_jit
    def step(self, action_discharge_w: Array, hvac_charge_w: Array, dt_seconds: float) -> tuple['GridThermalStorageModel', ThermalStorageOutput]:
        
        # --- 1. Physical Constants & Geometry ---
        WATER_RHO_CP = 4186.0 * 1000.0 # J/(m^3 K)
        voxel_vol = self.config.voxel_volume_m3
        voxel_mass_cap = voxel_vol * WATER_RHO_CP
        
        # Grid Dimensions
        Nz, Ny, Nx = self.config.grid_shape
        dz = self.config.voxel_height_m
        # Assuming uniform aspect ratio for Y/X relative to volume
        total_area = self.config.total_volume_m3 / self.config.height_m
        dx = dy = jnp.sqrt(total_area / (Ny * Nx)) # Approximation for cubic voxels horizontal
        
        # Surface Areas for conduction
        A_z = dx * dy # Top/Bottom face area
        A_y = dx * dz # Front/Back face area
        A_x = dy * dz # Left/Right face area

        # --- 2. Process Inputs (Source Terms) ---
        total_chg = jnp.clip(jnp.sum(hvac_charge_w), 0.0, self.config.max_charge_w)
        total_dis = jnp.clip(jnp.sum(action_discharge_w), 0.0, self.config.max_discharge_w)
        
        # Initialize Power Input Map (Watts per voxel)
        Q_source = jnp.zeros(self.config.grid_shape)
        
        # Apply Source (Heat Pump) to specific voxel
        cz, cy, cx = self.config.charge_inlet_idx
        Q_source = Q_source.at[cz, cy, cx].add(total_chg)
        
        # Apply Sink (Load) to specific voxel
        dz_out, dy_out, dx_out = self.config.discharge_outlet_idx
        Q_source = Q_source.at[dz_out, dy_out, dx_out].add(-total_dis)
        
        # --- 3. Compute Conduction (Diffusion) ---
        # We compute fluxes across faces.
        # T = (Nz, Ny, Nx)
        
        T = self.temperatures_c
        
        # Helper to compute flux between shifted arrays
        def compute_flux(T_curr, axis, area, dist):
            """Returns flux ENTERING T_curr from T_curr-shifted-along-axis."""
            T_neighbor = jnp.roll(T_curr, shift=1, axis=axis)
            
            # Gradient check for Buoyancy (Only on Z-axis, axis=0)
            k = self.config.thermal_conductivity_w_mk
            if axis == 0:
                # T_neighbor is "Above" T_curr (since z=0 is top).
                # Wait, usually Z indexing: 0=Top, 1=Below.
                # roll(shift=1) means T[1] gets value from T[0].
                # So T_neighbor is the voxel ABOVE T_curr.
                # If T_curr (below) > T_neighbor (above), we have instability (Hot below Cold).
                is_unstable = T_curr > T_neighbor
                k = jnp.where(is_unstable, self.config.convection_conductivity_w_mk, k)

            conductance = (k * area) / dist
            flux = (T_neighbor - T_curr) * conductance
            
            # Boundary Mask (Don't conduct wrap-around)
            # For roll(1), index 0 gets wrapped from index -1. We must mask index 0.
            mask = jnp.ones_like(T_curr)
            # Dynamically set slice for the specific axis
            mask = jax.lax.dynamic_update_slice(
                mask, 
                jnp.zeros_like(mask, shape=tuple(1 if i == axis else s for i, s in enumerate(mask.shape))),
                tuple(0 for _ in mask.shape)
            )
            return flux * mask

        # Z-Flux (Vertical)
        flux_from_top = compute_flux(T, 0, A_z, dz)
        flux_to_bottom = jnp.roll(flux_from_top, -1, axis=0) 
        # Note: roll(-1) moves flux at i to i-1. The flux LEAVING i to go down is the flux ENTERING i+1 from top.
        # Correct energy balance: dE = Flux_In - Flux_Out
        # dE_z = Flux_from_top - Flux_leaving_bottom
        # But Flux_leaving_bottom is just Flux_from_top calculated for the cell below.
        # Actually, let's be explicit:
        
        # Flux[i] = k * (T[i-1] - T[i]). This is heat from Above entering i.
        # Heat leaving i to Below is Flux[i+1].
        flux_z_in = flux_from_top
        flux_z_out = jnp.roll(flux_from_top, -1, axis=0)
        # Fix boundary for out: The bottom-most cell (index -1) flux out is 0 (insulated bottom)
        flux_z_out = flux_z_out.at[-1, :, :].set(0.0)

        # Y-Flux
        flux_y_in = compute_flux(T, 1, A_y, dy)
        flux_y_out = jnp.roll(flux_y_in, -1, axis=1).at[:, -1, :].set(0.0)
        
        # X-Flux
        flux_x_in = compute_flux(T, 2, A_x, dx)
        flux_x_out = jnp.roll(flux_x_in, -1, axis=2).at[:, :, -1].set(0.0)

        Q_conduction = (flux_z_in - flux_z_out) + \
                       (flux_y_in - flux_y_out) + \
                       (flux_x_in - flux_x_out)

        # --- 4. Boundary Losses ---
        # Simplified: Every voxel loses to ambient proportional to its external surface area.
        # Interior voxels have 0 external area.
        # This is tricky to vectorize perfectly without constructing an explicit area map.
        # Approx: Apply volumetric loss factor to ALL cells (assuming not huge tank).
        # Better: Construct a "Loss Mask" in __init__ or assume constant loss.
        # Let's use a Volumetric Loss approx for speed: Q_loss = U_vol * (T - T_amb)
        # U_vol derived from Surface Area / Volume.
        loss_w = (T - self.config.ambient_temp_c) * (self.config.loss_coeff_to_ambient_w_m2k * (total_area*6 / (Nz*Ny*Nx))) # Rough approx
        
        # --- 5. Integration ---
        Q_net = Q_source + Q_conduction - loss_w
        dT = (Q_net * dt_seconds) / voxel_mass_cap
        
        new_temps = T + dT
        new_temps = jnp.clip(new_temps, 10.0, 95.0) # Safety clamp

        # --- 6. Outputs ---
        # Rejected heat calculation
        total_req = jnp.where(total_chg > 1e-6, total_chg, 1.0)
        rejected = (hvac_charge_w / total_req) * (jnp.sum(hvac_charge_w) - total_chg)
        
        output = ThermalStorageOutput(
            actual_discharge_w=action_discharge_w, # Assuming simplistic ideal extraction for now
            rejected_heat_w=rejected,
            standing_loss_w=jnp.sum(loss_w)
        )
        
        return eqx.tree_at(lambda m: m.temperatures_c, self, new_temps), output