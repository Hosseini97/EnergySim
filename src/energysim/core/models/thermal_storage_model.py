# energysim/core/models/thermal_storage_model.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import ThermalStorageConfig, ThermalStorageOutput, Array

# --- 1. Define the Abstract Base Class ---
# This defines the "contract" for all thermal storage types.
class AbstractThermalStorage(eqx.Module):
    """Abstract base class for all thermal storage models."""
    soc: Array  # All must have an SOC to keep the PyTree shape consistent
    config: ThermalStorageConfig = eqx.field(static=True) # All must have a config

    @eqx.filter_jit
    def step(self, 
             action_discharge_w: Array, 
             hvac_charge_w: Array, 
             dt_seconds: float
    ) -> tuple['AbstractThermalStorage', ThermalStorageOutput]:
        """The abstract step function."""
        raise NotImplementedError


# --- 2. Implement the REAL Storage Model ---
class ThermalStorageModel(AbstractThermalStorage):
    """The full, stateful thermal storage (water tank) model."""
    
    def __init__(self, config: ThermalStorageConfig, initial_soc: float = 0.5):
        super().__init__(soc=jnp.array(initial_soc), config=config)


    @eqx.filter_jit
    def step(self,
            action_discharge_w: Array,  # Zonal request, e.g., shape (2,)
            hvac_charge_w: Array,       # Zonal input, e.g., shape (2,)
            dt_seconds: float
    ) -> tuple["ThermalStorageModel", ThermalStorageOutput]:
        """
        Updates the tank's single SOC based on aggregated zonal actions.
        """
        # --- 1. AGGREGATE ZONAL ACTIONS to scalar totals ---
        total_charge_request_w = jnp.sum(hvac_charge_w)
        total_discharge_request_w = jnp.sum(action_discharge_w)

        capacity_j = self.config.capacity_j
        current_energy_j = self.soc * capacity_j  # self.soc is scalar

        # --- 2. Handle Charging (all calculations are scalar) ---
        max_charge_j = (1.0 - self.soc) * capacity_j
        max_charge_w = max_charge_j / dt_seconds

        actual_charge_w = jnp.clip(total_charge_request_w, 0.0, self.config.max_charge_w)
        actual_charge_w = jnp.clip(actual_charge_w, 0.0, max_charge_w)
        d_energy_charge_j = actual_charge_w * dt_seconds

        rejected_heat_w = jnp.fmax(0.0, total_charge_request_w - actual_charge_w)

        # --- 3. Handle Discharging (all calculations are scalar) ---
        max_discharge_j = current_energy_j
        max_discharge_w = max_discharge_j / dt_seconds

        actual_discharge_w = jnp.clip(total_discharge_request_w, 0.0, self.config.max_discharge_w)
        actual_discharge_w = jnp.clip(actual_discharge_w, 0.0, max_discharge_w)
        d_energy_discharge_j = actual_discharge_w * dt_seconds

        # --- 4. Handle Standing Losses (Scalar) ---
        loss_w = self.soc * self.config.standing_loss_w_per_soc
        d_energy_loss_j = loss_w * dt_seconds

        # --- 5. State Update (Scalar) ---
        next_energy_j = current_energy_j + d_energy_charge_j - d_energy_discharge_j - d_energy_loss_j
        next_soc = jnp.where(capacity_j > 0, next_energy_j / capacity_j, 0.0)
        next_soc = jnp.clip(next_soc, 0.0, 1.0)

        # --- 6. DISTRIBUTE OUTPUTS back to zonal vectors ---
        total_request_safe = jnp.fmax(1e-6, total_discharge_request_w)
        discharge_fraction = action_discharge_w / total_request_safe
        actual_discharge_zonal_w = actual_discharge_w * discharge_fraction

        total_charge_safe = jnp.fmax(1e-6, total_charge_request_w)
        charge_fraction = hvac_charge_w / total_charge_safe
        rejected_heat_zonal_w = rejected_heat_w * charge_fraction

        output = ThermalStorageOutput(
            actual_discharge_w=actual_discharge_zonal_w,
            rejected_heat_w=rejected_heat_zonal_w
        )

        new_model = eqx.tree_at(lambda m: m.soc, self, next_soc)

        return new_model, output


# --- 3. Implement the "Passthrough" (Bypass) Model ---
class ThermalStoragePassthrough(AbstractThermalStorage):
    """
    A dummy storage model that provides a direct "bypass" from the
    heat pump to the thermal model. It has no state and no losses.
    """
    
    def __init__(self, dummy_config: ThermalStorageConfig):
        # Has a dummy SOC of 0.0 and holds the dummy config.
        # This ensures it has the *exact same PyTree structure* as the real model.
        super().__init__(soc=jnp.array(0.0), config=dummy_config)

    @eqx.filter_jit
    def step(self, 
             action_discharge_w: Array, 
             hvac_charge_w: Array, 
             dt_seconds: float
    ) -> tuple['ThermalStoragePassthrough', ThermalStorageOutput]:
        """
        A dummy step that passes heat straight through.
        - `action_discharge_w` is IGNORED (there's no tank to discharge).
        - `hvac_charge_w` is passed *directly* to the output.
        """
        
        # 1. Heat from HVAC passes directly to the room.
        actual_discharge_w = hvac_charge_w
        
        # 2. No heat is rejected (it all passes through).
        rejected_heat_w = 0.0
        
        # 3. State is unchanged (self.soc remains 0.0).
        new_model = self
        
        # 4. Create output struct.
        output = ThermalStorageOutput(
            actual_discharge_w=actual_discharge_w,
            rejected_heat_w=rejected_heat_w
        )
        
        return new_model, output