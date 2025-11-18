import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import ThermalStorageConfig, ThermalStorageOutput, Array

class AbstractThermalStorage(eqx.Module):
    soc: Array
    config: ThermalStorageConfig = eqx.field(static=True)

    @eqx.filter_jit
    def step(self, action_discharge_w: Array, hvac_charge_w: Array, dt_seconds: float) -> tuple['AbstractThermalStorage', ThermalStorageOutput]:
        raise NotImplementedError

class ThermalStorageModel(AbstractThermalStorage):
    """The full, stateful thermal storage (water tank) model."""
    def __init__(self, config: ThermalStorageConfig, initial_soc: float = 0.5):
        super().__init__(soc=jnp.array(initial_soc), config=config)

    @eqx.filter_jit
    def step(self, action_discharge_w: Array, hvac_charge_w: Array, dt_seconds: float) -> tuple["ThermalStorageModel", ThermalStorageOutput]:
        
        # 1. Aggregate Zonal Actions
        total_charge_request_w = jnp.sum(hvac_charge_w)
        total_discharge_request_w = jnp.sum(action_discharge_w)

        capacity_j = self.config.capacity_j
        current_energy_j = self.soc * capacity_j 

        # 2. Handle Charging
        max_charge_j = (1.0 - self.soc) * capacity_j
        max_charge_w = max_charge_j / dt_seconds
        actual_charge_w = jnp.clip(total_charge_request_w, 0.0, self.config.max_charge_w)
        actual_charge_w = jnp.clip(actual_charge_w, 0.0, max_charge_w)
        d_energy_charge_j = actual_charge_w * dt_seconds
        rejected_heat_w = jnp.fmax(0.0, total_charge_request_w - actual_charge_w)

        # 3. Handle Discharging
        max_discharge_j = current_energy_j
        max_discharge_w = max_discharge_j / dt_seconds
        actual_discharge_w = jnp.clip(total_discharge_request_w, 0.0, self.config.max_discharge_w)
        actual_discharge_w = jnp.clip(actual_discharge_w, 0.0, max_discharge_w)
        d_energy_discharge_j = actual_discharge_w * dt_seconds

        # 4. Handle Standing Losses (Scalar)
        # This heat is lost from the tank, but potentially gained by the room
        loss_w = self.soc * self.config.standing_loss_w_per_soc
        d_energy_loss_j = loss_w * dt_seconds

        # 5. State Update
        next_energy_j = current_energy_j + d_energy_charge_j - d_energy_discharge_j - d_energy_loss_j
        next_soc = jnp.where(capacity_j > 0, next_energy_j / capacity_j, 0.0)
        next_soc = jnp.clip(next_soc, 0.0, 1.0)

        # 6. Distribute Outputs
        total_request_safe = jnp.fmax(1e-6, total_discharge_request_w)
        discharge_fraction = action_discharge_w / total_request_safe
        actual_discharge_zonal_w = actual_discharge_w * discharge_fraction

        total_charge_safe = jnp.fmax(1e-6, total_charge_request_w)
        charge_fraction = hvac_charge_w / total_charge_safe
        rejected_heat_zonal_w = rejected_heat_w * charge_fraction

        # Output now includes standing_loss_w (scalar) to be applied to a zone
        output = ThermalStorageOutput(
            actual_discharge_w=actual_discharge_zonal_w,
            rejected_heat_w=rejected_heat_zonal_w,
            standing_loss_w=loss_w 
        )

        new_model = eqx.tree_at(lambda m: m.soc, self, next_soc)
        return new_model, output

class ThermalStoragePassthrough(AbstractThermalStorage):
    def __init__(self, dummy_config: ThermalStorageConfig):
        super().__init__(soc=jnp.array(0.0), config=dummy_config)

    @eqx.filter_jit
    def step(self, action_discharge_w: Array, hvac_charge_w: Array, dt_seconds: float) -> tuple['ThermalStoragePassthrough', ThermalStorageOutput]:
        output = ThermalStorageOutput(
            actual_discharge_w=hvac_charge_w,
            rejected_heat_w=0.0,
            standing_loss_w=0.0 # No losses in passthrough
        )
        return self, output