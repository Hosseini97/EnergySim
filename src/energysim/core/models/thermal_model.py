# energysim/core/models/thermal_model.py
import jax.numpy as jnp
import equinox as eqx
from ..shared.data_structs import ThermalConfig, ExogenousData, Array, SystemActions

class AbstractThermalModel(eqx.Module):
    """Abstract base class, now holds the T_vector."""
    # The state MUST be a PyTree-compatible field
    T_vector: Array 
    config: ThermalConfig # thermal config has Array matrices, so not static
    
    @eqx.filter_jit
    def step(self,
             actions: SystemActions,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> 'AbstractThermalModel':
        raise NotImplementedError

class RCNetworkModel(AbstractThermalModel):
    """
    A single, powerful thermal model that solves the
    matrix equation C * dT/dt = A*T + B*U
    """
    def __init__(self, config: ThermalConfig, initial_T_vector: Array):
        super().__init__(
            T_vector=initial_T_vector,
            config=config
        )

    @eqx.filter_jit
    def _build_input_vector(self,
                            heating_w: Array,
                            cooling_w: Array,
                            exo: ExogenousData
                        ) -> Array:
        """
        Gathers all heat inputs and maps them to the correct nodes
        using the B_matrix.
        """
        # 1. Flatten all (N_rooms,) inputs into a single vector
        # The order MUST match how the B_matrix was built.
        inputs_flat = jnp.concatenate([
            heating_w,
            cooling_w,
            exo.solar_gains_w,
            exo.occupancy_gains_w,
            exo.device_gains_w
        ])

        # 2. Use matrix multiplication to map inputs to nodes
        U_vector = self.config.B_matrix @ inputs_flat

        return U_vector

    @eqx.filter_jit
    def step(self,
             heating_w: Array,
             cooling_w: Array,
             exogenous: ExogenousData,
             dt_seconds: float
    ) -> 'RCNetworkModel':
        
        # 1. Get current state vector
        T_k = self.T_vector
        
        # --- Handle the Ambient Node ---
        # The ambient temperature is a special "input"
        # We must force the ambient node's temperature to the exogenous value.
        T_k = T_k.at[self.config.ambient_air_index].set(exogenous.ambient_temp)
        
        # 2. Build the input vector U
        # Note: We've combined B and U here.
        U_vector = self._build_input_vector(heating_w, cooling_w, exogenous)

        # 3. Calculate: A * T
        A_T = self.config.A_matrix @ T_k
        
        # 4. Calculate: dT/dt = C_inv * (A*T + U)
        dT_dt_vector = self.config.C_inv_vector * (A_T + U_vector)
        
        # --- Zero out derivative for ambient node ---
        # The ambient node's temp is *set*, not simulated.
        dT_dt_vector = dT_dt_vector.at[self.config.ambient_air_index].set(0.0)

        # 5. Integrate (Euler step)
        T_k_plus_1 = T_k + dT_dt_vector * dt_seconds
        
        # 6. Return new model with updated state
        return eqx.tree_at(lambda m: m.T_vector, self, T_k_plus_1)
