import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from functools import partial
from typing import Tuple, List, Dict

from ..core.shared.data_structs import (
    ThermalConfig, SystemActions, ExogenousData, SystemState
)
from ..core.models.thermal_model import RCNetworkModel
from ..sim.simulator import JAXSimulator

# =============================================================================
# 1. Learnable Parameters Wrapper
# =============================================================================

class LearnableParams(eqx.Module):
    """
    Holds learnable physics parameters in log-space to ensure positivity.
    Includes envelope (R, C), initial states, and HVAC efficiency.
    """
    # Envelope Physics
    log_C: jnp.ndarray
    log_G_edges: jnp.ndarray     # Log Conductance (1/R) for existing edges
    
    # Initial Condition correction
    # We assume we know T_air, but T_wall (hidden) is unknown.
    # This learns the offset: T_hidden = T_air_initial + offset
    hidden_state_offsets: jnp.ndarray

    # System Efficiency Correction
    # Allows the solver to distinguish between "leaky walls" and "weak heat pump"
    log_hvac_efficiency: jnp.ndarray 

    # Static masks to preserve topology
    adjacency_mask: jnp.ndarray = eqx.field(static=True)
    n_nodes: int = eqx.field(static=True)
    ambient_idx: int = eqx.field(static=True)

    def __init__(self, initial_config: ThermalConfig, n_nodes: int):
        """
        Initializes learnable params based on the template config.
        """
        self.n_nodes = n_nodes
        self.ambient_idx = initial_config.ambient_air_index

        # 1. Extract Capacitances (C)
        # Handle infinite capacity (ambient) by masking
        C_vector = jnp.where(
            initial_config.C_inv_vector > 1e-9, 
            1.0 / initial_config.C_inv_vector, 
            1e9 
        )
        self.log_C = jnp.log(C_vector)

        # 2. Extract Conductances (G = 1/R) from A_matrix
        A = initial_config.A_matrix
        # Mask off-diagonal non-zeros
        self.adjacency_mask = (jnp.abs(A) > 1e-9) & (1 - jnp.eye(n_nodes))
        
        G_initial = jnp.abs(A) * self.adjacency_mask + 1e-9
        self.log_G_edges = jnp.log(G_initial)

        # 3. Initialize States & Efficiency
        self.hidden_state_offsets = jnp.zeros(n_nodes) 
        self.log_hvac_efficiency = jnp.array(0.0) # exp(0) = 1.0 (100% of nominal)

    def get_config(self, template_config: ThermalConfig) -> ThermalConfig:
        """Reconstructs a valid ThermalConfig from learned params."""
        # 1. Reconstruct C
        C_est = jnp.exp(self.log_C)
        C_inv_est = jnp.where(C_est > 1e8, 0.0, 1.0 / C_est)

        # 2. Reconstruct G (Symmetric & Masked)
        G_dense = jnp.exp(self.log_G_edges)
        G_sym = (G_dense + G_dense.T) / 2.0
        G_final = G_sym * self.adjacency_mask

        # 3. Construct Laplacian A Matrix (Conservation of Energy)
        # A_ii = -sum(G_ij_leaving)
        diag_sum = jnp.sum(G_final, axis=1)
        A_matrix = G_final - jnp.diag(diag_sum)

        return eqx.tree_at(
            lambda c: (c.A_matrix, c.C_inv_vector),
            template_config,
            (A_matrix, C_inv_est)
        )
    
    def get_initial_temp_vector(self, true_t0_air: jnp.ndarray, obs_indices: jnp.ndarray) -> jnp.ndarray:
        """
        Constructs the full T vector at t=0.
        T_observed = Ground Truth
        T_hidden = Ground Truth (mean) + Learned Offset
        """
        # Start with mean air temp for everyone
        mean_air = jnp.mean(true_t0_air)
        T0 = jnp.full((self.n_nodes,), mean_air)
        
        # Add learned offsets to hidden nodes
        T0 = T0 + self.hidden_state_offsets
        
        # Hard override observed nodes with exact truth
        T0 = T0.at[obs_indices].set(true_t0_air)
        
        # Hard override ambient (if it's part of state vector)
        # Note: The thermal model usually overrides ambient index at every step anyway.
        return T0

    def get_efficiency_factor(self) -> jnp.ndarray:
        return jnp.exp(self.log_hvac_efficiency)


# =============================================================================
# 2. Differentiable Simulation Kernel
# =============================================================================

def simulation_loss_fn(
    learnable_params: LearnableParams, 
    # Static Template Data
    template_sim: JAXSimulator,
    # Dynamic Training Data
    true_temperatures: jnp.ndarray,   # (Time, N_observed)
    actions_seq: SystemActions,       # (Time, ...)
    exo_seq: ExogenousData,           # (Time, ...)
):
    """
    Fully differentiable rollout of HVAC + Thermal physics.
    """
    dt = template_sim.dt_seconds
    config = template_sim.thermal.config
    obs_indices = jnp.array(config.room_air_indices)

    # 1. Reconstruct Physics
    calibrated_t_config = learnable_params.get_config(config)
    T0 = learnable_params.get_initial_temp_vector(true_temperatures[0], obs_indices)
    hvac_eff = learnable_params.get_efficiency_factor()

    # 2. Initialize Models
    # We utilize the functional structure of the JAXSimulator models
    thermal_model = RCNetworkModel(calibrated_t_config, T0)
    
    # We rely on the template's HVAC models (HP, AC, Storage) for their logic,
    # but we will scale their thermal output by the learned efficiency.
    init_carry = (
        thermal_model,
        template_sim.heat_pump,
        template_sim.ac,
        template_sim.storage,
        template_sim.battery # Included for completeness, though usually decoupled
    )

    # 3. Define the Scan Step (The "Kernel")
    def step_fn(carry, inputs):
        (th_k, hp_k, ac_k, st_k, bat_k) = carry
        act_k, exo_k = inputs

        # A. Step HVAC Models
        hp_next, hp_out = hp_k.step(act_k.heat_pump_power_w, exo_k, dt)
        ac_next, ac_out = ac_k.step(act_k.ac_power_w, exo_k, dt)
        st_next, st_out = st_k.step(act_k.storage_discharge_w, hp_out.thermal_power_w, dt)
        bat_next = bat_k.step(act_k.battery_power_w, dt)

        # B. Calculate Thermal Inputs & Apply Learned Efficiency
        # We assume the learned efficiency applies to active heating/cooling generation
        # scaling Q_thermal relative to W_electric.
        heating_w = st_out.actual_discharge_w * hvac_eff
        cooling_w = ac_out.thermal_power_w * hvac_eff
        
        # Waste heat is usually dissipative, assume 1.0 or scale if needed.
        # Here we leave waste unscaled as it's a secondary effect.
        waste_w = st_out.standing_loss_w + jnp.sum(st_out.rejected_heat_w)

        # C. Step Thermal Model (The "Reality Check")
        th_next = th_k.step(heating_w, cooling_w, waste_w, exo_k, dt)

        new_carry = (th_next, hp_next, ac_next, st_next, bat_next)
        
        # We return the temperature vector for loss calculation
        return new_carry, th_next.T_vector

    # 4. Rollout (Scan)
    # We rely on JAX to unpack the SystemActions/ExogenousData pytrees automatically
    _, predicted_T_seq = jax.lax.scan(
        step_fn, 
        init_carry, 
        (actions_seq, exo_seq)
    )

    # 5. Compute Loss
    # Filter predictions to only the rooms we have sensors for
    pred_observed = predicted_T_seq[:, obs_indices]
    
    # MSE Loss
    mse = jnp.mean((pred_observed - true_temperatures)**2)
    
    # Regularization (optional but recommended)
    # Penalize extreme efficiency deviations (e.g., keep it between 0.5 and 1.5)
    reg_eff = (learnable_params.log_hvac_efficiency)**2 * 0.1
    
    return mse + reg_eff

# =============================================================================
# 3. Training Step
# =============================================================================

@eqx.filter_jit
def train_step(
    learnable_params: LearnableParams,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    # Static / Constant Data
    template_sim: JAXSimulator,
    true_temperatures: jnp.ndarray,
    actions_seq: SystemActions,
    exo_seq: ExogenousData,
):
    loss, grads = eqx.filter_value_and_grad(simulation_loss_fn)(
        learnable_params,
        template_sim,
        true_temperatures,
        actions_seq,
        exo_seq
    )
    
    updates, new_opt_state = optimizer.update(grads, opt_state, learnable_params)
    new_params = eqx.apply_updates(learnable_params, updates)
    
    return new_params, new_opt_state, loss

# =============================================================================
# 4. High-Level Interface
# =============================================================================

class SystemIdentifier:
    def __init__(self, simulator: JAXSimulator):
        self.sim = simulator
        self.n_nodes = simulator.thermal.config.C_inv_vector.shape[0]
        
    def calibrate(
        self, 
        true_room_temps: jnp.ndarray, # (Time, N_rooms)
        actions: SystemActions,       # (Time, ...)
        exo_data: ExogenousData,      # (Time, ...)
        learning_rate: float = 0.05,
        steps: int = 500
    ) -> Tuple[ThermalConfig, Dict]:
        """
        Calibrates the physics model to match real-world data.
        """
        print(f"--- Starting System Identification (Horizon: {true_room_temps.shape[0]}) ---")
        
        # 1. Setup Parameters
        params = LearnableParams(self.sim.thermal.config, self.n_nodes)
        
        # 2. Setup Optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        
        # 3. Training Loop
        loss_history = []
        
        # Ensure data is on the right device
        true_room_temps = jax.device_put(true_room_temps)
        actions = jax.device_put(actions)
        exo_data = jax.device_put(exo_data)

        for i in range(steps):
            params, opt_state, loss_val = train_step(
                params, opt_state, optimizer,
                self.sim,
                true_room_temps,
                actions,
                exo_data
            )
            loss_history.append(float(loss_val))
            
            if i % 100 == 0:
                eff = jnp.exp(params.log_hvac_efficiency)
                print(f"Step {i}: Loss={loss_val:.4f} | Est. HVAC Eff={eff:.3f}")

        # 4. Final Extraction
        final_config = params.get_config(self.sim.thermal.config)
        final_eff = float(jnp.exp(params.log_hvac_efficiency))
        
        print(f"Done. Final MSE: {loss_history[-1]:.5f}")
        print(f"Calibrated HVAC Efficiency Factor: {final_eff:.3f}")

        stats = {
            "loss_history": loss_history,
            "hvac_efficiency_factor": final_eff,
            "final_offsets": params.hidden_state_offsets
        }
        
        return final_config, stats