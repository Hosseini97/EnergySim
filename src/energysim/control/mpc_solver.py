import jax
import jax.numpy as jnp
from jax import jit, grad, lax
from functools import partial
from typing import Optional
import jaxopt

# Import all the models AND the factory
from ..core.models.battery_model import AbstractBatteryModel
from ..core.models.thermal_model import AbstractThermalModel
from ..core.models.heat_pump_model import AbstractHeatPumpModel
from ..core.models.air_conditioner_model import AbstractAirConditionerModel
from ..core.models.thermal_storage_model import AbstractThermalStorage
from ..core.models.solar_model import AbstractSolarModel
from ..core.models.factory import (
    create_battery, create_thermal, create_heat_pump,
    create_ac, create_storage, create_solar
)
from ..core.models.objectives import f_cost_step
from ..core.shared.data_structs import (
    AirConditionerState, HeatPumpState, SystemState, SystemActions, ExogenousData,
    ThermalConfig, BatteryConfig, RewardConfig,
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, SolarConfig,
    ThermalState, BatteryState, ThermalStorageState, SolarOutput
)
import equinox as eqx


class JAX_MPC_Solver:
    """
    A full MPC solver built on JAX, supporting modular components.
    """
    def __init__(
        self,
        N_horizon: int,
        dt_seconds: float,
        # --- Configs ---
        t_config: ThermalConfig,
        r_config: RewardConfig,
        b_config: Optional[BatteryConfig] = None,
        hp_config: Optional[HeatPumpConfig] = None,
        ac_config: Optional[AirConditionerConfig] = None,
        ts_config: Optional[ThermalStorageConfig] = None,
        s_config: Optional[SolarConfig] = None
    ):
        self.N = N_horizon
        self.dt = dt_seconds
        
        # --- UPDATED: Get n_rooms ---
        self.n_rooms = len(t_config.room_air_indices)
        if self.n_rooms == 0:
            raise ValueError(
                "ThermalConfig has no 'room_air_indices'. MPC solver requires n_rooms > 0."
            )

        # --- 1. Create Models using the Factory ---
        self.battery: AbstractBatteryModel = create_battery(b_config)
        self.thermal: AbstractThermalModel = create_thermal(t_config)
        self.heat_pump: AbstractHeatPumpModel = create_heat_pump(hp_config, self.n_rooms)
        self.ac: AbstractAirConditionerModel = create_ac(ac_config, self.n_rooms)         
        self.storage: AbstractThermalStorage = create_storage(ts_config)
        self.solar: AbstractSolarModel = create_solar(s_config)
        
        # --- 2. Store Configs for Cost Function ---
        self.configs = (
            self.thermal.config, self.battery.config, r_config,
            self.heat_pump.config, self.ac.config, self.storage.config,
            self.solar.config
        )
        
        # --- 3. Define the Scan Step Function ---
        @jit
        def mpc_scan_step(state_k: SystemState, inputs_k: tuple[SystemActions, ExogenousData]):
            action_k, exo_k = inputs_k
            
            solar_output_k = self.solar.calculate(exo_k)
            
            # --- Re-hydrate models ---
            battery_k: AbstractBatteryModel = eqx.tree_at(
                lambda m: (m.soc, m.soh),
                self.battery,
                (state_k.battery.soc, state_k.battery.soh)
            )
            thermal_k: AbstractThermalModel = eqx.tree_at(
                lambda m: m.T_vector, self.thermal, state_k.thermal.T_vector
            )
            storage_k: AbstractThermalStorage = eqx.tree_at(
                lambda m: m.soc, self.storage, state_k.storage.soc
            )
            hp_k: AbstractHeatPumpModel = eqx.tree_at(
                lambda m: m.current_electrical_w, self.heat_pump, state_k.heat_pump.current_electrical_w
            )
            ac_k: AbstractAirConditionerModel = eqx.tree_at(
                lambda m: m.current_electrical_w, self.ac, state_k.air_conditioner.current_electrical_w
            )
            
            # --- A. Run HVAC models ---
            next_hp, hp_output = hp_k.step(
                action_k.heat_pump_power_w, exo_k, self.dt
            )
            next_ac, ac_output = ac_k.step(
                action_k.ac_power_w, exo_k, self.dt
            )
            
            # --- B. Run other stateful models ---
            next_battery = battery_k.step(action_k.battery_power_w, self.dt)
            next_storage, storage_output = storage_k.step(
                action_k.storage_discharge_w,
                hp_output.thermal_power_w,
                self.dt
            )
            
            # --- UPDATED: Pass correct thermal inputs to thermal model ---
            heating_w = storage_output.actual_discharge_w
            cooling_w = ac_output.thermal_power_w
            
            next_thermal: AbstractThermalModel = thermal_k.step(
                heating_w,
                cooling_w,
                exo_k,
                self.dt
            )
            
            # --- C. Calculate cost of current step ---
            cost_k = f_cost_step(
                state_k, action_k, exo_k,
                hp_output, ac_output, storage_output,
                solar_output_k,
                self.configs, self.dt
            )
            
            # --- D. Create next state (data-only) ---
            state_k_plus_1 = SystemState(
                thermal=ThermalState(
                    T_vector=next_thermal.T_vector,
                ),
                battery=BatteryState(soc=next_battery.soc, soh=next_battery.soh),
                storage=ThermalStorageState(soc=next_storage.soc),
                heat_pump=HeatPumpState(current_electrical_w=next_hp.current_electrical_w),
                air_conditioner=AirConditionerState(current_electrical_w=next_ac.current_electrical_w)
            )
            
            return state_k_plus_1, cost_k
        
        # --- 4. Define the Total Horizon Cost Function ---
        @jit
        def f_horizon_cost(
            action_sequence: SystemActions, # PyTree of actions [N, ...]
            initial_state: SystemState,     # static
            exo_sequence: ExogenousData     # PyTree of forecasts [N, ...] (static)
        ):
            inputs_over_horizon = (action_sequence, exo_sequence)
            _, cost_sequence = lax.scan(
                mpc_scan_step, initial_state, inputs_over_horizon
            )
            return jnp.sum(cost_sequence)
        
        # --- 5. JIT-compile the cost function and its gradient ---
        self.objective_fn = f_horizon_cost
        
        # --- 6. Setup the Optimizer ---
        b_conf = self.battery.config
        hp_conf = self.heat_pump.config
        ac_conf = self.ac.config
        ts_conf = self.storage.config
        
        # --- UPDATED: Define shapes for scalar vs. zonal actions ---
        scalar_shape = (N_horizon,)
        zonal_shape = (N_horizon, self.n_rooms)

        # --- UPDATED: Use correct shapes for action bounds ---
        self.action_bounds = (
            SystemActions(
                battery_power_w=jnp.full(scalar_shape, -b_conf.max_power_w),
                heat_pump_power_w=jnp.full(zonal_shape, 0.0),
                ac_power_w=jnp.full(zonal_shape, 0.0),
                storage_discharge_w=jnp.full(zonal_shape, 0.0)
            ),
            SystemActions(
                battery_power_w=jnp.full(scalar_shape, b_conf.max_power_w),
                # Assume max power is total, divide it among rooms
                heat_pump_power_w=jnp.full(zonal_shape, hp_conf.max_electrical_power_w / self.n_rooms),
                ac_power_w=jnp.full(zonal_shape, ac_conf.max_electrical_power_w / self.n_rooms),
                storage_discharge_w=jnp.full(zonal_shape, ts_conf.max_discharge_w / self.n_rooms)
            )
        )
        
        self.optimizer = jaxopt.ProjectedGradient(
            fun=self.objective_fn,
            projection=jaxopt.projection.projection_box,
            maxiter=50,
            tol=1e-3,
        )
        
        # --- UPDATED: Create a correctly-shaped warm start PyTree ---
        self.zonal_warm_start = jax.tree.map(
            lambda x: jnp.zeros_like(x), self.action_bounds[0]
        )

    def solve(
        self,
        current_state: SystemState, # This is the data-only PyTree
        exo_forecast: ExogenousData,
        warm_start_actions: SystemActions = None
    ) -> SystemActions:
        
        # --- UPDATED: Use the correct zonal warm start ---
        if warm_start_actions is None:
            warm_start_actions = self.zonal_warm_start
        
        # 2. Run the optimization
        optim_result = self.optimizer.run(
            init_params=warm_start_actions,
            hyperparams_proj=self.action_bounds,
            initial_state=current_state,
            exo_sequence=exo_forecast
        )
        
        optimal_action_sequence = optim_result.params
        
        # 3. Return the *first* action of the optimal sequence
        first_action = jax.tree.map(lambda x: x[0], optimal_action_sequence)
        
        return first_action