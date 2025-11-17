import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Optional

from ..core.models.factory import (
    create_battery, create_thermal, create_heat_pump,
    create_ac, create_storage, create_solar
)
from ..core.models.battery_model import AbstractBatteryModel
from ..core.models.thermal_model import AbstractThermalModel
from ..core.models.heat_pump_model import AbstractHeatPumpModel
from ..core.models.air_conditioner_model import AbstractAirConditionerModel
from ..core.models.thermal_storage_model import AbstractThermalStorage
from ..core.models.solar_model import AbstractSolarModel
from ..core.models.objectives import f_cost_step
from ..core.shared.data_structs import (
    SystemActions, ExogenousData,
    ThermalConfig, BatteryConfig, RewardConfig,
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, SolarConfig,
    SystemState, BatteryState, ThermalState, ThermalStorageState,
    HeatPumpState, AirConditionerState, SolarOutput
)

class JAXSimulator:
    def __init__(
        self,
        dt_seconds: float,
        t_config: ThermalConfig,
        r_config: RewardConfig,
        b_config: Optional[BatteryConfig] = None,
        hp_config: Optional[HeatPumpConfig] = None,
        ac_config: Optional[AirConditionerConfig] = None,
        ts_config: Optional[ThermalStorageConfig] = None,
        s_config: Optional[SolarConfig] = None
    ):
        self.dt_seconds = dt_seconds
        
        # --- UPDATED: Get n_rooms ---
        n_rooms = len(t_config.room_air_indices)
        if n_rooms == 0:
            raise ValueError(
                "ThermalConfig has no 'room_air_indices'. Simulator requires n_rooms > 0."
            )

        # --- 1. Create Models using the Factory ---
        self.initial_battery = create_battery(b_config)
        self.initial_thermal = create_thermal(t_config)
        self.initial_heat_pump = create_heat_pump(hp_config, n_rooms) # <-- Pass n_rooms
        self.initial_ac = create_ac(ac_config, n_rooms)               # <-- Pass n_rooms
        self.initial_storage = create_storage(ts_config)
        self.initial_solar = create_solar(s_config)
        
        # --- 2. Store Configs for Cost Function ---
        self.configs = (
            self.initial_thermal.config, self.initial_battery.config, r_config,
            self.initial_heat_pump.config, self.initial_ac.config,
            self.initial_storage.config, self.initial_solar.config
        )
        
        # --- Mutable State Variables ---
        self._battery: AbstractBatteryModel = self.initial_battery
        self._thermal: AbstractThermalModel = self.initial_thermal
        self._heat_pump: AbstractHeatPumpModel = self.initial_heat_pump
        self._ac: AbstractAirConditionerModel = self.initial_ac
        self._storage: AbstractThermalStorage = self.initial_storage
        self._solar: AbstractSolarModel = self.initial_solar
        
        # --- 3. Pre-bind static arguments for the COST function ---
        self.cost_fn = partial(
            jit(f_cost_step, static_argnames=["dt_seconds"]),
            configs=self.configs,
            dt_seconds=self.dt_seconds
        )
        
        # --- 4. Store active configs (for wrappers) ---
        self.active_configs = {
            "battery": b_config, "heat_pump": hp_config,
            "ac": ac_config, "storage": ts_config,
            "solar": s_config
        }

    @property
    def state(self) -> SystemState:
        return SystemState(
            thermal=ThermalState(
                T_vector=self._thermal.T_vector
            ),
            battery=BatteryState(soc=self._battery.soc, soh=self._battery.soh),
            storage=ThermalStorageState(soc=self._storage.soc),
            heat_pump=HeatPumpState(current_electrical_w=self._heat_pump.current_electrical_w),
            air_conditioner=AirConditionerState(current_electrical_w=self._ac.current_electrical_w)
        )

    def reset(self) -> SystemState:
        """Resets the stateful models and returns the initial state."""
        self._battery = self.initial_battery
        self._thermal = self.initial_thermal
        self._heat_pump = self.initial_heat_pump
        self._ac = self.initial_ac
        self._storage = self.initial_storage
        self._solar = self.initial_solar
        return self.state

    def scan_step(self, actions: SystemActions, exo_data: ExogenousData) -> tuple[SystemState, jnp.ndarray]:
        """
        JIT-compatible step function for use with lax.scan.
        Returns the next state and the cost as a JAX array.
        """
        
        # --- 1. Run stateless models ---
        solar_output = self._solar.calculate(exo_data)
        
        # --- 2. Run HVAC models (now stateful) ---
        next_heat_pump, hp_output = self._heat_pump.step(
            actions.heat_pump_power_w,
            exo_data,
            self.dt_seconds
        )
        next_ac, ac_output = self._ac.step(
            actions.ac_power_w,
            exo_data,
            self.dt_seconds
        )
        
        # --- 3. Run other stateful models ---
        next_battery = self._battery.step(
            actions.battery_power_w, self.dt_seconds
        )
        
        next_storage, storage_output = self._storage.step(
            actions.storage_discharge_w,
            hp_output.thermal_power_w,
            self.dt_seconds
        )
        
        # --- 4. Calculate Cost (using state *before* the step) ---
        # Note: We use self.state (the state *before* this step) for the cost
        cost = self.cost_fn(
            self.state, actions, exo_data,
            hp_output, ac_output, storage_output,
            solar_output
        )
        
        # --- 5. Run final stateful model ---
        heating_w = storage_output.actual_discharge_w
        cooling_w = ac_output.thermal_power_w
        
        next_thermal = self._thermal.step(
            heating_w=heating_w,
            cooling_w=cooling_w,
            exogenous=exo_data,
            dt_seconds=self.dt_seconds
        )
        
        # --- 6. Update mutable state ---
        self._battery = next_battery
        self._thermal = next_thermal
        self._storage = next_storage
        self._heat_pump = next_heat_pump
        self._ac = next_ac
        
        # --- 7. Return new state and JAX array cost ---
        return self.state, cost

    # --- UPDATED: Public-facing Python-loop version ---
    def step(self, actions: SystemActions, exo_data: ExogenousData) -> tuple[SystemState, float]:
        """
        Public-facing step function for Python loops (e.g., Gym Env).
        Returns the next state and the cost as a Python float.
        """
        next_state, cost_array = self.scan_step(actions, exo_data)
        return next_state, float(cost_array)