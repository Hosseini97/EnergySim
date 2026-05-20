import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
from typing import Tuple, Optional, Dict

from ..core.models.factory import (
    create_battery, create_thermal, create_heat_pump,
    create_ac, create_storage, create_pv
)
from ..core.models.battery_model import AbstractBatteryModel
from ..core.models.thermal_model import AbstractThermalModel
from ..core.models.heat_pump_model import AbstractHeatPumpModel
from ..core.models.air_conditioner_model import AbstractAirConditionerModel
from ..core.models.thermal_storage_model import AbstractThermalStorage
from ..core.models.pv_model import AbstractPVModel

from ..core.shared.data_structs import (
    SystemActions, ExogenousData, SystemState,
    ThermalConfig, BatteryConfig, RewardConfig,
    HeatPumpConfig, AirConditionerConfig, ThermalStorageConfig, PVConfig,
    BatteryState, ThermalState, ThermalStorageState, 
    HeatPumpState, AirConditionerState, SystemOutputs
)

class JAXSimulator(eqx.Module):
    """
    A purely functional Simulator. 
    Contains both the configuration (static) and the current state (dynamic models).
    """
    # --- Sub-Models (These hold the state + config) ---
    battery: AbstractBatteryModel
    thermal: AbstractThermalModel
    heat_pump: AbstractHeatPumpModel
    ac: AbstractAirConditionerModel
    storage: AbstractThermalStorage
    pv: AbstractPVModel

    # --- Configs & Constants ---
    configs: Tuple
    dt_seconds: float = eqx.field(static=True)
    initial_battery_soc: float = eqx.field(static=True)

    def __init__(
        self,
        dt_seconds: float,
        t_config: ThermalConfig,
        r_config: RewardConfig,
        b_config: Optional[BatteryConfig] = None,
        hp_config: Optional[HeatPumpConfig] = None,
        ac_config: Optional[AirConditionerConfig] = None,
        ts_config: Optional[ThermalStorageConfig] = None,
        pv_config: Optional[PVConfig] = None,
        initial_battery_soc: float = 0.5,
    ):
        self.dt_seconds = dt_seconds
        self.initial_battery_soc = initial_battery_soc

        # 1. Create Models (Initial State is created here)
        n_rooms = len(t_config.room_air_indices)
        
        self.battery = create_battery(b_config, initial_soc=initial_battery_soc)
        self.thermal = create_thermal(t_config)
        self.heat_pump = create_heat_pump(hp_config, n_rooms)
        self.ac = create_ac(ac_config, n_rooms)
        self.storage = create_storage(ts_config)
        self.pv = create_pv(pv_config)

        # 2. Store Configs tuple for cost function
        self.configs = (
            self.thermal.config, self.battery.config, r_config,
            self.heat_pump.config, self.ac.config,
            self.storage.config, self.pv.config
        )

    @property
    def state(self) -> SystemState:
        """Extracts a Data-Only snapshot of the current system."""
        return SystemState(
            thermal=ThermalState(T_vector=self.thermal.T_vector),
            battery=BatteryState(soc=self.battery.soc, soh=self.battery.soh),
            storage=ThermalStorageState(temperatures_c=self.storage.temperatures_c),
            heat_pump=HeatPumpState(current_electrical_w=self.heat_pump.current_electrical_w, 
                                    current_thermal_w=self.heat_pump.current_thermal_w),
            air_conditioner=AirConditionerState(current_electrical_w=self.ac.current_electrical_w,
                                                 current_thermal_w=self.ac.current_thermal_w)
        )

    @jax.jit
    def step(
        self, 
        actions: SystemActions, 
        exo: ExogenousData
    ) -> Tuple['JAXSimulator', SystemOutputs]:
        """
        Functional Step: Returns (New_Simulator, Outputs)
        State is updated automatically within the new sub-models.
        """
        
        # 1. Run Sub-Models
        pv_out = self.pv.calculate(exo)
        
        next_hp, hp_out = self.heat_pump.step(actions.heat_pump_power_w, exo, self.dt_seconds)
        next_ac, ac_out = self.ac.step(actions.ac_power_w, exo, self.dt_seconds)
        
        next_battery = self.battery.step(actions.battery_power_w, self.dt_seconds)
        
        next_storage, storage_out = self.storage.step(
            actions.storage_discharge_w, 
            hp_out.thermal_power_w, 
            self.dt_seconds
        )

        heating_w = storage_out.actual_discharge_w
        cooling_w = ac_out.thermal_power_w
        # Sum waste heat sources:
        # - Storage standing losses
        # - Storage rejected heat (e.g. charging when full)
        # - Optionally HVAC electrical losses (Input - Output) could be added here if modeled
        total_waste_heat_w = storage_out.standing_loss_w + jnp.sum(storage_out.rejected_heat_w)

        # 3. Thermal Step
        next_thermal = self.thermal.step(
            heating_w, 
            cooling_w, 
            total_waste_heat_w,
            exo, 
            self.dt_seconds
        )

        outputs = SystemOutputs(
            pv=pv_out,
            hp=hp_out,
            ac=ac_out,
            storage=storage_out,
            total_waste_heat_w=total_waste_heat_w
        )
        
        new_sim = eqx.tree_at(
            lambda s: (s.battery, s.thermal, s.heat_pump, s.ac, s.storage),
            self,
            (next_battery, next_thermal, next_hp, next_ac, next_storage)
        )

        return new_sim, outputs

    def reset(self) -> tuple['JAXSimulator']:
        """
        Re-initializes the simulator to default starting values.
        (Actually, just re-runs __init__ logic or re-loads initial models)
        """
        return JAXSimulator(
            self.dt_seconds, self.thermal.config, self.configs[2], # RewardConfig
            self.battery.config, self.heat_pump.config, self.ac.config,
            self.storage.config, self.pv.config, self.initial_battery_soc
        )
