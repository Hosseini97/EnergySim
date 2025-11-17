# energysim/behavior/water_heater.py
import numpy as np
from .base import AbstractBehavioralModel
from ..core.shared.data_structs import SystemState

class ThermostaticLoadModel(AbstractBehavioralModel):
    """
    A model for a thermostatic load like a water heater.
    - Has an internal SOC (state of charge) representing stored heat.
    - Has standing losses.
    - Has stochastic demand (showers, sinks) that remove SOC.
    - Turns on a heating element (deadband control) when SOC is low.
    """
    def __init__(
        self,
        seed: int,
        power_kw: float,
        capacity_kwh: float,
        setpoint_soc: float = 0.9,
        deadband_soc: float = 0.2,
        standing_loss_rate_per_hr: float = 0.01, # 1% of capacity per hour
        demand_prob_per_step: float = 0.01,
        demand_kwh_mean: float = 2.0 # e.g., a 2 kWh shower
    ):
        super().__init__(seed)
        self.power_w = power_kw * 1000.0
        self.capacity_kwh = capacity_kwh
        self.setpoint_soc = setpoint_soc
        self.turn_on_soc = setpoint_soc - deadband_soc
        
        self.standing_loss_kwh_per_sec = (
            self.capacity_kwh * standing_loss_rate_per_hr
        ) / 3600.0
        
        self.demand_prob_per_step = demand_prob_per_step
        self.demand_kwh_mean = demand_kwh_mean
        
        # Internal state
        self.soc = self.setpoint_soc
        self.is_heating = False

    def reset(self):
        self.soc = self.setpoint_soc
        self.is_heating = False

    def step(self, step_idx: int, dt_seconds: float, state: SystemState) -> float:
        
        # 1. Apply standing losses
        loss_kwh = self.standing_loss_kwh_per_sec * dt_seconds
        self.soc -= loss_kwh / self.capacity_kwh

        # 2. Apply stochastic demand
        if self.rng.random() < self.demand_prob_per_step:
            # Add noise to demand size
            demand_kwh = self.rng.normal(self.demand_kwh_mean, 0.2 * self.demand_kwh_mean)
            self.soc -= max(0, demand_kwh) / self.capacity_kwh

        # 3. Heating element logic
        power_w = 0.0
        if self.is_heating:
            if self.soc >= self.setpoint_soc:
                self.is_heating = False
                power_w = 0.0
            else:
                power_w = self.power_w
        elif self.soc < self.turn_on_soc:
            self.is_heating = True
            power_w = self.power_w
            
        # 4. Apply energy from heating
        energy_kwh = (power_w * (dt_seconds / 3600.0)) / 1000.0
        self.soc += energy_kwh / self.capacity_kwh
        
        # 5. Clip SOC and return power
        self.soc = np.clip(self.soc, 0.0, 1.0)
        return power_w

    def forecast(self, start_idx: int, horizon: int, dt_seconds: float, state: SystemState) -> np.ndarray:
        """
        Forecasting a thermostatic load is very complex.
        A naive forecast of 0.0 is the safest, as it's an
        uncontrollable load the MPC must react to.
        """
        return np.zeros(horizon)