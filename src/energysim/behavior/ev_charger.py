# energysim/behavior/ev_charger.py
import numpy as np
from .base import AbstractBehavioralModel
from ..core.shared.data_structs import SystemState

class SimpleEVModel(AbstractBehavioralModel):
    """
    A simple stateful EV model.
    - Arrives home at 6 PM (18:00) with 20% SOC.
    - Plugs in and charges at 7kW.
    - Stops charging when it reaches 80% SOC.
    - Resets its state daily.
    """
    def __init__(
        self,
        seed: int = 42,
        capacity_kwh: float = 60.0,
        charge_rate_kw: float = 7.0,
        arrival_hour: int = 18,
        arrival_soc: float = 0.2,
        target_soc: float = 0.8
    ):
        super().__init__(seed)
        self.capacity_kwh = capacity_kwh
        self.charge_rate_w = charge_rate_kw * 1000.0
        self.arrival_hour = arrival_hour
        self.initial_soc = arrival_soc
        self.target_soc = target_soc
        
        # Initial internal state
        self.soc = self.initial_soc
        self.is_plugged_in = False
        self.last_day = -1 # To track day changes

    def reset(self):
        """Resets the EV to its initial state."""
        self.soc = self.initial_soc
        self.is_plugged_in = False
        self.last_day = -1

    def step(self, step_idx: int, dt_seconds: float, state: SystemState) -> float:
        
        # --- 1. Calculate current time ---
        total_seconds = step_idx * dt_seconds
        current_hour = (total_seconds // 3600) % 24
        current_day = total_seconds // 86400

        # --- 2. Check for daily state reset ---
        if current_day != self.last_day:
            self.soc = self.initial_soc
            self.is_plugged_in = False
            self.last_day = current_day
            
        # --- 3. Check for arrival ---
        if current_hour == self.arrival_hour and not self.is_plugged_in:
            self.is_plugged_in = True
            
        # --- 4. Calculate charging ---
        power_w = 0.0
        if self.is_plugged_in and self.soc < self.target_soc:
            # Set power
            power_w = self.charge_rate_w
            
            # Update internal SOC
            energy_kwh = (power_w * (dt_seconds / 3600.0)) / 1000.0
            delta_soc = energy_kwh / self.capacity_kwh
            self.soc = min(self.target_soc, self.soc + delta_soc)
            
        return power_w
    
    def forecast(self, start_idx: int, horizon: int, dt_seconds: float, state: SystemState) -> np.ndarray:
        """
        Predicts the EV's power draw for the given horizon.
        """
        predictions = np.zeros(horizon, dtype=np.float32)
        
        # This is a simple, stateless forecast based on time.
        # A more complex one might use the model's current state (self.soc, self.is_plugged_in)
        # but that is much harder. A time-based forecast is a great start.
        
        for i in range(horizon):
            step_idx = start_idx + i
            total_seconds = step_idx * dt_seconds
            current_hour = (total_seconds // 3600) % 24
            
            # Simple logic: will charge at 7kW between 6 PM and (e.g.) 10 PM
            # This logic should mirror your `step` method's logic.
            # For this simple model, we assume it arrives at 18:00 (arrival_hour)
            # and needs (e.g.) 3 hours to charge.
            # (target_soc - arrival_soc) * capacity_kwh / charge_rate_kw
            
            charge_needed_kwh = (self.target_soc - self.initial_soc) * self.capacity_kwh
            hours_to_charge = charge_needed_kwh / (self.charge_rate_w / 1000.0)
            end_hour = self.arrival_hour + int(np.ceil(hours_to_charge))
            
            if self.arrival_hour <= current_hour < end_hour:
                predictions[i] = self.charge_rate_w
        
        return predictions