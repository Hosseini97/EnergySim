import numpy as np
from .base import AbstractBehavioralModel
from ..core.shared.data_structs import SystemState

class ProfiledThermostaticLoad(AbstractBehavioralModel):
    """
    A water heater that simulates stochastic behavior during stepping,
    but provides a weighted probability profile during forecasting.
    
    This allows the MPC to see 'expected' demand and pre-charge storage.
    """
    def __init__(
        self,
        seed: int,
        power_kw: float,
        capacity_kwh: float,
        setpoint_soc: float = 0.9,
        deadband_soc: float = 0.2,
        daily_usage_kwh: float = 4.0
    ):
        super().__init__(seed)
        self.power_w = power_kw * 1000.0
        self.capacity_kwh = capacity_kwh
        self.setpoint_soc = setpoint_soc
        self.turn_on_soc = setpoint_soc - deadband_soc
        self.daily_usage_kwh = daily_usage_kwh

        # Internal state
        self.soc = self.setpoint_soc
        self.is_heating = False

        # Define a "Double Hump" daily profile (Morning 7-9, Evening 18-21)
        # Normalized to sum to 1.0
        self.hourly_profile = np.array([
            0.01, 0.01, 0.01, 0.01, 0.02, 0.05, # 00-06
            0.15, 0.15, 0.08, 0.04, 0.03, 0.03, # 06-12
            0.03, 0.03, 0.03, 0.04, 0.05, 0.08, # 12-18
            0.10, 0.10, 0.08, 0.05, 0.03, 0.02  # 18-24
        ])
        self.hourly_profile /= np.sum(self.hourly_profile)

    def reset(self):
        self.soc = self.setpoint_soc
        self.is_heating = False

    def step(self, step_idx: int, dt_seconds: float, state: SystemState) -> float:
        # 1. Determine Current Time
        total_seconds = step_idx * dt_seconds
        current_hour = int((total_seconds // 3600) % 24)

        # 2. Stochastic Demand Generation
        # We convert the profile probability into a probability of a "draw event"
        # Draw size varies, but average must equal profile[h] * daily_usage
        
        base_prob = self.hourly_profile[current_hour]
        # Scaling factor to make events somewhat sparse but impactful
        # If prob is 0.10, we want roughly that much energy gone in this hour.
        
        # Simplified stochastic logic:
        # Probability of a draw this step = (Hourly_Energy / Avg_Draw_Size) / Steps_Per_Hour
        avg_draw_size_kwh = 0.5 # e.g. a 5 min shower
        steps_per_hour = 3600 / dt_seconds
        
        energy_needed_this_hour = self.hourly_profile[current_hour] * self.daily_usage_kwh
        prob_draw = (energy_needed_this_hour / avg_draw_size_kwh) / steps_per_hour
        
        # Clamp prob
        prob_draw = np.clip(prob_draw, 0.0, 1.0)

        if self.rng.random() < prob_draw:
            # Apply Demand
            draw_kwh = self.rng.normal(avg_draw_size_kwh, avg_draw_size_kwh * 0.2)
            self.soc -= max(0, draw_kwh) / self.capacity_kwh

        # 3. Thermostatic Control (Deadband)
        power_w = 0.0
        if self.is_heating:
            if self.soc >= self.setpoint_soc:
                self.is_heating = False
            else:
                power_w = self.power_w
        elif self.soc < self.turn_on_soc:
            self.is_heating = True
            power_w = self.power_w

        # 4. Apply Heating Energy
        energy_added_kwh = (power_w * dt_seconds / 3600.0) / 1000.0
        self.soc = np.clip(self.soc + (energy_added_kwh / self.capacity_kwh), 0.0, 1.0)

        return power_w

    def forecast(self, start_idx: int, horizon: int, dt_seconds: float, state: SystemState) -> np.ndarray:
        """
        Top-Tier Feature: Returns the *Expected Value* of the load.
        The MPC uses this to plan battery charging or pre-cooling/heating.
        """
        predictions = np.zeros(horizon, dtype=np.float32)
        
        for i in range(horizon):
            current_step = start_idx + i
            total_seconds = current_step * dt_seconds
            current_hour = int((total_seconds // 3600) % 24)
            
            # Expected Energy Consumption (kWh) for this specific step
            # = (Total_Daily_kWh * Profile_Fraction_Hour) / Steps_Per_Hour
            steps_per_hour = 3600 / dt_seconds
            expected_energy_kwh = (self.daily_usage_kwh * self.hourly_profile[current_hour]) / steps_per_hour
            
            # Convert Energy (kWh) -> Power (kW) -> Power (W)
            # Power (kW) = Energy (kWh) / Time (h)
            time_h = dt_seconds / 3600.0
            expected_power_kw = expected_energy_kwh / time_h
            
            predictions[i] = expected_power_kw * 1000.0

        return predictions