# energysim/behavior/shiftable_load.py
import numpy as np
from .base import AbstractBehavioralModel
from ..core.shared.data_structs import SystemState
from typing import Tuple

class StochasticTimeModel(AbstractBehavioralModel):
    """
    A model for a shiftable load that runs once per day.
    - Has a 'daily_prob' of running at all.
    - If it runs, it picks a random start hour within a 'start_window'.
    - Once started, it runs for a fixed 'duration_minutes'.
    """
    def __init__(
        self,
        seed: int,
        power_kw: float,
        duration_minutes: float,
        start_window: Tuple[int, int], # e.g., (18, 22) for 6 PM to 10 PM
        daily_prob: float = 1.0        # Probability of running on any given day
    ):
        super().__init__(seed)
        self.power_w = power_kw * 1000.0
        self.duration_minutes = duration_minutes
        self.start_window = start_window # (min_hour, max_hour)
        self.daily_prob = daily_prob

        # Internal state
        self.is_running = False
        self.run_steps_remaining = 0
        self.last_day = -1
        self.start_hour_today = -1 # Hour it's scheduled to start today

    def reset(self):
        self.is_running = False
        self.run_steps_remaining = 0
        self.last_day = -1
        self.start_hour_today = -1

    def _check_daily_reset(self, current_day: int, dt_seconds: float):
        """Checks if it's a new day and sets up the run for today."""
        if current_day != self.last_day:
            self.last_day = current_day
            self.is_running = False
            self.run_steps_remaining = 0
            
            # Decide if the device will run today
            if self.rng.random() < self.daily_prob:
                # Pick a random start hour
                self.start_hour_today = self.rng.integers(
                    self.start_window[0], self.start_window[1] + 1
                )
                self.total_run_steps = int(
                    (self.duration_minutes * 60) / dt_seconds
                )
            else:
                self.start_hour_today = -1

    def step(self, step_idx: int, dt_seconds: float, state: SystemState) -> float:
        total_seconds = step_idx * dt_seconds
        current_day = total_seconds // 86400
        current_hour = (total_seconds // 3600) % 24

        # 1. Check for daily reset and schedule
        self._check_daily_reset(current_day, dt_seconds)

        # 2. Check if we are currently running
        if self.is_running:
            self.run_steps_remaining -= 1
            if self.run_steps_remaining <= 0:
                self.is_running = False
            return self.power_w

        # 3. Check if we should start
        if (not self.is_running and 
            self.start_hour_today != -1 and 
            current_hour == self.start_hour_today):
            
            self.is_running = True
            self.run_steps_remaining = self.total_run_steps
            self.start_hour_today = -1 # Mark as started
            return self.power_w
            
        return 0.0

    def forecast(self, start_idx: int, horizon: int, dt_seconds: float, state: SystemState) -> np.ndarray:
        """
        A simple, non-stochastic forecast.
        Assumes the device *will* run at the *start* of its window.
        """
        predictions = np.zeros(horizon, dtype=np.float32)
        
        # Calculate run duration in hours
        hours_to_run = self.duration_minutes / 60.0
        start_hour = self.start_window[0]
        end_hour = start_hour + int(np.ceil(hours_to_run))

        for i in range(horizon):
            step_idx = start_idx + i
            total_seconds = step_idx * dt_seconds
            current_hour = (total_seconds // 3600) % 24
            
            # Daily probability is ignored, assumes it runs
            if start_hour <= current_hour < end_hour:
                predictions[i] = self.power_w

        return predictions