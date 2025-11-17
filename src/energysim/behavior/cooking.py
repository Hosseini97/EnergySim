# energysim/behavior/cooking.py
import numpy as np
from .base import AbstractBehavioralModel
from ..core.shared.data_structs import SystemState
from typing import List, Tuple

class StochasticImpulseModel(AbstractBehavioralModel):
    """
    A model for "peaky" loads that occur in specific windows (e.g., cooking).
    - Runs for 'duration_minutes' at 'power_kw'.
    - Can be triggered during multiple 'time_windows' (e.g., breakfast, lunch, dinner).
    - Has a 'prob_per_window' of triggering *once* at the start of each window.
    """
    def __init__(
        self,
        seed: int,
        power_kw: float,
        duration_minutes: float,
        time_windows: List[Tuple[int, int]], # e.g., [(7, 9), (12, 14), (18, 21)]
        prob_per_window: float = 0.8        # Prob of an event in each window
    ):
        super().__init__(seed)
        self.power_w = power_kw * 1000.0
        self.duration_minutes = duration_minutes
        self.time_windows = time_windows
        self.prob_per_window = prob_per_window

        # Internal state
        self.is_running = False
        self.run_steps_remaining = 0
        self.last_day = -1
        # Tracks which windows have been checked today to prevent re-triggering
        self.window_checked_today = {start: False for start, end in time_windows}

    def reset(self):
        self.is_running = False
        self.run_steps_remaining = 0
        self.last_day = -1
        self.window_checked_today = {start: False for start, end in self.time_windows}

    def _check_daily_reset(self, current_day: int):
        """Resets the daily window checks."""
        if current_day != self.last_day:
            self.last_day = current_day
            self.window_checked_today = {start: False for start, end in self.time_windows}

    def step(self, step_idx: int, dt_seconds: float, state: SystemState) -> float:
        total_seconds = step_idx * dt_seconds
        current_day = total_seconds // 86400
        current_hour = (total_seconds // 3600) % 24

        # 1. Check for daily reset
        self._check_daily_reset(current_day)

        # 2. Check if we are currently running
        if self.is_running:
            self.run_steps_remaining -= 1
            if self.run_steps_remaining <= 0:
                self.is_running = False
            return self.power_w
            
        # 3. Check if we should start a new run
        for start_hour, end_hour in self.time_windows:
            # Check if we are at the *start* of a window that hasn't been checked
            if current_hour == start_hour and not self.window_checked_today[start_hour]:
                
                self.window_checked_today[start_hour] = True # Mark as checked
                
                # Roll the dice for this window
                if self.rng.random() < self.prob_per_window:
                    self.is_running = True
                    self.run_steps_remaining = int(
                        (self.duration_minutes * 60) / dt_seconds
                    )
                    return self.power_w
            
        return 0.0

    def forecast(self, start_idx: int, horizon: int, dt_seconds: float, state: SystemState) -> np.ndarray:
        """
        Forecasting this is complex. A naive forecast of 0.0 is safest.
        The MPC will treat this as a reactive, uncontrollable load.
        """
        return np.zeros(horizon)