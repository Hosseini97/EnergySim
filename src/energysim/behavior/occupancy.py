# energysim/behavior/occupancy.py
import numpy as np
from .base import AbstractBehavioralModel
from ..core.shared.data_structs import SystemState

class StochasticOccupancyModel(AbstractBehavioralModel):
    """
    Simulates household occupancy and corresponding metabolic heat gains.
    Each occupant independently decides to be home based on an hourly probability profile.
    """
    def __init__(
        self,
        seed: int,
        num_occupants: int = 2,
        heat_per_person_w: float = 100.0,
        hourly_prob_home: np.ndarray = None
    ):
        super().__init__(seed)
        self.num_occupants = num_occupants
        self.heat_per_person_w = heat_per_person_w
        
        # Default to a standard "working household" profile if none provided
        if hourly_prob_home is None:
            # 0-6: Sleeping (1.0), 7-8: Waking/Leaving, 9-16: Work (0.1), 17-23: Home
            self.hourly_prob_home = np.array([
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,      # 00:00 - 06:00
                0.8, 0.4,                               # 07:00 - 08:00 (leaving)
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, # 09:00 - 16:00 (work)
                0.5, 0.8, 0.9, 0.9, 0.9, 1.0, 1.0       # 17:00 - 23:00 (return & sleep)
            ])
        else:
            self.hourly_prob_home = hourly_prob_home
            
        self.current_occupants = num_occupants
        self.last_step_hour = -1

    def reset(self):
        self.current_occupants = self.num_occupants
        self.last_step_hour = -1

    def step(self, step_idx: int, dt_seconds: float, state: SystemState) -> float:
        total_seconds = step_idx * dt_seconds
        current_hour = int((total_seconds // 3600) % 24)
        
        # Only roll the dice when the hour changes to prevent jitter
        if current_hour != self.last_step_hour:
            self.last_step_hour = current_hour
            prob = self.hourly_prob_home[current_hour]
            
            # Roll a binomial distribution: n trials (occupants), p probability of success
            self.current_occupants = self.rng.binomial(n=self.num_occupants, p=prob)
            
        return float(self.current_occupants * self.heat_per_person_w)

    def forecast(self, start_idx: int, horizon: int, dt_seconds: float, state: SystemState) -> np.ndarray:
        """
        Returns the expected value (E[X] = n * p) of metabolic heat gain for the MPC horizon.
        """
        predictions = np.zeros(horizon, dtype=np.float32)
        for i in range(horizon):
            step_time = (start_idx + i) * dt_seconds
            hour = int((step_time // 3600) % 24)
            
            expected_occupants = self.num_occupants * self.hourly_prob_home[hour]
            predictions[i] = expected_occupants * self.heat_per_person_w
            
        return predictions