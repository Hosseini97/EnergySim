# energysim/behavior/base.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from ..core.shared.data_structs import SystemState

class AbstractBehavioralModel(ABC):
    """
    Abstract base class for a stateful, non-JAX behavioral model.
    These models can use standard Python logic, randomness, and if/else.
    """
    def __init__(self, seed: Optional[int] = None):
        """
        Initializes the model, optionally with a random seed.
        """
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def step(self, step_idx: int, dt_seconds: float, state: SystemState) -> float:
        """
        Returns the power consumption (W) for the current step.
        
        Args:
            step_idx: The current step index of the simulation.
            dt_seconds: The duration of a single simulation step.
            state: The current SystemState (e.g., to check battery SOC).
            
        Returns:
            The electrical power (W) consumed by this device.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Resets the internal state of the model."""
        raise NotImplementedError

    def forecast(self, start_idx: int, horizon: int, dt_seconds: float, state: SystemState) -> np.ndarray:
        """
        Returns a *predicted* power profile (W) for the MPC horizon.
        
        By default, a naive forecast assumes no future usage.
        More complex models can override this.
        """
        return np.zeros(horizon)