import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Any

@dataclass
class HEMSConfig:
    """Configuration for the real-world dataset environment."""
    max_steps: int = 24  # 24-hour daily cycle
    target_temp_c: float = 22.0
    lambda_comfort: float = 2.0
    battery_capacity_kwh: float = 13.5 # Standard residential battery size
    
    # Placeholders for the real data we will hook up later
    weather_data_path: str = "data/weather.csv"
    pricing_data_path: str = "data/pricing.csv"

class HEMSMultiDeviceEnv(gym.Env):
    """
    Multi-Device HEMS Environment complying with UnifiedEnvProtocol.
    """
    def __init__(self, config: HEMSConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        # Phase 1 Implementation: 7-Dimensional State Space
        # [Hour, Price, Temp_out, Irradiance, SoC, Temp_in, Base_Load]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        # Phase 1 Implementation: 20 Joint Actions (5 battery * 4 heat pump)
        self.action_space = spaces.Discrete(20)
        
        # Required by CustomEnvProtocol
        self.allowed_actions = list(range(20))
        
        self.current_step = 0
        self.state = np.zeros(7, dtype=np.float32)

    def set_action_space(self, allowed_action_indices: List[int]) -> None:
        """Allows AACL to dynamically restrict available hardware topologies."""
        self.allowed_actions = allowed_action_indices

    def get_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Converts observation to tensor for the PyTorch agents."""
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize with dummy values until we hook up the CSV dataset
        # [Hour=0, Price=0.15, T_out=15.0, Irr=0.0, SoC=0.5, T_in=20.0, Load=1.0]
        self.state = np.array([0.0, 0.15, 15.0, 0.0, 0.5, 20.0, 1.0], dtype=np.float32)
        return self.state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Phase 4 Implementation: Physics and Reward calculation will go here
        
        self.current_step += 1
        terminated = self.current_step >= self.config.max_steps
        truncated = False
        
        dummy_reward = 0.0 
        
        return self.state, dummy_reward, terminated, truncated, {}
        
    # --- Dummy implementations for remaining CustomEnvProtocol requirements ---
    def get_params(self) -> Any: 
        return self.config
        
    def get_optimal_value_function_plot(self, V_data: np.ndarray) -> Any: 
        pass
        
    def get_agent_value_function_plot(self, value_net: torch.nn.Module) -> Any: 
        pass