import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Any
from data_loader import RealWorldDataLoader
import matplotlib.pyplot as plt

@dataclass
class HEMSConfig:
    """Configuration for the real-world dataset environment."""
    max_steps: int = 96  # 24-hour daily cycle
    target_temp_c: float = 22.0
    lambda_comfort: float = 2.0
    battery_capacity_kwh: float = 13.5 # Standard residential battery size
    data_path: str = "data/sample_data_with_weather.csv" # Path to your final CSV


class HEMSMultiDeviceEnv(gym.Env):
    """
    Multi-Device HEMS Environment complying with UnifiedEnvProtocol.
    """
    def __init__(self, config: HEMSConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        # Load the real dataset
        self.data_loader = RealWorldDataLoader(config.data_path, config.max_steps)
        self.current_episode_data = None
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(20)
        self.allowed_actions = list(range(20))
        self.current_step = 0
        self.state = np.zeros(7, dtype=np.float32)

    def set_action_space(self, allowed_action_indices: List[int]) -> None:
        """Allows AACL to dynamically restrict available hardware topologies."""
        self.allowed_actions = allowed_action_indices

    def get_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Converts observation to tensor for the PyTorch agents."""
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        
        # Fetch a new random 24-hour period for this episode
        self.current_episode_data = self.data_loader.sample_episode()
        
        # Initial state setup
        hour = self.current_episode_data['hour'][0]
        price = self.current_episode_data['price'][0]
        t_out = self.current_episode_data['t_out'][0]
        irr = self.current_episode_data['irr'][0]
        load = self.current_episode_data['load'][0]
        
        # Start at 50% SoC and Target Temperature
        self.state = np.array([hour, price, t_out, irr, 0.5, self.config.target_temp_c, load], dtype=np.float32)
        return self.state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # --- 1. DECODE JOINT ACTION ---
        # 20 actions total: 5 battery states * 4 HP states
        batt_idx = action // 4
        hp_idx = action % 4
        
        # Action mappings (kW)
        batt_action_kw = [-7.0, -3.5, 0.0, 3.5, 7.0][batt_idx]
        hp_action_kw = [0.0, 1.0, 2.0, 3.0][hp_idx]
        
        # Unpack current state
        hour, price, t_out, irr, soc, t_in, base_load = self.state
        
        # --- 2. PHYSICS SIMULATION ---
        
        # A. Battery Dynamics
        dt_hours = 0.25 
        energy_req_kwh = batt_action_kw * dt_hours
        
        # ADD THIS LINE: Default to the requested energy (which handles the 0.0 case)
        actual_batt_kwh = energy_req_kwh 
        
        if energy_req_kwh > 0: # Charging
            max_charge_kwh = (1.0 - soc) * self.config.battery_capacity_kwh
            actual_batt_kwh = min(energy_req_kwh, max_charge_kwh)
        elif energy_req_kwh < 0: # Discharging
            max_discharge_kwh = soc * self.config.battery_capacity_kwh
            actual_batt_kwh = -min(abs(energy_req_kwh), max_discharge_kwh)
            
        new_soc = soc + (actual_batt_kwh / self.config.battery_capacity_kwh)
        actual_batt_kw = actual_batt_kwh / dt_hours # Convert back to power for grid calculation
        
        # B. Heat Pump & Thermal Dynamics
        # Coefficient of Performance (COP) drops as outdoor temperature drops
        cop = max(1.0, 4.5 + 0.1 * (t_out - 5.0)) 
        heat_provided_kw = hp_action_kw * cop
        
        # Thermal model: T_new = T_old - heat_loss + heat_gained
        # alpha = insulation loss rate, beta = heating capacity factor
        alpha, beta = 0.05, 0.8
        heat_loss = alpha * (t_in - t_out)
        new_t_in = t_in - heat_loss + (beta * heat_provided_kw)
        
        # C. Solar PV Generation
        # Simple approximation: 1000 W/m2 irradiance yields max 5kW solar output
        pv_kw = self.current_episode_data['pv'][self.current_step]
        
        # --- 3. REWARD CALCULATION ---
        
        # Economic Cost
        net_grid_kw = base_load + hp_action_kw + actual_batt_kw - pv_kw
        r_cost = -(net_grid_kw * price)
        
        # Comfort Penalty (Squared error from target)
        r_comfort = -self.config.lambda_comfort * ((new_t_in - self.config.target_temp_c) ** 2)
        
        # Total Reward
        reward = r_cost + r_comfort
        
        # --- 4. STATE UPDATE ---
        self.current_step += 1
        terminated = self.current_step >= self.config.max_steps
        truncated = False
        
        if not terminated:
            new_hour = self.current_episode_data['hour'][self.current_step]
            new_price = self.current_episode_data['price'][self.current_step]
            new_t_out = self.current_episode_data['t_out'][self.current_step]
            new_irr = self.current_episode_data['irr'][self.current_step]
            new_base_load = self.current_episode_data['load'][self.current_step]
        else:
            # The episode is over. Re-use the current external conditions 
            # for the final state observation to avoid an IndexError.
            new_hour = hour
            new_price = price
            new_t_out = t_out
            new_irr = irr
            new_base_load = base_load
            
        self.state = np.array([
            new_hour, new_price, new_t_out, new_irr, new_soc, new_t_in, new_base_load
        ], dtype=np.float32)
        
        # Return info dict for tracking and debugging in aim/tensorboard
        info = {
            'net_grid_kw': net_grid_kw,
            't_in': new_t_in,
            'soc': new_soc,
            'r_cost': r_cost,
            'r_comfort': r_comfort,
            'hp_cop': cop
        }
        
        return self.state, float(reward), terminated, truncated, info
        
    # --- Dummy implementations for remaining CustomEnvProtocol requirements ---
    def get_params(self) -> Any: 
        return self.config
        
    def get_optimal_value_function_plot(self, V_data: np.ndarray) -> Any: 
        pass
        
    def get_agent_value_function_plot(self, value_net: torch.nn.Module) -> Any: 
        # Return a blank figure so the logger doesn't crash
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "2D Plot not applicable for 7D State", ha='center')
        return fig, ax