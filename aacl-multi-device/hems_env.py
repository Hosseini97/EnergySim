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
        
        # A. Battery Dynamics (Assuming 1 step = 1 hour, so kW == kWh)
        # We must prevent the agent from overcharging or over-discharging
        energy_req = batt_action_kw 
        actual_batt_kw = energy_req
        
        if energy_req > 0: # Charging
            max_charge = (1.0 - soc) * self.config.battery_capacity_kwh
            actual_batt_kw = min(energy_req, max_charge)
        elif energy_req < 0: # Discharging
            max_discharge = soc * self.config.battery_capacity_kwh
            actual_batt_kw = -min(abs(energy_req), max_discharge)
            
        new_soc = soc + (actual_batt_kw / self.config.battery_capacity_kwh)
        
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
        pv_kw = (irr / 1000.0) * 5.0 
        
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
        new_hour = (hour + 1) % 24
        
        # TODO in Phase 5: Fetch next step values from real-world datasets
        # For now, keep weather/price static to ensure the code runs
        new_price, new_t_out, new_irr, new_base_load = price, t_out, irr, base_load
        
        self.state = np.array([
            new_hour, new_price, new_t_out, new_irr, new_soc, new_t_in, new_base_load
        ], dtype=np.float32)
        
        terminated = self.current_step >= self.config.max_steps
        truncated = False
        
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
        pass