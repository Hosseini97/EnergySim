# energysim/rl/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Dict, List, Callable, Any

from ..sim.simulator import JAXSimulator
from ..core.data.dataset import SimulationDataset
from ..core.shared.data_structs import (
    SystemActions, SystemState, ExogenousData
)
# --- NEW IMPORTS ---
from ..behavior.base import AbstractBehavioralModel
from dataclasses import fields

# --- REMOVED ProgrammaticFn ---

class EnergySimEnv(gym.Env):
    """
    A Gymnasium Env wrapper that orchestrates the JAXSimulator
    and the stateful behavioral models.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        simulator: JAXSimulator,
        dataset: SimulationDataset,
        behavioral_models: Optional[Dict[str, AbstractBehavioralModel]] = None
    ):
        super().__init__()
        self.simulator = simulator
        self.dataset = dataset
        self.behavioral_models = behavioral_models or {}
        self.dt_seconds = self.simulator.dt_seconds
        
        self._current_step = 0
        self._current_exo_data: ExogenousData = self.dataset[0] # Hold current step's data

        # Dynamically build action/obs spaces
        self._build_spaces()

    def _build_spaces(self):
        # --- 1. ACTION SPACE (unchanged) ---
        self.action_map: Dict[str, int] = {}
        action_lows: List[float] = []
        action_highs: List[float] = []
        idx_a = 0
        if self.simulator.active_configs["battery"]:
            conf = self.simulator.initial_battery.config
            self.action_map["battery_power_w"] = idx_a
            action_lows.append(-conf.max_power_w)
            action_highs.append(conf.max_power_w)
            idx_a += 1
        if self.simulator.active_configs["heat_pump"]:
            conf = self.simulator.initial_heat_pump.config
            self.action_map["heat_pump_power_w"] = idx_a
            action_lows.append(0.0)
            action_highs.append(conf.max_electrical_power_w)
            idx_a += 1
        if self.simulator.active_configs["ac"]:
            conf = self.simulator.initial_ac.config
            self.action_map["ac_power_w"] = idx_a
            action_lows.append(0.0)
            action_highs.append(conf.max_electrical_power_w)
            idx_a += 1
        if self.simulator.active_configs["storage"]:
            conf = self.simulator.initial_storage.config
            self.action_map["storage_discharge_w"] = idx_a
            action_lows.append(0.0)
            action_highs.append(conf.max_discharge_w)
            idx_a += 1
        self.action_space = spaces.Box(
            low=np.array(action_lows, dtype=np.float32),
            high=np.array(action_highs, dtype=np.float32)
        )
        
        # --- 2. OBSERVATION SPACE (Dynamically built) ---
        self.obs_map: Dict[str, int] = {}
        obs_lows: List[float] = []
        obs_highs: List[float] = []
        idx_o = 0

        # --- Internal States ---
        # Get SystemState fields recursively
        def add_state_keys(prefix, pytree_node):
            nonlocal idx_o
            if hasattr(pytree_node, "shape"): # It's an array leaf
                self.obs_map[prefix] = idx_o
                obs_lows.append(-np.inf) # Simple bounds for now
                obs_highs.append(np.inf)
                idx_o += 1
            elif hasattr(pytree_node, "__dataclass_fields__"):
                for field in fields(pytree_node):
                    add_state_keys(f"{prefix}.{field.name}", getattr(pytree_node, field.name))
        
        # Add all fields from SystemState (e.g., thermal.room_temp)
        add_state_keys("state", self.simulator.state) 

        # --- Exogenous Data ---
        # Add all fields from ExogenousData
        for field in fields(ExogenousData):
            self.obs_map[f"exo.{field.name}"] = idx_o
            obs_lows.append(-np.inf)
            obs_highs.append(np.inf)
            idx_o += 1
            
        self.observation_space = spaces.Box(
            low=np.array(obs_lows, dtype=np.float32),
            high=np.array(obs_highs, dtype=np.float32)
        )

    def _unflatten_action(self, action: np.ndarray) -> SystemActions:
        all_actions = {"battery_power_w": 0.0, "heat_pump_power_w": 0.0, "ac_power_w": 0.0, "storage_discharge_w": 0.0}
        for key, idx in self.action_map.items():
            all_actions[key] = float(action[idx])
        return SystemActions(
            battery_power_w=jnp.array(all_actions["battery_power_w"]),
            heat_pump_power_w=jnp.array(all_actions["heat_pump_power_w"]),
            ac_power_w=jnp.array(all_actions["ac_power_w"]),
            storage_discharge_w=jnp.array(all_actions["storage_discharge_w"])
        )

    def _get_merged_exo_data(self, step_idx: int, current_state: SystemState) -> ExogenousData:
        """
        Gets base data from dataset, runs behavioral models,
        and merges the results.
        """
        # 1. Get base data from the dataset
        base_exo = self.dataset[step_idx]

        if not self.behavioral_models:
            return base_exo

        # 2. Run all behavioral models
        programmatic_data_dict = {}
        total_device_gains_w = 0.0
        
        for key, model in self.behavioral_models.items():
            # e.g., key="ev_charger"
            field_name = f"{key}_load_w" # e.g., "ev_charger_load_w"
            
            if not hasattr(base_exo, field_name):
                print(f"Warning: Behavioral model key '{key}' has no matching field '{field_name}' in ExogenousData. Skipping.")
                continue

            # Run the model's step function
            power_w = model.step(step_idx, self.dt_seconds, current_state)
            programmatic_data_dict[field_name] = jnp.array(power_w)
            
            # Assume 100% of electrical load becomes heat
            total_device_gains_w += power_w

        # 3. Add calculated device gains
        programmatic_data_dict["device_gains_w"] = jnp.array(total_device_gains_w)

        # 4. Merge results into the base data
        merged_exo = base_exo.replace(**programmatic_data_dict)
        return merged_exo

    def _build_obs(self, state: SystemState, exo: ExogenousData) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # --- Internal States ---
        def fill_state_obs(prefix, pytree_node):
            if hasattr(pytree_node, "shape"): # It's an array leaf
                obs[self.obs_map[prefix]] = pytree_node
            elif hasattr(pytree_node, "__dataclass_fields__"):
                for field in fields(pytree_node):
                    fill_state_obs(f"{prefix}.{field.name}", getattr(pytree_node, field.name))
        
        fill_state_obs("state", state)

        # --- Exogenous Data ---
        for field in fields(exo):
            key = f"exo.{field.name}"
            if key in self.obs_map:
                obs[self.obs_map[key]] = getattr(exo, field.name)
        
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Reset JAX simulator
        state = self.simulator.reset()
        
        # Reset time
        self._current_step = 0
        
        # Reset all behavioral models
        for model in self.behavioral_models.values():
            model.reset()
            # Also reset their RNG if a new seed is provided
            if seed is not None:
                model.rng = np.random.default_rng(seed)

        # Get the *first* set of merged exogenous data for the observation
        self._current_exo_data = self._get_merged_exo_data(self._current_step, state)
        
        obs = self._build_obs(state, self._current_exo_data)
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        # 1. Get merged exogenous data for the *current* step
        # This was calculated at the end of the *last* step (or in reset)
        exo_data_k = self._current_exo_data
        
        # 2. Convert action from RL agent to simulator format
        actions_struct = self._unflatten_action(action)

        # 3. Step the simulator
        next_state, cost = self.simulator.step(actions_struct, exo_data_k)

        # 4. Advance time
        self._current_step += 1

        # 5. Check for termination
        terminated = self._current_step >= len(self.dataset)
        truncated = False
        
        # 6. Get exogenous data for the *next* step (to build the next observation)
        if terminated:
            # Use last available data
            self._current_exo_data = exo_data_k 
        else:
            # Run behavioral models for the *next* step
            self._current_exo_data = self._get_merged_exo_data(self._current_step, next_state)

        # 7. Build results for Gym
        obs = self._build_obs(next_state, self._current_exo_data)
        reward = -cost
        info = {"cost": cost}

        return obs, reward, terminated, truncated, info