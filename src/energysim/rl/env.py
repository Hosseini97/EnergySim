# energysim/rl/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Dict, List, Callable, Any, Tuple

from ..sim.simulator import JAXSimulator
from ..core.data.dataset import SimulationDataset
from ..core.shared.data_structs import (
    SystemActions, SystemState, ExogenousData
)
from ..behavior.base import AbstractBehavioralModel
from dataclasses import fields


class EnergySimEnv(gym.Env):
    """
    A Gymnasium Env wrapper that orchestrates the JAXSimulator
    and the stateful behavioral models.

    --- UPDATED ---
    This class is now zonal-aware. It dynamically builds action
    and observation spaces based on the number of rooms detected
    in the simulator's ThermalConfig.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        simulator: JAXSimulator,
        dataset: SimulationDataset,
        behavioral_models: Optional[Dict[str, AbstractBehavioralModel]] = None,
        internal_gain_devices: List[str] = ["dishwasher", "cooking", "clothes_dryer"]
    ):
        super().__init__()
        self.simulator = simulator
        self.dataset = dataset
        self.behavioral_models = behavioral_models or {}
        self.dt_seconds = self.simulator.dt_seconds
        self.internal_gain_devices = internal_gain_devices

        t_config = self.simulator.initial_thermal.config
        self.n_rooms = len(t_config.room_air_indices)
        if self.n_rooms == 0:
            raise ValueError(
                "ThermalConfig has no 'room_air_indices'. Env requires n_rooms > 0."
            )
        self.room_air_indices = jnp.array(t_config.room_air_indices)

        self._current_step = 0
        self._current_exo_data: ExogenousData = self._get_merged_exo_data(
            0, self.simulator.state, is_forecast=False
        )

        self.action_map_slices: Dict[str, Tuple[int, int]] = {}
        self.obs_map_slices: Dict[str, slice] = {}

        self._build_spaces()

    def _build_spaces(self):
        action_lows: List[float] = []
        action_highs: List[float] = []
        idx_a = 0

        def add_action(key: str, n_dims: int, low: float, high: float):
            nonlocal idx_a
            self.action_map_slices[key] = (idx_a, idx_a + n_dims)
            action_lows.extend([low] * n_dims)
            action_highs.extend([high] * n_dims)
            idx_a += n_dims

        if self.simulator.active_configs["battery"]:
            conf = self.simulator.initial_battery.config
            add_action("battery_power_w", 1, -conf.max_power_w, conf.max_power_w)

        if self.simulator.active_configs["heat_pump"]:
            conf = self.simulator.initial_heat_pump.config
            add_action("heat_pump_power_w", self.n_rooms, 0.0, conf.max_electrical_power_w / self.n_rooms)

        if self.simulator.active_configs["ac"]:
            conf = self.simulator.initial_ac.config
            add_action("ac_power_w", self.n_rooms, 0.0, conf.max_electrical_power_w / self.n_rooms)

        if self.simulator.active_configs["storage"]:
            conf = self.simulator.initial_storage.config
            add_action("storage_discharge_w", self.n_rooms, 0.0, conf.max_discharge_w / self.n_rooms)

        self.action_space = spaces.Box(
            low=np.array(action_lows, dtype=np.float32),
            high=np.array(action_highs, dtype=np.float32)
        )

        obs_lows: List[float] = []
        obs_highs: List[float] = []
        idx_o = 0

        def _flatten_and_add_keys(prefix, pytree_node):
            nonlocal idx_o
            if hasattr(pytree_node, "shape"):
                size = int(np.prod(pytree_node.shape))
                self.obs_map_slices[prefix] = slice(idx_o, idx_o + size)
                obs_lows.extend([-np.inf] * size)
                obs_highs.extend([np.inf] * size)
                idx_o += size
            elif hasattr(pytree_node, "__dataclass_fields__"):
                for field in fields(pytree_node):
                    _flatten_and_add_keys(
                        f"{prefix}.{field.name}",
                        getattr(pytree_node, field.name)
                    )

        _flatten_and_add_keys("state", self.simulator.state)
        _flatten_and_add_keys("exo", self._current_exo_data)

        self.observation_space = spaces.Box(
            low=np.array(obs_lows, dtype=np.float32),
            high=np.array(obs_highs, dtype=np.float32)
        )

    def _unflatten_action(self, action: np.ndarray) -> SystemActions:
        actions_dict = {
            "battery_power_w": jnp.array(0.0),
            "heat_pump_power_w": jnp.zeros(self.n_rooms),
            "ac_power_w": jnp.zeros(self.n_rooms),
            "storage_discharge_w": jnp.zeros(self.n_rooms)
        }

        for key, (start, end) in self.action_map_slices.items():
            val = action[start:end]
            if end - start == 1:
                actions_dict[key] = jnp.array(val[0])
            else:
                actions_dict[key] = jnp.array(val)

        return SystemActions(**actions_dict)

    def _get_merged_exo_data(
        self, step_idx: int, current_state: SystemState, is_forecast: bool = False
    ) -> ExogenousData:
        if is_forecast:
            base_exo = self.dataset.get_forecast(step_idx, self.n_rooms)
            base_exo = self.dataset[step_idx]
        else:
            base_exo = self.dataset[step_idx]

        scalar_solar = base_exo.solar_gains_w
        scalar_occupancy = base_exo.occupancy_gains_w

        zonal_solar_gains = jnp.full((self.n_rooms,), scalar_solar / self.n_rooms)
        zonal_occupancy_gains = jnp.full((self.n_rooms,), scalar_occupancy / self.n_rooms)

        base_exo = base_exo.replace(
            solar_gains_w=zonal_solar_gains,
            occupancy_gains_w=zonal_occupancy_gains
        )

        if not self.behavioral_models:
            return base_exo

        programmatic_data_dict = {}
        internal_gains_load_w = 0.0

        for key, model in self.behavioral_models.items():
            field_name = f"{key}_load_w"

            if not hasattr(base_exo, field_name):
                print(f"Warning: Behavioral model key '{key}' has no matching field '{field_name}' in ExogenousData. Skipping.")
                continue

            power_w = model.step(step_idx, self.dt_seconds, current_state)
            programmatic_data_dict[field_name] = jnp.array(power_w)

            if key in self.internal_gain_devices:
                internal_gains_load_w += power_w

        zonal_device_gains = jnp.full(
            (self.n_rooms,), internal_gains_load_w / self.n_rooms
        )
        programmatic_data_dict["device_gains_w"] = zonal_device_gains

        merged_exo = base_exo.replace(**programmatic_data_dict)
        return merged_exo

    def _build_obs(self, state: SystemState, exo: ExogenousData) -> np.ndarray:
        # --- UPDATED for Zonal/Vector ---
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Recursive helper to fill obs vector
        def fill_obs_vec(prefix, pytree_node):
            if hasattr(pytree_node, "shape"): # It's an array leaf
                sl = self.obs_map_slices.get(prefix)
                if sl:
                    obs[sl] = np.array(pytree_node).flatten()
            elif hasattr(pytree_node, "__dataclass_fields__"):
                for field in fields(pytree_node):
                    fill_obs_vec(
                        f"{prefix}.{field.name}", 
                        getattr(pytree_node, field.name)
                    )
        
        fill_obs_vec("state", state)
        fill_obs_vec("exo", exo)
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