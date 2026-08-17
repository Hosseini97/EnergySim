import warnings

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from .shared.data_structs import ThermalConfig

@dataclass
class _RCNode:
    name: str
    capacity_j_k: float
    index: int = -1

@dataclass
class _Resistor:
    node_a_name: str
    node_b_name: str
    R_k_w: float

@dataclass
class _InputMapping:
    input_key: str
    node_name: str
    room_index: Optional[int]
    fraction: float

class RCNetworkBuilder:
    def __init__(self, n_rooms: int, splits: Optional[Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]] = None):
        self.n_rooms = n_rooms
        self._nodes: Dict[str, _RCNode] = {}
        self._resistors: List[_Resistor] = []
        self._mappings: List[_InputMapping] = []

        if splits:
            self._solar_split_factors, self._occupancy_split_factors, self._device_split_factors = splits
        else:
            self._solar_split_factors = (1.0/n_rooms,) * n_rooms
            self._occupancy_split_factors = (1.0/n_rooms,) * n_rooms
            self._device_split_factors = (1.0/n_rooms,) * n_rooms

        # Infiltration / Coupling Settings
        self._infiltration_enabled = False
        self._inf_params = (0.1, 0.0, 0.0) # k1, k2, k3
        self._total_volume = 0.0
        self._waste_heat_node_name = None

        self._input_keys_order = [
            "heating_w", "cooling_w", "solar_gains_w", 
            "occupancy_gains_w", "device_gains_w"
        ]
        self.add_node("ambient", capacity_j_k=jnp.inf)

    def add_node(self, name: str, capacity_j_k: float):
        if name in self._nodes: raise ValueError(f"Node {name} exists.")
        self._nodes[name] = _RCNode(name, capacity_j_k)

    def add_resistor(self, node_a: str, node_b: str, R_k_w: float):
        if R_k_w <= 0: raise ValueError("R must be > 0")
        self._resistors.append(_Resistor(node_a, node_b, R_k_w))

    def add_input_mapping(self, input_key: str, node_name: str, room_index: int, fraction: float = 1.0):
        """
        Define how a specific room's input vector maps to thermal nodes.
        Allows uneven splitting (e.g. Solar -> 70% Floor, 30% Air).
        """
        if input_key not in self._input_keys_order:
            raise ValueError(f"Unknown key {input_key}")
        if node_name not in self._nodes:
            raise ValueError(f"Unknown node {node_name}")
        self._mappings.append(_InputMapping(input_key, node_name, room_index, fraction))

    def set_infiltration(self, total_volume_m3: float, k1: float = 0.1, k2: float = 0.0, k3: float = 0.0):
        """Enables dynamic infiltration model."""
        self._infiltration_enabled = True
        self._total_volume = total_volume_m3
        self._inf_params = (k1, k2, k3)

    def set_waste_heat_node(self, node_name: str):
        """Sets the node where storage/HVAC waste heat is dumped."""
        if node_name not in self._nodes:
            raise ValueError(f"Unknown node {node_name}")
        self._waste_heat_node_name = node_name

    def _get_input_col_index(self, key: str, room_idx: int) -> int:
        base_offset = self._input_keys_order.index(key) * self.n_rooms
        return base_offset + room_idx

    # A building's slowest lump settles in hours to a few days. Anything past
    # this means the loss paths are too weak to balance the gains on any horizon
    # a simulation actually runs for, so temperature just ramps.
    MAX_PLAUSIBLE_TIME_CONSTANT_S = 14 * 86400.0

    def _validate_losses(self, node_order, A_matrix, c_inv_vector):
        """Catch envelopes that cannot shed the heat pushed into them.

        Solar and internal gains are strictly non-negative, so they only ever
        add energy. If a node cannot reach ambient, or can only reach it through
        a conductance that is orders of magnitude too small, the network stays
        formally stable but converges to a nonsense fixed point over a horizon
        far longer than the simulation -- which reads as unbounded drift.
        """
        amb_idx = self._nodes["ambient"].index

        # 1. Reachability: every capacitive node needs a path to ambient.
        adjacency = {name: set() for name in node_order}
        for res in self._resistors:
            adjacency[res.node_a_name].add(res.node_b_name)
            adjacency[res.node_b_name].add(res.node_a_name)
        if self._infiltration_enabled:
            # Infiltration is an air<->ambient path that never enters A_matrix.
            for name in node_order:
                if name.startswith("room_air_"):
                    adjacency[name].add("ambient")
                    adjacency["ambient"].add(name)

        seen, stack = {"ambient"}, ["ambient"]
        while stack:
            for nb in adjacency[stack.pop()]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        orphans = [n for n in node_order if n not in seen]
        if orphans:
            raise ValueError(
                f"Nodes {orphans} have no conductance path to 'ambient'. Any gain "
                "mapped onto them accumulates forever. Add a resistor to ambient "
                "(directly or via a wall), or call set_infiltration()."
            )

        # 2. Magnitude: total UA to ambient vs. total capacitance.
        ua_to_ambient = float(-A_matrix[amb_idx, amb_idx])
        if self._infiltration_enabled:
            ua_to_ambient += (self._inf_params[0] * self._total_volume * 1200.0) / 3600.0

        total_capacity = float(sum(
            1.0 / c for c in c_inv_vector if c > 0.0
        ))
        if ua_to_ambient <= 0.0:
            raise ValueError("Network has zero total conductance to 'ambient'.")

        tau_s = total_capacity / ua_to_ambient
        if tau_s > self.MAX_PLAUSIBLE_TIME_CONSTANT_S:
            warnings.warn(
                f"Thermal network has UA={ua_to_ambient:.2f} W/K to ambient against "
                f"C={total_capacity:.3g} J/K, a global time constant of "
                f"{tau_s / 86400:.0f} days. Indoor temperature will ramp rather than "
                "settle under steady gains. Resistances passed to add_resistor() are "
                "absolute K/W, not area-normalised R-values (m2K/W) -- divide the "
                "R-value by the area it acts over.",
                stacklevel=3,
            )

    def compile(self) -> ThermalConfig:
        # Node ordering
        node_names = sorted([n for n in self._nodes if n != "ambient"])
        final_node_order = ["ambient"] + node_names
        N_nodes = len(final_node_order)
        for i, name in enumerate(final_node_order):
            self._nodes[name].index = i

        # Vectors/Matrices
        c_inv_vector = np.zeros(N_nodes, dtype=np.float32)
        for name, node in self._nodes.items():
            c_inv_vector[node.index] = 1.0 / node.capacity_j_k

        A_matrix = np.zeros((N_nodes, N_nodes), dtype=np.float32)
        for res in self._resistors:
            i = self._nodes[res.node_a_name].index
            j = self._nodes[res.node_b_name].index
            G = 1.0 / res.R_k_w
            A_matrix[i, i] -= G; A_matrix[j, j] -= G
            A_matrix[i, j] += G; A_matrix[j, i] += G

        N_inputs_flat = len(self._input_keys_order) * self.n_rooms
        B_matrix = np.zeros((N_nodes, N_inputs_flat), dtype=np.float32)
        for m in self._mappings:
            r_idx = self._nodes[m.node_name].index
            c_idx = self._get_input_col_index(m.input_key, m.room_index)
            B_matrix[r_idx, c_idx] += m.fraction

        # Indices
        def find_indices(prefix):
            return tuple(sorted([n.index for name, n in self._nodes.items() if name.startswith(prefix)]))
        
        waste_idx = -1
        if self._waste_heat_node_name:
            waste_idx = self._nodes[self._waste_heat_node_name].index

        # N_inputs_flat is 10 (5 keys * 2 rooms)
        # N_raw_inputs is 7 (2 heating + 2 cooling + 1 solar + 1 occ + 1 dev)
        N_raw_inputs = (2 * self.n_rooms) + 3 
        split_matrix = np.zeros((N_inputs_flat, N_raw_inputs), dtype=np.float32)

        # Row index track (where it goes in the 10-vector)
        # Column index track (where it comes from in the 7-vector)
        col_ptr = 0

        for key in self._input_keys_order:
            if key in ["heating_w", "cooling_w"]:
                # These are already per-room (length 2), so map 1-to-1
                for r in range(self.n_rooms):
                    row_idx = self._get_input_col_index(key, r)
                    split_matrix[row_idx, col_ptr] = 1.0
                    col_ptr += 1
            else:
                # These are scalars (length 1), so map 1-to-many using split factors
                for r in range(self.n_rooms):
                    row_idx = self._get_input_col_index(key, r)
                    if key == "solar_gains_w":
                        split_matrix[row_idx, col_ptr] = self._solar_split_factors[r]
                    elif key == "occupancy_gains_w":
                        split_matrix[row_idx, col_ptr] = self._occupancy_split_factors[r]
                    elif key == "device_gains_w":
                        split_matrix[row_idx, col_ptr] = self._device_split_factors[r]
                # Move to next raw input only after filling all room rows for this scalar
                col_ptr += 1

        B_matrix_final = B_matrix @ split_matrix

        self._validate_losses(final_node_order, A_matrix, c_inv_vector)

        return ThermalConfig(
            A_matrix=jnp.array(A_matrix),
            C_inv_vector=jnp.array(c_inv_vector),
            B_matrix=jnp.array(B_matrix_final),

            ambient_air_index=self._nodes["ambient"].index,
            room_air_indices=find_indices("room_air_"),
            wall_indices=find_indices("wall_"),
            mass_indices=find_indices("mass_"),
            waste_heat_node_index=waste_idx,
            use_dynamic_infiltration=self._infiltration_enabled,
            room_vol_m3=self._total_volume,
            inf_k1=self._inf_params[0],
            inf_k2=self._inf_params[1],
            inf_k3=self._inf_params[2]
        )