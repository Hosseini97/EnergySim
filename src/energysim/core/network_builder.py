# energysim/core/network_builder.py
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from .shared.data_structs import ThermalConfig

@dataclass
class _RCNode:
    """Internal helper class to store node data during build."""
    name: str
    capacity_j_k: float
    index: int = -1 # Final index in the T-vector

@dataclass
class _Resistor:
    """Internal helper class for resistors."""
    node_a_name: str
    node_b_name: str
    R_k_w: float

@dataclass
class _InputMapping:
    """Internal helper class for input mappings."""
    input_key: str  # e.g., "hvac", "solar"
    node_name: str
    room_index: Optional[int] # For vectorized inputs
    fraction: float

class RCNetworkBuilder:
    """
    A non-JAX builder to create the matrices for an RCNetworkModel.
    
    Provides a friendly API to define nodes (capacitors), resistors,
    and input mappings, then compiles them into the JAX-compatible
    ThermalConfig object.
    """
    
    def __init__(self, n_rooms: int):
        """
        Args:
            n_rooms: The number of independent rooms/zones in the model.
                     This defines the size of the vectorized inputs 
                     (e.g., HVAC actions, solar gains).
        """
        self.n_rooms = n_rooms
        self._nodes: Dict[str, _RCNode] = {}
        self._resistors: List[_Resistor] = []
        self._mappings: List[_InputMapping] = []
        
        # Define the static order of the flat input vector (U_flat).
        # This is the "contract" between the builder and the model.
        self._input_keys_order = [
                        "heating_w",         # <-- CHANGED (e.g., from storage)
                        "cooling_w",         # <-- CHANGED (e.g., from AC, will be negative)
                        "solar_gains_w",
                        "occupancy_gains_w",
                        "device_gains_w"
                    ]
        
        # Add the ambient node by default
        self.add_node("ambient", capacity_j_k=jnp.inf)

    def add_node(self, name: str, capacity_j_k: float):
        """
        Adds a thermal node (capacitor) to the network.
        
        Args:
            name: A unique string name (e.g., "living_room_air", "shared_wall").
            capacity_j_k: The thermal capacity in Joules/Kelvin.
                          Use jnp.inf for the 'ambient' node.
        """
        if name in self._nodes:
            raise ValueError(f"Node name '{name}' already exists.")
        self._nodes[name] = _RCNode(name=name, capacity_j_k=capacity_j_k)

    def add_resistor(self, node_a: str, node_b: str, R_k_w: float):
        """
        Adds a thermal resistor (connection) between two nodes.
        
        Args:
            node_a: Name of the first node.
            node_b: Name of the second node.
            R_k_w: The thermal resistance in Kelvin/Watt.
        """
        if R_k_w <= 0:
            raise ValueError("Resistance must be > 0.")
        self._resistors.append(_Resistor(node_a, node_b, R_k_w))

    def add_input_mapping(self, 
                          input_key: str, 
                          node_name: str, 
                          room_index: Optional[int] = None, 
                          fraction: float = 1.0):
        """
        Maps a heat input (from actions or exogenous data) to a node.
        
        Args:
            input_key: The name of the input. Must be one of:
                       ["heat_pump_power_w", "ac_power_w", "storage_discharge_w",
                        "solar_gains_w", "occupancy_gains_w", "device_gains_w"]
            node_name: The name of the node to receive the heat.
            room_index: The index (0 to n_rooms-1) if this is a vectorized input.
                        Must be provided for all inputs.
            fraction: The fraction (0.0 to 1.0) of this input to assign
                      to this node.
        """
        if input_key not in self._input_keys_order:
            raise ValueError(f"Unknown input_key '{input_key}'. Must be one of {self._input_keys_order}")
        if node_name not in self._nodes:
            raise ValueError(f"Node name '{node_name}' not found. Call add_node() first.")
        if room_index is None:
             raise ValueError(f"room_index must be provided for input_key '{input_key}'.")
        if not (0 <= room_index < self.n_rooms):
            raise ValueError(f"room_index {room_index} is out of range for n_rooms={self.n_rooms}.")
            
        self._mappings.append(_InputMapping(input_key, node_name, room_index, fraction))

    def _get_input_col_index(self, key: str, room_idx: int) -> int:
        """Calculates the column index in the B_matrix for a given input."""
        base_offset = self._input_keys_order.index(key) * self.n_rooms
        return base_offset + room_idx

    def compile(self) -> ThermalConfig:
        """
        Compiles the defined network into the required matrices
        and returns a ThermalConfig object.
        """
        
        # --- 1. Finalize Node Order ---
        # Put 'ambient' at index 0, others sorted alphabetically
        node_names = sorted([name for name in self._nodes if name != "ambient"])
        final_node_order = ["ambient"] + node_names
        
        N_nodes = len(final_node_order)
        
        # Update nodes with their final index
        for i, name in enumerate(final_node_order):
            self._nodes[name].index = i

        # --- 2. Build C_inv_vector ---
        # (N_nodes,)
        c_inv_vector = np.zeros(N_nodes, dtype=np.float32)
        for name, node in self._nodes.items():
            c_inv_vector[node.index] = 1.0 / node.capacity_j_k
        
        # --- 3. Build A_matrix ---
        # (N_nodes, N_nodes)
        A_matrix = np.zeros((N_nodes, N_nodes), dtype=np.float32)
        
        for res in self._resistors:
            if res.node_a_name not in self._nodes or res.node_b_name not in self._nodes:
                raise ValueError(f"Resistor connects to unknown node: {res}")
                
            i = self._nodes[res.node_a_name].index
            j = self._nodes[res.node_b_name].index
            G = 1.0 / res.R_k_w
            
            A_matrix[i, i] -= G
            A_matrix[j, j] -= G
            A_matrix[i, j] += G
            A_matrix[j, i] += G
            
        # --- 4. Build B_matrix ---
        N_inputs_flat = len(self._input_keys_order) * self.n_rooms
        B_matrix = np.zeros((N_nodes, N_inputs_flat), dtype=np.float32)
        
        for mapping in self._mappings:
            row_idx = self._nodes[mapping.node_name].index
            col_idx = self._get_input_col_index(mapping.input_key, mapping.room_index)
            
            B_matrix[row_idx, col_idx] += mapping.fraction
            
        # --- 5. Get Key Indices ---
        ambient_air_index = self._nodes["ambient"].index
        
        # Helper to find nodes by prefix
        def find_indices(prefix):
            indices = []
            for name, node in self._nodes.items():
                if name.startswith(prefix):
                    indices.append(node.index)
            # Sort by index to ensure order
            return tuple(sorted(indices))
            
        # This is a convention: name your nodes like "room_air_0", "room_air_1", etc.
        room_air_indices = find_indices("room_air_")
        wall_indices = find_indices("wall_")
        mass_indices = find_indices("mass_")
        
        if len(room_air_indices) != self.n_rooms:
            print(f"Warning: Found {len(room_air_indices)} nodes with prefix 'room_air_' but n_rooms={self.n_rooms}")

        # --- 6. Create ThermalConfig ---
        return ThermalConfig(
            A_matrix=jnp.array(A_matrix),
            C_inv_vector=jnp.array(c_inv_vector),
            B_matrix=jnp.array(B_matrix),
            ambient_air_index=ambient_air_index,
            room_air_indices=room_air_indices,
            wall_indices=wall_indices,
            mass_indices=mass_indices,
            setpoint=21.0, # You can override this later
            comfort_band=1.0 # You can override this later
        )