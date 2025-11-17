import jax.numpy as jnp
import equinox as eqx
from energysim.core.network_builder import RCNetworkBuilder

def create_2_room_house():
    """
    Creates a simple 2-room house configuration.
    - Living Room (Zone 0)
    - Bedroom (Zone 1)
    - Both coupled to ambient and to each other via a shared wall.
    """
    
    # 1. Initialize builder for 2 rooms
    builder = RCNetworkBuilder(n_rooms=2)
    
    # 2. Add nodes
    # Capacities (J/K)
    C_air_living = 5.0e5
    C_wall_living = 1.0e7
    C_air_bed = 3.0e5
    C_wall_bed = 0.8e7
    C_shared_wall = 0.5e7
    
    builder.add_node("room_air_0", capacity_j_k=C_air_living) # Living room
    builder.add_node("wall_0", capacity_j_k=C_wall_living)  # Living room outer wall
    builder.add_node("room_air_1", capacity_j_k=C_air_bed)     # Bedroom
    builder.add_node("wall_1", capacity_j_k=C_wall_bed)      # Bedroom outer wall
    builder.add_node("shared_wall", capacity_j_k=C_shared_wall)
    
    # 3. Add resistors (Connections)
    # Resistances (K/W)
    R_wall_amb = 2.0
    R_air_wall = 1.0
    R_air_shared = 1.6
    R_vent = 4.0 # Ventilation
    
    # Living Room (Zone 0)
    builder.add_resistor("wall_0", "ambient", R_k_w=R_wall_amb)
    builder.add_resistor("room_air_0", "wall_0", R_k_w=R_air_wall)
    builder.add_resistor("room_air_0", "ambient", R_k_w=R_vent) # Ventilation
    
    # Bedroom (Zone 1)
    builder.add_resistor("wall_1", "ambient", R_k_w=R_wall_amb)
    builder.add_resistor("room_air_1", "wall_1", R_k_w=R_air_wall)
    builder.add_resistor("room_air_1", "ambient", R_k_w=R_vent) # Ventilation
    
    # --- Multi-room coupling ---
    builder.add_resistor("room_air_0", "shared_wall", R_k_w=R_air_shared)
    builder.add_resistor("room_air_1", "shared_wall", R_k_w=R_air_shared)
    
    # 4. Map inputs
    # --- HVAC (Thermal) ---
    # This maps thermal power from storage (heating) to the rooms
    builder.add_input_mapping("heating_w", "room_air_0", room_index=0)
    builder.add_input_mapping("heating_w", "room_air_1", room_index=1)
    # This maps thermal power from AC (cooling) to the rooms
    builder.add_input_mapping("cooling_w", "room_air_0", room_index=0)
    builder.add_input_mapping("cooling_w", "room_air_1", room_index=1)

    # --- Solar (Exogenous) ---
    # Split solar gains for Zone 0: 70% to wall, 30% to air
    builder.add_input_mapping("solar_gains_w", "wall_0", room_index=0, fraction=0.7)
    builder.add_input_mapping("solar_gains_w", "room_air_0", room_index=0, fraction=0.3)
    # All solar for Zone 1 hits the wall
    builder.add_input_mapping("solar_gains_w", "wall_1", room_index=1, fraction=1.0)

    # --- Occupancy (Exogenous) ---
    builder.add_input_mapping("occupancy_gains_w", "room_air_0", room_index=0)
    builder.add_input_mapping("occupancy_gains_w", "room_air_1", room_index=1)

    # --- Device Gains (Exogenous) ---
    builder.add_input_mapping("device_gains_w", "room_air_0", room_index=0)
    builder.add_input_mapping("device_gains_w", "room_air_1", room_index=1)
    
    # 5. Compile
    print("Compiling network...")
    thermal_config = builder.compile()
    print("Compilation complete.")
    
    return thermal_config

if __name__ == "__main__":
    t_config = create_2_room_house()
    
    # Save the config object
    save_path = "my_house_config.eqx"
    eqx.tree_serialise_leaves(save_path, t_config)
    
    print(f"\nSaved ThermalConfig to {save_path}")
    print(f"  N_nodes = {t_config.C_inv_vector.shape[0]}")
    print(f"  A_matrix shape = {t_config.A_matrix.shape}")
    print(f"  B_matrix shape = {t_config.B_matrix.shape}")
    print(f"  Ambient index = {t_config.ambient_air_index}")
    print(f"  Room Air indices = {t_config.room_air_indices}")