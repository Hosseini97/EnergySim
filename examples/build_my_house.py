import jax.numpy as jnp
from energysim.core.network_builder import RCNetworkBuilder

# --- Geometry (the numbers every R and C below is derived from) ---
# Living room: 35 m2 floor, 2.5 m ceiling, 45 m2 of exterior envelope (wall+roof+glazing).
# Bedroom:     25 m2 floor, 2.5 m ceiling, 30 m2 of exterior envelope.
# Shared partition between them: 20 m2.
ROOM_FLOOR_M2 = (35.0, 25.0)
CEILING_HEIGHT_M = 2.5
EXT_ENVELOPE_M2 = (45.0, 30.0)
PARTITION_M2 = 20.0

ROOM_VOL_M3 = tuple(a * CEILING_HEIGHT_M for a in ROOM_FLOOR_M2)   # 87.5, 62.5
TOTAL_VOL_M3 = sum(ROOM_VOL_M3)                                    # 150.0

# --- Material properties ---
RHO_CP_AIR_J_M3K = 1200.0
# Room air nodes lump in furniture and light internal partitions, which is
# conventionally 3-8x the bare air capacity.
FURNITURE_FACTOR = 5.0

# Exterior assembly: 0.25 m of masonry, U = 0.40 W/m2K including both surface films.
# Modelled 2R1C, i.e. the mass sits mid-wall and each half carries R_TOT/2.
EXT_ASSEMBLY_R_M2K_W = 1.0 / 0.40
EXT_WALL_CAPACITY_J_M2K = 0.25 * 1800.0 * 900.0        # thickness * rho * cp
# Interior partition: 0.12 m masonry, inner film + half-thickness conduction per side.
PARTITION_R_PER_SIDE_M2K_W = 0.125 + (0.06 / 0.7)
PARTITION_CAPACITY_J_M2K = 0.12 * 1800.0 * 900.0


def create_2_room_house():
    """
    Creates a simple 2-room house configuration (Living Room + Bedroom).

    Every resistance below is an *absolute* node-to-node resistance in K/W, i.e.
    an area-normalised R-value (m2K/W) divided by the area it acts over. Passing
    the bare R-value straight to add_resistor() is the easy mistake here: it
    leaves the envelope roughly two orders of magnitude too resistive, and the
    house then integrates solar gains for years without reaching equilibrium.
    """
    # 1. Initialize builder
    solar_split_factors = (0.7, 0.3)  # Solar hits Living Room more than Bedroom
    occupancy_split_factors = (0.6, 0.4)  # More people spend time in Living Room than Bedroom
    device_split_factors = (0.5, 0.5)  # Devices are evenly split for simplicity
    builder = RCNetworkBuilder(n_rooms=2, splits=(solar_split_factors, occupancy_split_factors, device_split_factors))

    # 2. Add nodes (Capacities in J/K)
    for i, vol in enumerate(ROOM_VOL_M3):
        builder.add_node(f"room_air_{i}", capacity_j_k=vol * RHO_CP_AIR_J_M3K * FURNITURE_FACTOR)
    for i, area in enumerate(EXT_ENVELOPE_M2):
        builder.add_node(f"wall_{i}", capacity_j_k=area * EXT_WALL_CAPACITY_J_M2K)
    builder.add_node("shared_wall", capacity_j_k=PARTITION_M2 * PARTITION_CAPACITY_J_M2K)

    # 3. Add connections (Resistances in K/W)
    # Exterior envelope, split half inside / half outside the wall mass.
    half_r = EXT_ASSEMBLY_R_M2K_W / 2.0
    for i, area in enumerate(EXT_ENVELOPE_M2):
        builder.add_resistor(f"room_air_{i}", f"wall_{i}", R_k_w=half_r / area)
        builder.add_resistor(f"wall_{i}", "ambient", R_k_w=half_r / area)

    # Inter-zone coupling through the shared partition.
    for i in range(2):
        builder.add_resistor(f"room_air_{i}", "shared_wall", R_k_w=PARTITION_R_PER_SIDE_M2K_W / PARTITION_M2)

    # Ventilation / infiltration. This replaces the old fixed room->ambient
    # resistors: it is the one loss path expressed directly in air changes per
    # hour, so it cannot silently drift away from a physical value.
    # ACH = k1 + k2*|T_amb - T_room| + k3*wind  ~= 0.5 ACH under typical conditions.
    builder.set_infiltration(total_volume_m3=TOTAL_VOL_M3, k1=0.35, k2=0.008, k3=0.05)

    # 4. Map Inputs (HVAC & Gains)
    # Map heating/cooling actions to specific rooms
    for i in range(2):
        builder.add_input_mapping("heating_w", f"room_air_{i}", room_index=i)
        builder.add_input_mapping("cooling_w", f"room_air_{i}", room_index=i)
        builder.add_input_mapping("occupancy_gains_w", f"room_air_{i}", room_index=i)
        builder.add_input_mapping("device_gains_w", f"room_air_{i}", room_index=i)

    # Solar hits walls mostly (70%), air slightly (30%)
    builder.add_input_mapping("solar_gains_w", "wall_0", room_index=0, fraction=0.7)
    builder.add_input_mapping("solar_gains_w", "room_air_0", room_index=0, fraction=0.3)
    builder.add_input_mapping("solar_gains_w", "wall_1", room_index=1, fraction=1.0)

    return builder.compile()
