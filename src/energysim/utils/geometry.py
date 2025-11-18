import numpy as np
from typing import List, Tuple

def get_polygon_normal(coords: List[np.array]) -> np.array:
    """
    Calculates the normal vector of a 3D polygon using Newell's Method.
    Robust against non-planar or complex polygons.
    """
    normal = np.array([0.0, 0.0, 0.0])
    for i in range(len(coords)):
        curr = coords[i]
        next_ = coords[(i + 1) % len(coords)]
        
        normal[0] += (curr[1] - next_[1]) * (curr[2] + next_[2])
        normal[1] += (curr[2] - next_[2]) * (curr[0] + next_[0])
        normal[2] += (curr[0] - next_[0]) * (curr[1] + next_[1])
        
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.array([0.0, 0.0, 1.0]) # Fallback to Up
    return normal / norm

def get_polygon_area_3d(coords: List[np.array]) -> float:
    """
    Calculates 3D surface area using the magnitude of the cross product sum.
    """
    if len(coords) < 3: return 0.0
    
    total = np.array([0.0, 0.0, 0.0])
    p0 = coords[0]
    for i in range(1, len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i+1]
        # Cross product of edges from p0
        cross = np.cross(p1 - p0, p2 - p0)
        total += cross
        
    return 0.5 * np.linalg.norm(total)

def get_azimuth_tilt(normal: np.array) -> Tuple[float, float]:
    """
    Converts normal vector to:
    Azimuth (0=North, 90=East, 180=South, 270=West)
    Tilt (0=Flat Roof, 90=Vertical Wall)
    """
    x, y, z = normal
    
    # Tilt: Angle with vertical (Z)
    tilt_rad = np.arccos(np.clip(z, -1.0, 1.0))
    tilt_deg = np.degrees(tilt_rad)
    
    # Azimuth
    # Math convention: arctan2(y, x) -> 0 at East.
    # E+ Convention: 0 at North (Y-axis).
    # We calculate standard math angle, then rotate.
    if abs(x) < 1e-4 and abs(y) < 1e-4:
        azimuth_deg = 0.0 # Flat
    else:
        # Vector projection on ground
        azimuth_deg = np.degrees(np.arctan2(x, y))
        if azimuth_deg < 0: 
            azimuth_deg += 360.0
            
    return azimuth_deg, tilt_deg