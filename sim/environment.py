"""
FloodEnvironment: Simulates hydro-meteorological conditions including rainfall,
water levels, and soil saturation across a grid-based terrain.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class RainfallEvent:
    """Represents a rainfall event with intensity, duration, and timing."""
    intensity: float
    duration: int
    start_step: int
    
    def is_active(self, step: int) -> bool:
        return self.start_step <= step < self.start_step + self.duration
    
    def get_intensity(self, step: int) -> float:
        if not self.is_active(step):
            return 0.0
        progress = (step - self.start_step) / self.duration
        if progress < 0.3:
            return self.intensity * (progress / 0.3)
        elif progress > 0.7:
            return self.intensity * ((1.0 - progress) / 0.3)
        return self.intensity


@dataclass
class Zone:
    """Represents a monitoring zone with associated grid cells."""
    zone_id: int
    cells: List[Tuple[int, int]]
    is_river_zone: bool = False


class FloodEnvironment:
    """
    Simulates a flood-prone environment with:
    - Grid-based terrain with elevation
    - River system
    - Rainfall events
    - Water level dynamics
    - Soil saturation
    """
    
    def __init__(self, config: dict, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        self.grid_size = config['simulation']['grid_size']
        self.num_zones = config['simulation']['num_zones']
        
        self._init_terrain()
        self._init_zones()
        self._init_state()
        self.rainfall_events: List[RainfallEvent] = []
        self.current_step = 0
        
    def _init_terrain(self):
        """Initialize terrain elevation with river depression."""
        self.elevation = self.rng.uniform(
            self.config['elevation']['min'],
            self.config['elevation']['max'],
            (self.grid_size, self.grid_size)
        )
        
        river_width = self.config['river']['path_width']
        river_col = self.grid_size // 2
        for row in range(self.grid_size):
            for dc in range(-river_width // 2, river_width // 2 + 1):
                col = river_col + dc
                if 0 <= col < self.grid_size:
                    self.elevation[row, col] -= self.config['elevation']['river_depression']
                    self.elevation[row, col] = max(0, self.elevation[row, col])
        
        self.river_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for row in range(self.grid_size):
            for dc in range(-river_width // 2, river_width // 2 + 1):
                col = river_col + dc
                if 0 <= col < self.grid_size:
                    self.river_mask[row, col] = True
    
    def _init_zones(self):
        """Divide grid into monitoring zones."""
        sqrt_nz = int(np.sqrt(self.num_zones))
        if sqrt_nz * sqrt_nz != self.num_zones:
            raise ValueError(
                f"num_zones={self.num_zones} is not a perfect square. "
                f"Zone grid requires a perfect square (e.g. 4, 9, 16)."
            )
        if sqrt_nz > self.grid_size:
            raise ValueError(
                f"num_zones={self.num_zones} too large for grid_size={self.grid_size}. "
                f"sqrt(num_zones)={sqrt_nz} must be <= grid_size={self.grid_size}."
            )
        self.zones: List[Zone] = []
        zone_rows = self.grid_size // sqrt_nz
        zone_cols = self.grid_size // sqrt_nz
        
        zone_id = 0
        zones_per_row = sqrt_nz
        
        for zi in range(zones_per_row):
            for zj in range(zones_per_row):
                row_start = zi * zone_rows
                row_end = (zi + 1) * zone_rows if zi < zones_per_row - 1 else self.grid_size
                col_start = zj * zone_cols
                col_end = (zj + 1) * zone_cols if zj < zones_per_row - 1 else self.grid_size
                cells = []
                is_river = False
                for i in range(row_start, row_end):
                    for j in range(col_start, col_end):
                        cells.append((i, j))
                        if self.river_mask[i, j]:
                            is_river = True
                
                self.zones.append(Zone(zone_id=zone_id, cells=cells, is_river_zone=is_river))
                zone_id += 1
        
        self.cell_to_zone = {}
        for zone in self.zones:
            for cell in zone.cells:
                self.cell_to_zone[cell] = zone.zone_id
    
    def _init_state(self):
        """Initialize water level and soil saturation."""
        self.water_level = np.zeros((self.grid_size, self.grid_size))
        self.water_level[self.river_mask] = self.config['river']['base_flow']
        
        self.soil_saturation = np.clip(
            self.rng.normal(
                self.config['soil']['saturation_init_mean'],
                self.config['soil']['saturation_init_std'],
                (self.grid_size, self.grid_size)
            ),
            0.0, 1.0
        )
    
    def reset(self, soil_saturation_init: Optional[float] = None):
        """Reset environment to initial state."""
        self.current_step = 0
        self.rainfall_events = []
        self._init_state()
        
        if soil_saturation_init is not None:
            self.soil_saturation = np.clip(
                self.rng.normal(soil_saturation_init, 0.1, (self.grid_size, self.grid_size)),
                0.0, 1.0
            )
    
    def add_rainfall_event(self, intensity: float, duration: int, start_step: int):
        """Add a rainfall event to the simulation."""
        self.rainfall_events.append(RainfallEvent(intensity, duration, start_step))
    
    def generate_random_rainfall(self, scenario: str = 'normal'):
        """Generate rainfall event based on scenario config."""
        rain_cfg = self.config['rainfall']['scenarios'][scenario]
        
        intensity = max(0, self.rng.normal(rain_cfg['intensity_mean'], rain_cfg['intensity_std']))
        duration = max(10, int(self.rng.normal(rain_cfg['duration_mean'], rain_cfg['duration_std'])))
        start_step = self.rng.integers(rain_cfg['start_range'][0], rain_cfg['start_range'][1])
        
        self.add_rainfall_event(intensity, duration, start_step)
    
    def step(self) -> Dict[str, np.ndarray]:
        """
        Advance simulation by one step.
        Returns current state measurements.
        """
        current_rainfall = sum(
            event.get_intensity(self.current_step) for event in self.rainfall_events
        )
        
        self._update_soil(current_rainfall)
        self._update_water(current_rainfall)
        
        self.current_step += 1
        
        return {
            'water_level': self.water_level.copy(),
            'soil_saturation': self.soil_saturation.copy(),
            'rainfall': current_rainfall,
            'step': self.current_step
        }
    
    def _update_soil(self, rainfall: float):
        """Update soil saturation based on rainfall and infiltration."""
        infiltration = np.minimum(
            rainfall * (1 - self.soil_saturation),
            self.config['soil']['infiltration_rate']
        )
        self.soil_saturation = np.clip(
            self.soil_saturation + infiltration,
            0.0, self.config['soil']['max_saturation']
        )
        
        evap = self.config['water']['evaporation_rate'] * 0.5
        self.soil_saturation = np.maximum(0, self.soil_saturation - evap)
    
    def _update_water(self, rainfall: float):
        """Update water levels based on rainfall, runoff, and flow."""
        runoff = rainfall * self.soil_saturation
        self.water_level += runoff
        
        upstream_inflow = self.config['water']['upstream_inflow']
        self.water_level[0, self.river_mask[0, :]] += upstream_inflow
        
        spread = self.config['water']['spread_factor']
        new_water = self.water_level.copy()
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.water_level[i, j] > 0:
                    neighbors = self._get_lower_neighbors(i, j)
                    if neighbors:
                        flow_per_neighbor = self.water_level[i, j] * spread / len(neighbors)
                        for ni, nj in neighbors:
                            new_water[ni, nj] += flow_per_neighbor
                        new_water[i, j] -= flow_per_neighbor * len(neighbors)
        
        self.water_level = np.maximum(0, new_water)
        
        evap = self.config['water']['evaporation_rate']
        self.water_level = np.maximum(0, self.water_level - evap)
    
    def _get_lower_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Get neighboring cells with lower elevation."""
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                if self.elevation[ni, nj] < self.elevation[i, j]:
                    neighbors.append((ni, nj))
        return neighbors
    
    def get_zone_state(self, zone_id: int) -> Dict[str, float]:
        """Get aggregated state for a specific zone."""
        zone = self.zones[zone_id]
        water_vals = [self.water_level[c] for c in zone.cells]
        soil_vals = [self.soil_saturation[c] for c in zone.cells]
        
        return {
            'water_mean': np.mean(water_vals),
            'water_max': np.max(water_vals),
            'water_min': np.min(water_vals),
            'soil_mean': np.mean(soil_vals),
            'soil_max': np.max(soil_vals),
            'is_river_zone': zone.is_river_zone
        }
    
    def is_flooded(self, zone_id: int) -> bool:
        """Check if zone water level exceeds flood threshold."""
        zone = self.zones[zone_id]
        threshold = self.config['water']['flood_threshold']
        water_vals = [self.water_level[c] for c in zone.cells]
        return np.mean(water_vals) >= threshold
    
    def get_flood_status(self) -> Dict[int, bool]:
        """Get flood status for all zones."""
        return {z.zone_id: self.is_flooded(z.zone_id) for z in self.zones}
    
    def apply_pump(self, zone_id: int, capacity: float):
        """Apply pump to reduce water level in zone."""
        zone = self.zones[zone_id]
        for cell in zone.cells:
            self.water_level[cell] = max(0, self.water_level[cell] - capacity / len(zone.cells))
    
    def apply_gate(self, reduction_factor: float):
        """Reduce upstream inflow (gate closure)."""
        self.water_level[0, self.river_mask[0, :]] *= (1 - reduction_factor)
