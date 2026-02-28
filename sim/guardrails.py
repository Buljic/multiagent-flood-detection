"""
Guardrails module: State machine with hysteresis, debouncing, consensus gating,
and health-aware degradation for stable flood alert decisions.
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
import numpy as np


class AlertState(Enum):
    """Alert state machine states."""
    NORMAL = "NORMAL"
    SUSPECTED = "SUSPECTED"
    ALERT = "ALERT"
    COOLDOWN = "COOLDOWN"


@dataclass
class GuardrailsConfig:
    """Configuration for guardrails system."""
    TH_UP: float = 0.6
    TH_DOWN: float = 0.4
    K_UP: int = 3
    K_DOWN: int = 5
    CONS_MIN: float = 0.5
    HEALTH_MIN: float = 0.6
    degraded_TH_UP: float = 0.7
    degraded_K_UP: int = 5
    cooldown_steps: int = 10
    
    @classmethod
    def from_dict(cls, config: dict) -> 'GuardrailsConfig':
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class RingBuffer:
    """Fixed-size ring buffer for time series data."""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data: deque = deque(maxlen=max_size)
    
    def append(self, value: float):
        self.data.append(value)
    
    def get_last_n(self, n: int) -> List[float]:
        return list(self.data)[-n:] if len(self.data) >= n else list(self.data)
    
    def mean(self, n: int) -> float:
        vals = self.get_last_n(n)
        return np.mean(vals) if vals else 0.0
    
    def slope(self, n: int) -> float:
        vals = self.get_last_n(n)
        if len(vals) < 2:
            return 0.0
        x = np.arange(len(vals))
        return np.polyfit(x, vals, 1)[0]
    
    def max(self, n: int) -> float:
        vals = self.get_last_n(n)
        return np.max(vals) if vals else 0.0
    
    def sum(self, n: int) -> float:
        vals = self.get_last_n(n)
        return np.sum(vals)
    
    def __len__(self):
        return len(self.data)


class AlertStateMachine:
    """
    State machine for flood alert with hysteresis and debouncing.
    
    States: NORMAL -> SUSPECTED -> ALERT -> COOLDOWN -> NORMAL
    
    Transitions controlled by:
    - Hysteresis thresholds (TH_UP, TH_DOWN)
    - Debouncing counters (K_UP, K_DOWN consecutive steps)
    - Consensus gating (minimum sensor agreement)
    - Health-aware degradation (stricter thresholds when sensors fail)
    """
    
    def __init__(self, config: GuardrailsConfig):
        self.config = config
        self.state = AlertState.NORMAL
        self.consecutive_up = 0
        self.consecutive_down = 0
        self.cooldown_remaining = 0
        self.is_degraded = False
        self.state_changes = 0
    
    def reset(self):
        """Reset state machine to initial state."""
        self.state = AlertState.NORMAL
        self.consecutive_up = 0
        self.consecutive_down = 0
        self.cooldown_remaining = 0
        self.is_degraded = False
        self.state_changes = 0
    
    def get_effective_thresholds(self) -> tuple:
        """Get current thresholds based on degradation mode."""
        if self.is_degraded:
            return self.config.degraded_TH_UP, self.config.TH_DOWN, self.config.degraded_K_UP
        return self.config.TH_UP, self.config.TH_DOWN, self.config.K_UP
    
    def update(self, risk: float, consensus: float, health: float) -> AlertState:
        """
        Update state machine based on risk score, consensus, and health.

        Args:
            risk: ML model risk score [0, 1]
            consensus: Fraction of sensors showing rising trend [0, 1]
            health: Fraction of operational sensors [0, 1]

        Returns:
            Current alert state after update
        """
        risk = float(np.clip(risk, 0.0, 1.0))
        consensus = float(np.clip(consensus, 0.0, 1.0))
        health = float(np.clip(health, 0.0, 1.0))

        self.is_degraded = health < self.config.HEALTH_MIN
        th_up, th_down, k_up = self.get_effective_thresholds()
        
        consensus_ok = consensus >= self.config.CONS_MIN
        
        if self.state == AlertState.COOLDOWN:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining <= 0:
                self._transition_to(AlertState.NORMAL)
            return self.state
        
        if risk >= th_up and consensus_ok:
            self.consecutive_up += 1
            self.consecutive_down = 0
        elif risk < th_down:
            self.consecutive_down += 1
            self.consecutive_up = 0
        else:
            self.consecutive_up = max(0, self.consecutive_up - 1)
            self.consecutive_down = max(0, self.consecutive_down - 1)
        
        if self.state == AlertState.NORMAL:
            if self.consecutive_up >= k_up:
                self._transition_to(AlertState.SUSPECTED)
                
        elif self.state == AlertState.SUSPECTED:
            if self.consecutive_up >= k_up:
                self._transition_to(AlertState.ALERT)
            elif self.consecutive_down >= self.config.K_DOWN:
                self._transition_to(AlertState.NORMAL)
                
        elif self.state == AlertState.ALERT:
            if self.consecutive_down >= self.config.K_DOWN:
                self._transition_to(AlertState.COOLDOWN)
                self.cooldown_remaining = self.config.cooldown_steps
        
        return self.state
    
    def _transition_to(self, new_state: AlertState):
        """Transition to a new state."""
        if self.state != new_state:
            self.state = new_state
            self.consecutive_up = 0
            self.consecutive_down = 0
            self.state_changes += 1
    


class OutlierClipper:
    """Clips outlier values based on expected delta per step."""
    
    def __init__(self, max_delta: float = 0.3):
        self.max_delta = max_delta
        self.last_value: Optional[float] = None
    
    def clip(self, value: float) -> float:
        """Clip value if delta exceeds threshold."""
        if self.last_value is None:
            self.last_value = value
            return value
        
        delta = value - self.last_value
        if abs(delta) > self.max_delta:
            clipped = self.last_value + np.sign(delta) * self.max_delta
            self.last_value = clipped
            return clipped
        
        self.last_value = value
        return value
    
    def reset(self):
        self.last_value = None


class MissingValueHandler:
    """Handles missing sensor values with carry-forward and health penalty."""
    
    def __init__(self, max_carry: int = 5):
        self.max_carry = max_carry
        self.last_values: Dict[int, float] = {}
        self.carry_counts: Dict[int, int] = {}
    
    def process(self, sensor_id: int, value: Optional[float]) -> tuple:
        """
        Process sensor value, handling missing data.
        
        Returns:
            (processed_value, health_penalty)
        """
        if value is not None:
            self.last_values[sensor_id] = value
            self.carry_counts[sensor_id] = 0
            return value, 0.0
        
        if sensor_id in self.last_values:
            carry_count = self.carry_counts.get(sensor_id, 0) + 1
            self.carry_counts[sensor_id] = carry_count
            
            if carry_count <= self.max_carry:
                penalty = carry_count / self.max_carry * 0.5
                return self.last_values[sensor_id], penalty
        
        return 0.0, 1.0
    
    def reset(self):
        self.last_values.clear()
        self.carry_counts.clear()


class FeatureExtractor:
    """Extracts features from sensor buffers for ML model."""
    
    def __init__(self):
        self.water_buffer = RingBuffer(50)
        self.rain_buffer = RingBuffer(50)
        self.soil_buffer = RingBuffer(50)
    
    def update(self, water: float, rain: float, soil: float):
        """Add new readings to buffers."""
        self.water_buffer.append(water)
        self.rain_buffer.append(rain)
        self.soil_buffer.append(soil)
    
    def extract(self, consensus: float, health: float) -> Dict[str, float]:
        """Extract feature vector for ML model."""
        return {
            'water_mean_5': self.water_buffer.mean(5),
            'water_slope_5': self.water_buffer.slope(5),
            'water_max_10': self.water_buffer.max(10),
            'rain_sum_20': self.rain_buffer.sum(20),
            'rain_mean_10': self.rain_buffer.mean(10),
            'soil_mean_10': self.soil_buffer.mean(10),
            'consensus': consensus,
            'health': health
        }
    
    def reset(self):
        """Reset all buffers."""
        self.water_buffer = RingBuffer(50)
        self.rain_buffer = RingBuffer(50)
        self.soil_buffer = RingBuffer(50)
