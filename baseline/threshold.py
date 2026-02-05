"""
Baseline threshold-based flood detection system.
Simple heuristic without ML, state machine, or consensus.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class ThresholdConfig:
    """Configuration for threshold baseline."""
    water_threshold: float = 0.5
    rain_threshold: float = 2.0
    window_size: int = 5
    # Minimal hysteresis for fair comparison
    hysteresis_margin: float = 0.1  # TH_DOWN = water_threshold - margin
    debounce_steps: int = 2  # Minimal debouncing


class ThresholdBaseline:
    """
    Threshold-based flood detection baseline with minimal guardrails.
    
    Alert logic:
    - ALERT if water_mean_5 > water_threshold AND rain_sum_20 > rain_threshold
    - Uses minimal hysteresis (TH_UP != TH_DOWN) for fair comparison
    - Uses minimal debouncing (2 consecutive steps)
    - No consensus, no health-awareness (simpler than MAS)
    """
    
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            self.cfg = ThresholdConfig()
        else:
            baseline_cfg = config.get('baseline', {})
            self.cfg = ThresholdConfig(
                water_threshold=baseline_cfg.get('water_threshold', 0.5),
                rain_threshold=baseline_cfg.get('rain_threshold', 2.0),
                hysteresis_margin=baseline_cfg.get('hysteresis_margin', 0.1),
                debounce_steps=baseline_cfg.get('debounce_steps', 2)
            )
        
        self.water_buffer: deque = deque(maxlen=50)
        self.rain_buffer: deque = deque(maxlen=50)
        
        self.current_state = 'NORMAL'
        self.history: List[Dict] = []
        self.state_changes = 0
        self.consecutive_trigger = 0  # For debouncing
        self.consecutive_clear = 0
    
    def reset(self):
        """Reset baseline state."""
        self.water_buffer.clear()
        self.rain_buffer.clear()
        self.current_state = 'NORMAL'
        self.history.clear()
        self.state_changes = 0
        self.consecutive_trigger = 0
        self.consecutive_clear = 0
    
    def update(self, water: float, rain: float, step: int) -> str:
        """
        Update baseline with new readings.
        
        Args:
            water: Current water level reading
            rain: Current rainfall reading
            step: Current simulation step
            
        Returns:
            Current alert state ('NORMAL' or 'ALERT')
        """
        self.water_buffer.append(water)
        self.rain_buffer.append(rain)
        
        water_vals = list(self.water_buffer)[-self.cfg.window_size:]
        water_mean = np.mean(water_vals) if water_vals else 0
        
        rain_vals = list(self.rain_buffer)[-20:]
        rain_sum = np.sum(rain_vals)
        
        old_state = self.current_state
        
        # Hysteresis: different thresholds for up/down
        th_up = self.cfg.water_threshold
        th_down = self.cfg.water_threshold - self.cfg.hysteresis_margin
        
        # Determine if conditions are met
        if self.current_state == 'NORMAL':
            # Need to exceed th_up to trigger
            trigger = water_mean > th_up and rain_sum > self.cfg.rain_threshold
            if trigger:
                self.consecutive_trigger += 1
                self.consecutive_clear = 0
            else:
                self.consecutive_trigger = 0
            
            # Debouncing: need consecutive_trigger >= debounce_steps
            if self.consecutive_trigger >= self.cfg.debounce_steps:
                self.current_state = 'ALERT'
                self.consecutive_trigger = 0
        else:  # ALERT state
            # Need to drop below th_down to clear
            clear = water_mean < th_down or rain_sum < self.cfg.rain_threshold * 0.5
            if clear:
                self.consecutive_clear += 1
                self.consecutive_trigger = 0
            else:
                self.consecutive_clear = 0
            
            if self.consecutive_clear >= self.cfg.debounce_steps:
                self.current_state = 'NORMAL'
                self.consecutive_clear = 0
        
        if old_state != self.current_state:
            self.state_changes += 1
        
        self.history.append({
            'step': step,
            'water_mean_5': water_mean,
            'rain_sum_20': rain_sum,
            'state': self.current_state,
            'state_changes': self.state_changes
        })
        
        return self.current_state
    
    def is_alert(self) -> bool:
        """Check if currently in alert state."""
        return self.current_state == 'ALERT'
    
    def get_history_df(self) -> pd.DataFrame:
        """Get history as DataFrame."""
        return pd.DataFrame(self.history)


class ZonedThresholdBaseline:
    """
    Multi-zone threshold baseline for comparison with MAS.
    """
    
    def __init__(self, num_zones: int, config: Optional[dict] = None):
        self.num_zones = num_zones
        self.config = config
        self.zone_baselines: Dict[int, ThresholdBaseline] = {
            i: ThresholdBaseline(config) for i in range(num_zones)
        }
        self.global_alarm = False
        self.history: List[Dict] = []
    
    def reset(self):
        """Reset all zone baselines."""
        for baseline in self.zone_baselines.values():
            baseline.reset()
        self.global_alarm = False
        self.history.clear()
    
    def update(self, zone_readings: Dict[int, Dict[str, float]], step: int) -> Dict[int, str]:
        """
        Update all zones with new readings.
        
        Args:
            zone_readings: Dict mapping zone_id to {'water': ..., 'rain': ...}
            step: Current simulation step
            
        Returns:
            Dict mapping zone_id to alert state
        """
        zone_states = {}
        
        for zone_id, readings in zone_readings.items():
            if zone_id in self.zone_baselines:
                state = self.zone_baselines[zone_id].update(
                    readings['water'],
                    readings['rain'],
                    step
                )
                zone_states[zone_id] = state
        
        self.global_alarm = any(s == 'ALERT' for s in zone_states.values())
        
        self.history.append({
            'step': step,
            'global_alarm': self.global_alarm,
            'zone_states': zone_states.copy(),
            'total_state_changes': sum(b.state_changes for b in self.zone_baselines.values())
        })
        
        return zone_states
    
    def get_zone_history(self, zone_id: int) -> pd.DataFrame:
        """Get history for specific zone."""
        return self.zone_baselines[zone_id].get_history_df()
    
    def get_global_history(self) -> pd.DataFrame:
        """Get global history."""
        return pd.DataFrame(self.history)
    
    def get_total_state_changes(self) -> int:
        """Get total state changes across all zones."""
        return sum(b.state_changes for b in self.zone_baselines.values())
