"""
Multi-Agent System agents for flood detection:
- SensorAgent: Emits noisy sensor readings
- EdgeAggregatorAgent: Aggregates sensors, extracts features, runs ML + guardrails
- CoordinatorAgent: Global fusion and alarm management
- MitigationAgent: Optional countermeasure execution
"""

import numpy as np
from mesa import Agent
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import joblib

from .guardrails import (
    AlertState, AlertStateMachine, GuardrailsConfig,
    FeatureExtractor, OutlierClipper, MissingValueHandler
)


class SensorAgent(Agent):
    """
    Sensor agent that emits noisy readings of water level, rainfall, and soil saturation.
    Can randomly drop out to simulate sensor failures.
    """
    
    def __init__(self, unique_id: int, model, zone_id: int, cell: tuple, config: dict):
        super().__init__(unique_id, model)
        self.zone_id = zone_id
        self.cell = cell
        self.config = config
        self.noise_std = config['sensors']['noise_std']
        self.dropout_rate = config['sensors']['dropout_rate']
        self.is_active = True
        self.last_reading: Optional[Dict[str, float]] = None
        self.trend_rising = False
    
    def step(self):
        """Generate sensor reading with noise and potential dropout."""
        if self.model.rng.random() < self.dropout_rate:
            self.is_active = False
            self.last_reading = None
            return
        
        self.is_active = True
        env = self.model.environment
        
        water_true = env.water_level[self.cell]
        soil_true = env.soil_saturation[self.cell]
        rain_true = sum(e.get_intensity(env.current_step) for e in env.rainfall_events)
        
        water = water_true + self.model.rng.normal(0, self.noise_std)
        soil = soil_true + self.model.rng.normal(0, self.noise_std * 0.5)
        rain = rain_true + self.model.rng.normal(0, self.noise_std * 0.3)
        
        water = np.clip(water, 0, 2.0)
        soil = np.clip(soil, 0, 1.0)
        rain = np.clip(rain, 0, 1.0)
        
        if self.last_reading is not None:
            self.trend_rising = water > self.last_reading.get('water', water)
        
        self.last_reading = {
            'water': water,
            'soil': soil,
            'rain': rain,
            'step': env.current_step
        }
    
    def get_reading(self) -> Optional[Dict[str, float]]:
        """Return last reading or None if dropped out."""
        return self.last_reading if self.is_active else None


class EdgeAggregatorAgent(Agent):
    """
    Edge aggregator agent that:
    1. Collects readings from zone sensors
    2. Extracts temporal features
    3. Runs ML model for risk prediction
    4. Applies guardrails state machine
    5. Reports status to coordinator
    """
    
    def __init__(self, unique_id: int, model, zone_id: int, config: dict):
        super().__init__(unique_id, model)
        self.zone_id = zone_id
        self.config = config
        
        guard_config = GuardrailsConfig.from_dict(config['guardrails'])
        guard_config.cooldown_steps = config.get('countermeasures', {}).get('cooldown_steps', 10)
        
        self.state_machine = AlertStateMachine(guard_config)
        self.feature_extractor = FeatureExtractor()
        self.outlier_clipper = OutlierClipper(config['sensors']['outlier_clip'])
        self.missing_handler = MissingValueHandler()
        
        self.ml_model = None
        self.sensors: List[SensorAgent] = []
        
        self.current_risk = 0.0
        self.current_state = AlertState.NORMAL
        self.current_features: Dict[str, float] = {}
        self.health = 1.0
        self.consensus = 0.0
        
        self.history: List[Dict] = []
    
    def set_ml_model(self, model):
        """Set the ML model for risk prediction."""
        self.ml_model = model
    
    def add_sensor(self, sensor: SensorAgent):
        """Register a sensor with this edge aggregator."""
        self.sensors.append(sensor)
    
    def step(self):
        """Process sensor data, extract features, predict risk, update state."""
        readings = []
        active_count = 0
        rising_count = 0
        total_penalty = 0.0

        for sensor in self.sensors:
            reading = sensor.get_reading()
            if reading is not None:
                processed_water, penalty = self.missing_handler.process(
                    sensor.unique_id, reading['water']
                )
                reading['water'] = self.outlier_clipper.clip(processed_water)
                readings.append(reading)
                active_count += 1
                total_penalty += penalty
                if sensor.trend_rising:
                    rising_count += 1
            else:
                _, penalty = self.missing_handler.process(sensor.unique_id, None)
                total_penalty += penalty

        # Base health from active sensor ratio, reduced by carry-forward penalties
        base_health = active_count / len(self.sensors) if self.sensors else 0.0
        avg_penalty = total_penalty / len(self.sensors) if self.sensors else 0.0
        self.health = max(0.0, base_health - avg_penalty)
        self.consensus = rising_count / active_count if active_count > 0 else 0.0
        
        if readings:
            water_mean = np.mean([r['water'] for r in readings])
            rain_mean = np.mean([r['rain'] for r in readings])
            soil_mean = np.mean([r['soil'] for r in readings])
            
            self.feature_extractor.update(water_mean, rain_mean, soil_mean)
        
        self.current_features = self.feature_extractor.extract(self.consensus, self.health)
        
        if self.ml_model is not None:
            feature_names = self.config.get('ml', {}).get('features', [
                'water_mean_5', 'water_slope_5', 'water_max_10',
                'rain_sum_20', 'rain_mean_10', 'soil_mean_10',
                'consensus', 'health'
            ])
            feature_vector = np.array([[
                self.current_features[f] for f in feature_names
            ]])
            self.current_risk = self.ml_model.predict_proba(feature_vector)[0, 1]
        else:
            self.current_risk = self._heuristic_risk()
        
        self.current_state = self.state_machine.update(
            self.current_risk, self.consensus, self.health
        )
        
        self.history.append({
            'step': self.model.environment.current_step,
            'zone_id': self.zone_id,
            'risk': self.current_risk,
            'state': self.current_state.value,
            'health': self.health,
            'consensus': self.consensus,
            **self.current_features
        })
    
    def _heuristic_risk(self) -> float:
        """Fallback heuristic risk when no ML model available."""
        water = self.current_features.get('water_mean_5', 0)
        rain = self.current_features.get('rain_sum_20', 0)
        slope = self.current_features.get('water_slope_5', 0)
        
        risk = 0.3 * water + 0.3 * rain + 0.4 * max(0, slope * 10)
        return np.clip(risk, 0, 1)
    
    def get_status(self) -> Dict[str, Any]:
        """Return current status for coordinator."""
        return {
            'zone_id': self.zone_id,
            'risk': self.current_risk,
            'state': self.current_state,
            'health': self.health,
            'consensus': self.consensus,
            'state_changes': self.state_machine.state_changes
        }
    
    def reset(self):
        """Reset agent state."""
        self.state_machine.reset()
        self.feature_extractor.reset()
        self.outlier_clipper.reset()
        self.missing_handler.reset()
        self.current_risk = 0.0
        self.current_state = AlertState.NORMAL
        self.history.clear()


class CoordinatorAgent(Agent):
    """
    Coordinator agent that:
    1. Collects status from all edge aggregators
    2. Fuses zone risks into global assessment
    3. Manages global alarm state
    4. Logs decisions
    """
    
    def __init__(self, unique_id: int, model, config: dict):
        super().__init__(unique_id, model)
        self.config = config
        self.edges: List[EdgeAggregatorAgent] = []
        
        self.global_risk = 0.0
        self.global_alarm = False
        self.zones_in_alert: List[int] = []
        
        self.history: List[Dict] = []
    
    def add_edge(self, edge: EdgeAggregatorAgent):
        """Register an edge aggregator."""
        self.edges.append(edge)
    
    def step(self):
        """Collect edge statuses and compute global assessment."""
        statuses = [edge.get_status() for edge in self.edges]
        
        risks = [s['risk'] for s in statuses]
        self.global_risk = max(risks) if risks else 0.0
        
        self.zones_in_alert = [
            s['zone_id'] for s in statuses 
            if s['state'] == AlertState.ALERT
        ]
        
        self.global_alarm = len(self.zones_in_alert) > 0
        
        self.history.append({
            'step': self.model.environment.current_step,
            'global_risk': self.global_risk,
            'global_alarm': self.global_alarm,
            'num_zones_in_alert': len(self.zones_in_alert),
            'zones_in_alert': list(self.zones_in_alert),
            'zone_risks': {s['zone_id']: s['risk'] for s in statuses},
            'zone_states': {s['zone_id']: s['state'].value for s in statuses}
        })
    
    def get_global_status(self) -> Dict[str, Any]:
        """Return global system status."""
        return {
            'global_risk': self.global_risk,
            'global_alarm': self.global_alarm,
            'zones_in_alert': self.zones_in_alert,
            'num_edges': len(self.edges)
        }
    
    def reset(self):
        """Reset coordinator state."""
        self.global_risk = 0.0
        self.global_alarm = False
        self.zones_in_alert.clear()
        self.history.clear()


class MitigationAgent(Agent):
    """
    Mitigation agent that executes countermeasures when zone enters ALERT.
    Actions: Pump (reduce water) and Gate (reduce upstream inflow).
    """
    
    def __init__(self, unique_id: int, model, zone_id: int, config: dict):
        super().__init__(unique_id, model)
        self.zone_id = zone_id
        self.config = config
        self.enabled = config.get('countermeasures', {}).get('enabled', False)
        self.pump_capacity = config.get('countermeasures', {}).get('pump_capacity', 0.05)
        self.gate_reduction = config.get('countermeasures', {}).get('gate_reduction', 0.3)
        self.cooldown_steps = config.get('countermeasures', {}).get('cooldown_steps', 10)
        
        self.pump_active = False
        self.gate_active = False
        self.cooldown_remaining = 0
        
        self.history: List[Dict] = []
    
    def step(self):
        """Execute countermeasures based on zone state."""
        if not self.enabled:
            return
        
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return
        
        edge = self._get_zone_edge()
        if edge is None:
            return
        
        if edge.current_state == AlertState.ALERT:
            self._activate_pump()
            # Activate gate for river zones (upstream inflow control)
            zone = self.model.environment.zones[self.zone_id]
            if zone.is_river_zone:
                self._activate_gate()
        elif edge.current_state == AlertState.NORMAL:
            self._deactivate_all()
    
    def _get_zone_edge(self) -> Optional[EdgeAggregatorAgent]:
        """Find the edge aggregator for this zone."""
        return self.model.edges.get(self.zone_id)
    
    def _activate_pump(self):
        """Activate pump to reduce water level."""
        if not self.pump_active:
            self.pump_active = True
            self.history.append({
                'step': self.model.environment.current_step,
                'action': 'pump_on',
                'zone_id': self.zone_id
            })
        self.model.environment.apply_pump(self.zone_id, self.pump_capacity)
    
    def _activate_gate(self):
        """Activate gate to reduce upstream inflow."""
        if not self.gate_active:
            self.gate_active = True
            self.history.append({
                'step': self.model.environment.current_step,
                'action': 'gate_on',
                'zone_id': self.zone_id
            })
        self.model.environment.apply_gate(self.gate_reduction)
    
    def _deactivate_all(self):
        """Deactivate all countermeasures."""
        if self.pump_active or self.gate_active:
            self.pump_active = False
            self.gate_active = False
            self.cooldown_remaining = self.cooldown_steps
            self.history.append({
                'step': self.model.environment.current_step,
                'action': 'all_off',
                'zone_id': self.zone_id
            })
    
    def reset(self):
        """Reset mitigation agent state."""
        self.pump_active = False
        self.gate_active = False
        self.cooldown_remaining = 0
        self.history.clear()
