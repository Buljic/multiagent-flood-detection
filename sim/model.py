"""
FloodModel: Main Mesa model orchestrating the multi-agent flood detection simulation.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from mesa import Model, Agent
from typing import Optional, Dict, List, Any
import joblib
import logging

from .environment import FloodEnvironment
from .agents import SensorAgent, EdgeAggregatorAgent, CoordinatorAgent, MitigationAgent
from .guardrails import AlertState


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FloodModel(Model):
    """
    Mesa model for multi-agent flood early warning system.
    
    Architecture:
    - Environment: Simulates hydro-meteorological conditions
    - SensorAgents: Emit noisy readings per zone
    - EdgeAggregatorAgents: Feature extraction + ML + guardrails per zone
    - CoordinatorAgent: Global fusion and alarm
    - MitigationAgents: Optional countermeasures
    """
    
    def __init__(self, config: dict, ml_model=None, seed: Optional[int] = None):
        super().__init__()
        self.config = config
        self.seed = seed if seed is not None else config.get('seed', 42)
        self.rng = np.random.default_rng(self.seed)
        
        self.environment = FloodEnvironment(config, self.seed)
        
        self.ml_model = ml_model
        self._agent_list: List[Any] = []
        
        self._create_agents()
        
        self.running = True
        self.step_count = 0
    
    def _create_agents(self):
        """Create all agents in the system."""
        agent_id = 0
        
        self.coordinator = CoordinatorAgent(agent_id, self, self.config)
        self._agent_list.append(self.coordinator)
        agent_id += 1
        
        self.edges: Dict[int, EdgeAggregatorAgent] = {}
        self.sensors: List[SensorAgent] = []
        self.mitigations: Dict[int, MitigationAgent] = {}
        
        for zone in self.environment.zones:
            edge = EdgeAggregatorAgent(agent_id, self, zone.zone_id, self.config)
            if self.ml_model is not None:
                edge.set_ml_model(self.ml_model)
            self._agent_list.append(edge)
            self.edges[zone.zone_id] = edge
            self.coordinator.add_edge(edge)
            agent_id += 1
            
            sensors_per_zone = self.config['sensors']['per_zone']
            zone_cells = zone.cells
            sensor_cells = self.rng.choice(
                len(zone_cells), 
                min(sensors_per_zone, len(zone_cells)), 
                replace=False
            )
            
            for idx in sensor_cells:
                cell = zone_cells[idx]
                sensor = SensorAgent(agent_id, self, zone.zone_id, cell, self.config)
                self._agent_list.append(sensor)
                self.sensors.append(sensor)
                edge.add_sensor(sensor)
                agent_id += 1
            
            if self.config.get('countermeasures', {}).get('enabled', False):
                mitigation = MitigationAgent(agent_id, self, zone.zone_id, self.config)
                self._agent_list.append(mitigation)
                self.mitigations[zone.zone_id] = mitigation
                agent_id += 1
    
    def step(self):
        """Advance simulation by one step."""
        self.environment.step()
        for agent in self.sensors:
            agent.step()
        for edge in self.edges.values():
            edge.step()
        self.coordinator.step()
        for mitigation in self.mitigations.values():
            mitigation.step()
        self.step_count += 1
    
    def run(self, steps: int) -> pd.DataFrame:
        """
        Run simulation for specified number of steps.
        
        Returns:
            DataFrame with per-step logs
        """
        for _ in range(steps):
            self.step()
        
        return self.get_logs()
    
    def get_logs(self) -> pd.DataFrame:
        """Compile logs from all agents into DataFrame."""
        records = []
        
        for zone_id, edge in self.edges.items():
            for entry in edge.history:
                flood_status = self.environment.is_flooded(zone_id)
                records.append({
                    **entry,
                    'ground_truth_flooded': flood_status
                })
        
        if records:
            return pd.DataFrame(records)
        return pd.DataFrame()
    
    def get_coordinator_logs(self) -> pd.DataFrame:
        """Get coordinator-level logs."""
        return pd.DataFrame(self.coordinator.history)
    
    def reset(self, soil_saturation_init: Optional[float] = None):
        """Reset simulation to initial state."""
        self.environment.reset(soil_saturation_init)
        self.coordinator.reset()
        for edge in self.edges.values():
            edge.reset()
        for mitigation in self.mitigations.values():
            mitigation.reset()
        self.step_count = 0
    
    def add_rainfall_event(self, intensity: float, duration: int, start_step: int):
        """Add rainfall event to environment."""
        self.environment.add_rainfall_event(intensity, duration, start_step)
    
    def generate_random_rainfall(self, scenario: str = 'normal'):
        """Generate random rainfall based on scenario."""
        self.environment.generate_random_rainfall(scenario)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Run FloodMAS simulation')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained ML model (pkl)')
    parser.add_argument('--log', type=str, default='outputs/logs/run.parquet',
                        help='Path to output log file')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of simulation steps')
    parser.add_argument('--scenario', type=str, default='normal',
                        choices=['normal', 'extreme'],
                        help='Rainfall scenario')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    ml_model = None
    if args.model and Path(args.model).exists():
        ml_model = joblib.load(args.model)
        logger.info(f"Loaded ML model from {args.model}")
    
    model = FloodModel(config, ml_model=ml_model)
    model.generate_random_rainfall(args.scenario)
    
    steps = args.steps or config['simulation']['steps_per_episode']
    logger.info(f"Running simulation for {steps} steps with scenario '{args.scenario}'")
    
    logs = model.run(steps)
    
    output_path = Path(args.log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logs.to_parquet(output_path, index=False)
    logger.info(f"Saved logs to {output_path}")
    
    coord_logs = model.get_coordinator_logs()
    coord_path = output_path.parent / f"{output_path.stem}_coordinator.parquet"
    coord_logs.to_parquet(coord_path, index=False)
    logger.info(f"Saved coordinator logs to {coord_path}")
    
    global_status = model.coordinator.get_global_status()
    logger.info(f"Final status: {global_status}")


if __name__ == '__main__':
    main()
