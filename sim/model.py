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

        # Record per-step ground truth for each zone (must happen DURING
        # simulation, not after, to capture the true flood status at each step)
        for zone_id, edge in self.edges.items():
            if edge.history:
                edge.history[-1]['ground_truth_flooded'] = \
                    self.environment.is_flooded(zone_id)

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
        """Compile logs from all agents into DataFrame.

        Ground truth ``ground_truth_flooded`` is recorded per-step during
        ``step()`` so that it reflects the actual flood status at each
        simulation step rather than only the final state.
        """
        records = []

        for zone_id, edge in self.edges.items():
            for entry in edge.history:
                records.append(entry)

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


def steps_to_real_time(steps: int, config: dict) -> str:
    """Convert simulation steps to human-readable real-world time.

    Uses ``step_duration_minutes`` from *config* (default 15 min per step).
    Returns a string like ``"2h 30min"`` or ``"4d 4h 0min"``.
    """
    minutes_per_step = config.get('step_duration_minutes', 15)
    total_minutes = steps * minutes_per_step
    days, remainder = divmod(total_minutes, 1440)
    hours, minutes = divmod(remainder, 60)
    if days > 0:
        return f"{int(days)}d {int(hours)}h {int(minutes)}min"
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}min"
    return f"{int(minutes)}min"


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Run FloodMAS simulation')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--scenarios-file', type=str, default='configs/scenarios.yaml',
                        help='Path to scenarios configuration file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained ML model (pkl)')
    parser.add_argument('--log', type=str, default='outputs/logs/run.parquet',
                        help='Path to output log file')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of simulation steps')
    parser.add_argument('--scenario', type=str, default='normal_wet',
                        help='Scenario name (from scenarios.yaml) or rainfall type (normal/extreme)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Load and validate scenario
    scenarios_config = {}
    if Path(args.scenarios_file).exists():
        scenarios_config = load_config(args.scenarios_file)
    
    scenario_name = args.scenario
    rainfall_type = None
    soil_init = None
    
    # Check if scenario is a named scenario from scenarios.yaml
    if 'scenarios' in scenarios_config:
        scenario_list = scenarios_config['scenarios']
        scenario_names = [s['name'] for s in scenario_list]
        
        if scenario_name in scenario_names:
            # Use named scenario
            scenario_data = next(s for s in scenario_list if s['name'] == scenario_name)
            rainfall_type = scenario_data.get('rainfall_type', 'normal')
            soil_init = scenario_data.get('soil_saturation_init', 0.3)
            
            # Apply scenario-specific sensor config
            if 'dropout_rate' in scenario_data:
                config['sensors']['dropout_rate'] = scenario_data['dropout_rate']
            if 'noise_level' in scenario_data:
                noise_levels = scenarios_config.get('noise_levels', {})
                if scenario_data['noise_level'] in noise_levels:
                    config['sensors']['noise_std'] = noise_levels[scenario_data['noise_level']]['sensor_noise_std']
            
            logger.info(f"Using named scenario '{scenario_name}': rainfall={rainfall_type}, soil_init={soil_init}, dropout={config['sensors']['dropout_rate']}, noise={config['sensors']['noise_std']}")
        elif scenario_name in ['normal', 'extreme']:
            # Backward compatibility: use as rainfall type
            rainfall_type = scenario_name
            soil_init = 0.3
            logger.info(f"Using rainfall type '{rainfall_type}' (backward compatibility mode)")
        else:
            # Invalid scenario
            logger.error(f"Invalid scenario '{scenario_name}'")
            logger.error(f"Valid scenarios: {', '.join(scenario_names + ['normal', 'extreme'])}")
            return
    else:
        # No scenarios.yaml, use scenario as rainfall type
        if scenario_name in ['normal', 'extreme']:
            rainfall_type = scenario_name
            soil_init = 0.3
        else:
            logger.error(f"Scenario file not found and '{scenario_name}' is not a valid rainfall type (normal/extreme)")
            return
    
    ml_model = None
    if args.model and Path(args.model).exists():
        ml_model = joblib.load(args.model)
        logger.info(f"Loaded ML model from {args.model}")

    model = FloodModel(config, ml_model=ml_model, seed=config.get('seed', 42))
    if soil_init is not None:
        model.reset(soil_init)
    model.generate_random_rainfall(rainfall_type)

    steps = args.steps or config['simulation']['steps_per_episode']
    real_time = steps_to_real_time(steps, config)
    logger.info(f"Running simulation for {steps} steps (~{real_time} real-world) with scenario '{args.scenario}'")

    # Run MAS + baseline in parallel, recording per-step ground truth
    from baseline.threshold import ZonedThresholdBaseline
    num_zones = config['simulation']['num_zones']
    baseline = ZonedThresholdBaseline(num_zones=num_zones, config=config)
    per_step_gt = {}

    for step in range(steps):
        model.step()

        for zone_id in range(num_zones):
            per_step_gt[(step, zone_id)] = model.environment.is_flooded(zone_id)

        zone_readings = {}
        for zone_id, edge in model.edges.items():
            zone_readings[zone_id] = {
                'water': edge.current_features.get('water_mean_5', 0),
                'rain': edge.current_features.get('rain_sum_20', 0) / 20
            }
        baseline.update(zone_readings, step)

    logs = model.get_logs()

    output_path = Path(args.log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logs.to_parquet(output_path, index=False)
    logger.info(f"Saved logs to {output_path}")

    coord_logs = model.get_coordinator_logs()
    coord_path = output_path.parent / f"{output_path.stem}_coordinator.parquet"
    coord_logs.to_parquet(coord_path, index=False)
    logger.info(f"Saved coordinator logs to {coord_path}")

    # Save baseline logs with per-step ground truth
    baseline_records = []
    for zone_id in range(num_zones):
        zone_history = baseline.get_zone_history(zone_id)
        for _, row in zone_history.iterrows():
            baseline_records.append({
                'step': row['step'],
                'zone_id': zone_id,
                'state': row['state'],
                'ground_truth_flooded': per_step_gt.get(
                    (row['step'], zone_id), False)
            })
    baseline_logs = pd.DataFrame(baseline_records)
    baseline_path = output_path.parent / f"{output_path.stem}_baseline.parquet"
    baseline_logs.to_parquet(baseline_path, index=False)
    logger.info(f"Saved baseline logs to {baseline_path}")

    global_status = model.coordinator.get_global_status()
    logger.info(f"Final status: {global_status}")


if __name__ == '__main__':
    main()
