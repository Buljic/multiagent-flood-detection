"""
Experiment runner for comparing MAS flood detection with baseline.
Runs multiple scenarios and collects comprehensive metrics.
"""

import argparse
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import logging
from tqdm import tqdm

import sys
import hashlib
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.model import FloodModel
from baseline.threshold import ZonedThresholdBaseline
from eval.metrics import MetricsCalculator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Runs experiments comparing MAS flood detection with baseline across scenarios.
    """
    
    def __init__(self, config: dict, ml_model=None):
        self.config = config
        self.ml_model = ml_model
        self.metrics_calc = MetricsCalculator()
    
    def run_scenario(self, scenario: dict, steps: int = 400,
                     seed: int = 42) -> Dict:
        """
        Run single scenario comparing MAS and baseline.
        
        Args:
            scenario: Scenario configuration dict
            steps: Number of simulation steps
            seed: Random seed
            
        Returns:
            Dict with scenario results
        """
        scenario_config = self.config.copy()
        
        if 'dropout_rate' in scenario:
            scenario_config['sensors']['dropout_rate'] = scenario['dropout_rate']
        if 'noise_level' in scenario:
            noise_levels = self.config.get('noise_levels', {})
            if scenario['noise_level'] in noise_levels:
                scenario_config['sensors']['noise_std'] = \
                    noise_levels[scenario['noise_level']]['sensor_noise_std']
        
        soil_init = scenario.get('soil_saturation_init', 0.3)
        rainfall_type = scenario.get('rainfall_type', 'normal')
        
        mas_model = FloodModel(scenario_config, ml_model=self.ml_model, seed=seed)
        mas_model.reset(soil_init)
        mas_model.generate_random_rainfall(rainfall_type)
        
        baseline = ZonedThresholdBaseline(
            num_zones=scenario_config['simulation']['num_zones'],
            config=scenario_config
        )
        
        for step in range(steps):
            mas_model.step()
            
            zone_readings = {}
            for zone_id, edge in mas_model.edges.items():
                zone_readings[zone_id] = {
                    'water': edge.current_features.get('water_mean_5', 0),
                    'rain': edge.current_features.get('rain_sum_20', 0) / 20
                }
            baseline.update(zone_readings, step)
        
        mas_logs = mas_model.get_logs()
        
        baseline_records = []
        for zone_id in range(scenario_config['simulation']['num_zones']):
            zone_history = baseline.get_zone_history(zone_id)
            for _, row in zone_history.iterrows():
                flood_status = mas_model.environment.is_flooded(zone_id)
                baseline_records.append({
                    'step': row['step'],
                    'zone_id': zone_id,
                    'state': row['state'],
                    'ground_truth_flooded': flood_status
                })
        baseline_logs = pd.DataFrame(baseline_records)
        
        mas_metrics = self.metrics_calc.compute_from_logs(mas_logs)
        baseline_metrics = self.metrics_calc.compute_from_logs(baseline_logs)
        
        return {
            'scenario': scenario,
            'mas': mas_metrics,
            'baseline': baseline_metrics,
            'mas_state_changes': sum(e.state_machine.state_changes 
                                     for e in mas_model.edges.values()),
            'baseline_state_changes': baseline.get_total_state_changes()
        }
    
    def run_all_scenarios(self, scenarios: List[dict], 
                          steps: int = 400,
                          repeats: int = 3) -> Dict:
        """
        Run all scenarios with multiple repeats.
        
        Args:
            scenarios: List of scenario configurations
            steps: Steps per scenario
            repeats: Number of repeats per scenario
            
        Returns:
            Dict with aggregated results
        """
        all_results = []
        
        for scenario in tqdm(scenarios, desc="Running scenarios"):
            scenario_results = []
            
            for rep in range(repeats):
                seed = 42 + rep * 1000
                result = self.run_scenario(scenario, steps, seed)
                result['repeat'] = rep
                scenario_results.append(result)
            
            all_results.append({
                'scenario_name': scenario.get('name', 'unnamed'),
                'scenario_config': scenario,
                'runs': scenario_results,
                'aggregated': self._aggregate_results(scenario_results)
            })
        
        summary = self._compute_summary(all_results)
        
        # Add run metadata for reproducibility
        run_metadata = self._generate_run_metadata(scenarios, repeats)
        
        return {
            'run_metadata': run_metadata,
            'scenarios': all_results,
            'summary': summary
        }
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results from multiple runs."""
        agg = {
            'mas': {'detection': {}, 'stability': {}, 'lead_time': {}},
            'baseline': {'detection': {}, 'stability': {}, 'lead_time': {}}
        }
        
        for system in ['mas', 'baseline']:
            for category in ['detection', 'stability', 'lead_time']:
                if category not in results[0].get(system, {}):
                    continue
                    
                metrics = results[0][system][category].keys()
                for metric in metrics:
                    if metric == 'confusion_matrix':
                        continue
                    values = [r[system][category].get(metric, 0) for r in results
                             if category in r.get(system, {})]
                    if values:
                        agg[system][category][metric] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values))
                        }
        
        return agg
    
    def _generate_run_metadata(self, scenarios: List[dict], repeats: int) -> Dict:
        """Generate metadata for reproducibility."""
        # Compute config hash for versioning
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Model hash if available
        model_hash = None
        if self.ml_model is not None:
            model_bytes = joblib.hash(self.ml_model)
            model_hash = model_bytes[:8] if model_bytes else None
        
        return {
            'timestamp': datetime.now().isoformat(),
            'seed': self.config.get('seed', 42),
            'config_hash': config_hash,
            'model_hash': model_hash,
            'num_scenarios': len(scenarios),
            'repeats_per_scenario': repeats,
            'guardrails_params': self.config.get('guardrails', {}),
            'baseline_params': self.config.get('baseline', {}),
            'sensor_params': {
                'noise_std': self.config['sensors']['noise_std'],
                'dropout_rate': self.config['sensors']['dropout_rate'],
                'per_zone': self.config['sensors']['per_zone']
            },
            'ml_horizon_T': self.config['ml']['horizon_T']
        }
    
    def _compute_summary(self, all_results: List[Dict]) -> Dict:
        """Compute overall summary across all scenarios."""
        mas_f1s = []
        baseline_f1s = []
        mas_stability = []
        baseline_stability = []
        
        for scenario in all_results:
            agg = scenario['aggregated']
            if 'f1' in agg['mas'].get('detection', {}):
                mas_f1s.append(agg['mas']['detection']['f1']['mean'])
            if 'f1' in agg['baseline'].get('detection', {}):
                baseline_f1s.append(agg['baseline']['detection']['f1']['mean'])
            if 'total_state_changes' in agg['mas'].get('stability', {}):
                mas_stability.append(agg['mas']['stability']['total_state_changes']['mean'])
            if 'total_state_changes' in agg['baseline'].get('stability', {}):
                baseline_stability.append(agg['baseline']['stability']['total_state_changes']['mean'])
        
        return {
            'mas_avg_f1': float(np.mean(mas_f1s)) if mas_f1s else 0,
            'baseline_avg_f1': float(np.mean(baseline_f1s)) if baseline_f1s else 0,
            'f1_improvement': float(np.mean(mas_f1s) - np.mean(baseline_f1s)) if mas_f1s and baseline_f1s else 0,
            'mas_avg_state_changes': float(np.mean(mas_stability)) if mas_stability else 0,
            'baseline_avg_state_changes': float(np.mean(baseline_stability)) if baseline_stability else 0,
            'stability_improvement': float(np.mean(baseline_stability) - np.mean(mas_stability)) if mas_stability and baseline_stability else 0,
            'num_scenarios': len(all_results)
        }


def main():
    parser = argparse.ArgumentParser(description='Run flood detection experiments')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to base configuration')
    parser.add_argument('--scenarios-config', type=str, default='configs/scenarios.yaml',
                        help='Path to scenarios configuration')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained ML model')
    parser.add_argument('--baseline', type=str, default='threshold',
                        help='Baseline type')
    parser.add_argument('--out', type=str, default='outputs/experiments/results.json',
                        help='Output path for results')
    parser.add_argument('--steps', type=int, default=400,
                        help='Steps per scenario')
    parser.add_argument('--repeats', type=int, default=3,
                        help='Repeats per scenario')
    
    args = parser.parse_args()
    
    with open(args.scenarios_config, 'r') as f:
        scenarios_config = yaml.safe_load(f)
    
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    base_config.update({k: v for k, v in scenarios_config.items() if k != 'scenarios'})
    
    ml_model = None
    if args.model and Path(args.model).exists():
        ml_model = joblib.load(args.model)
        logger.info(f"Loaded ML model from {args.model}")
    
    runner = ExperimentRunner(base_config, ml_model=ml_model)
    
    scenarios = scenarios_config.get('scenarios', [])
    logger.info(f"Running {len(scenarios)} scenarios with {args.repeats} repeats each")
    
    results = runner.run_all_scenarios(
        scenarios=scenarios,
        steps=args.steps,
        repeats=args.repeats
    )
    
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_path}")
    
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    summary = results['summary']
    logger.info(f"MAS Average F1: {summary['mas_avg_f1']:.4f}")
    logger.info(f"Baseline Average F1: {summary['baseline_avg_f1']:.4f}")
    logger.info(f"F1 Improvement: {summary['f1_improvement']:.4f}")
    logger.info(f"MAS Avg State Changes: {summary['mas_avg_state_changes']:.1f}")
    logger.info(f"Baseline Avg State Changes: {summary['baseline_avg_state_changes']:.1f}")
    logger.info(f"Stability Improvement: {summary['stability_improvement']:.1f}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
