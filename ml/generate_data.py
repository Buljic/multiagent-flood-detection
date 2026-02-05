"""
Synthetic data generation for ML model training.
Runs multiple episodes of simulation and extracts labeled feature vectors.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.environment import FloodEnvironment
from sim.guardrails import RingBuffer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGenerator:
    """
    Generates synthetic training data from flood simulation episodes.
    
    For each step in each episode:
    - Extracts features (water, rain, soil statistics)
    - Computes ground truth label: flood_in_next_T (will zone flood in next T steps?)
    """
    
    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.horizon_T = config['ml']['horizon_T']
        
    def generate(self, num_episodes: int, steps_per_episode: int,
                 output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate dataset from multiple simulation episodes.
        
        Args:
            num_episodes: Number of episodes to simulate
            steps_per_episode: Steps per episode
            output_path: Optional path to save dataset
            
        Returns:
            DataFrame with features and labels
        """
        all_records = []
        
        scenarios = ['normal', 'extreme']
        
        for episode in tqdm(range(num_episodes), desc="Generating episodes"):
            episode_seed = self.seed + episode
            scenario = self.rng.choice(scenarios, p=[0.4, 0.6])
            
            soil_init = self.rng.uniform(0.1, 0.8)
            dropout_rate = self.rng.choice([0.0, 0.1, 0.2, 0.3], p=[0.4, 0.3, 0.2, 0.1])
            noise_std = self.rng.choice([0.03, 0.05, 0.1], p=[0.5, 0.3, 0.2])
            
            records = self._run_episode(
                episode_id=episode,
                seed=episode_seed,
                scenario=scenario,
                soil_init=soil_init,
                dropout_rate=dropout_rate,
                noise_std=noise_std,
                steps=steps_per_episode
            )
            all_records.extend(records)
        
        df = pd.DataFrame(all_records)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved dataset with {len(df)} samples to {output_path}")
        
        return df
    
    def _run_episode(self, episode_id: int, seed: int, scenario: str,
                     soil_init: float, dropout_rate: float, noise_std: float,
                     steps: int) -> List[Dict]:
        """Run single episode and collect labeled data."""
        rng = np.random.default_rng(seed)
        
        config = self.config.copy()
        config['sensors']['dropout_rate'] = dropout_rate
        config['sensors']['noise_std'] = noise_std
        
        env = FloodEnvironment(config, seed)
        env.reset(soil_init)
        env.generate_random_rainfall(scenario)
        
        zone_buffers = {
            z.zone_id: {
                'water': RingBuffer(50),
                'rain': RingBuffer(50),
                'soil': RingBuffer(50)
            }
            for z in env.zones
        }
        
        future_flood = {z.zone_id: [] for z in env.zones}
        all_states = []
        
        for step in range(steps):
            state = env.step()
            
            zone_states = {}
            for zone in env.zones:
                zs = env.get_zone_state(zone.zone_id)
                zone_states[zone.zone_id] = {
                    'water': zs['water_mean'],
                    'rain': state['rainfall'],
                    'soil': zs['soil_mean'],
                    'flooded': env.is_flooded(zone.zone_id)
                }
                
                zone_buffers[zone.zone_id]['water'].append(zs['water_mean'])
                zone_buffers[zone.zone_id]['rain'].append(state['rainfall'])
                zone_buffers[zone.zone_id]['soil'].append(zs['soil_mean'])
            
            all_states.append(zone_states)
        
        for zone in env.zones:
            for step in range(steps):
                future_steps = all_states[step + 1: step + 1 + self.horizon_T]
                will_flood = any(s[zone.zone_id]['flooded'] for s in future_steps)
                future_flood[zone.zone_id].append(will_flood)
        
        records = []
        min_warmup = 20
        
        for zone in env.zones:
            water_buf = RingBuffer(50)
            rain_buf = RingBuffer(50)
            soil_buf = RingBuffer(50)
            
            for step in range(steps - self.horizon_T):
                zs = all_states[step][zone.zone_id]
                water_buf.append(zs['water'])
                rain_buf.append(zs['rain'])
                soil_buf.append(zs['soil'])
                
                if step < min_warmup:
                    continue
                
                active_sensors = int((1 - dropout_rate) * self.config['sensors']['per_zone'])
                total_sensors = self.config['sensors']['per_zone']
                health = active_sensors / total_sensors if total_sensors > 0 else 1.0
                
                water_slope = water_buf.slope(5)
                consensus = 0.5 + 0.5 * np.tanh(water_slope * 10) + rng.normal(0, 0.1)
                consensus = np.clip(consensus, 0, 1)
                
                features = {
                    'episode_id': episode_id,
                    'step': step,
                    'zone_id': zone.zone_id,
                    'scenario': scenario,
                    'dropout_rate': dropout_rate,
                    'noise_std': noise_std,
                    'water_mean_5': water_buf.mean(5),
                    'water_slope_5': water_slope,
                    'water_max_10': water_buf.max(10),
                    'rain_sum_20': rain_buf.sum(20),
                    'rain_mean_10': rain_buf.mean(10),
                    'soil_mean_10': soil_buf.mean(10),
                    'consensus': consensus,
                    'health': health,
                    'flood_in_next_T': int(future_flood[zone.zone_id][step])
                }
                records.append(features)
        
        return records


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic training data')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of simulation episodes')
    parser.add_argument('--steps', type=int, default=400,
                        help='Steps per episode')
    parser.add_argument('--out', type=str, default='outputs/datasets/sim.parquet',
                        help='Output path for dataset')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    seed = args.seed if args.seed is not None else config.get('seed', 42)
    
    generator = DataGenerator(config, seed=seed)
    df = generator.generate(
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        output_path=args.out
    )
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Positive rate: {df['flood_in_next_T'].mean():.3f}")
    logger.info(f"Features: {list(df.columns)}")


if __name__ == '__main__':
    main()
