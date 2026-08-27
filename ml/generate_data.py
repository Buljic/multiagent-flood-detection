"""
Synthetic data generation for ML model training.
Runs multiple episodes of simulation and extracts labeled feature vectors.
"""

import argparse
import copy
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

_project_root = str(Path(__file__).parent.parent)
import sys
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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
        """Run single episode and collect labeled data.

        Simulates actual sensor noise/dropout per step to produce realistic
        consensus and health values matching runtime behaviour.
        """
        rng = np.random.default_rng(seed)

        config = copy.deepcopy(self.config)
        config['sensors']['dropout_rate'] = dropout_rate
        config['sensors']['noise_std'] = noise_std

        env = FloodEnvironment(config, seed)
        env.reset(soil_init)
        env.generate_random_rainfall(scenario)

        sensors_per_zone = config['sensors']['per_zone']

        future_flood = {z.zone_id: [] for z in env.zones}
        all_states = []

        # Per-zone sensor state: last water reading per sensor for trend detection
        zone_sensor_prev = {
            z.zone_id: [None] * sensors_per_zone for z in env.zones
        }
        zone_sensor_carry = {
            z.zone_id: [0] * sensors_per_zone for z in env.zones
        }
        # Runtime edge uses ONE OutlierClipper per zone (max_delta = outlier_clip)
        zone_clipper_last = {z.zone_id: None for z in env.zones}
        outlier_clip = config['sensors'].get('outlier_clip', 0.3)

        # Per-zone tracking of consensus/health per step
        zone_step_meta = {z.zone_id: [] for z in env.zones}

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

                # Simulate actual sensors to compute consensus & health
                active_count = 0
                rising_count = 0
                missing_penalty = 0.0
                for s_idx in range(sensors_per_zone):
                    prev = zone_sensor_prev[zone.zone_id][s_idx]
                    # Simulate dropout
                    if rng.random() < dropout_rate:
                        if prev is None:
                            missing_penalty += 1.0
                        else:
                            carry = zone_sensor_carry[zone.zone_id][s_idx] + 1
                            zone_sensor_carry[zone.zone_id][s_idx] = carry
                            missing_penalty += (carry / 5.0) * 0.5 if carry <= 5 else 1.0
                        zone_sensor_prev[zone.zone_id][s_idx] = None
                        continue
                    active_count += 1
                    zone_sensor_carry[zone.zone_id][s_idx] = 0
                    # Simulate noisy reading
                    noisy_water = zs['water_mean'] + rng.normal(0, noise_std)
                    noisy_water = np.clip(noisy_water, 0, 2.0)
                    # Check trend
                    if prev is not None and noisy_water > prev:
                        rising_count += 1
                    zone_sensor_prev[zone.zone_id][s_idx] = noisy_water

                health = max(0.0, (active_count - missing_penalty) / sensors_per_zone) if sensors_per_zone > 0 else 1.0
                consensus = rising_count / active_count if active_count > 0 else 0.0

                zone_step_meta[zone.zone_id].append({
                    'consensus': consensus,
                    'health': health
                })

            all_states.append(zone_states)

        # Per-zone noisy feature means (mirror runtime edge: mean over ACTIVE
        # sensors of clipped water and raw noisy rain/soil readings)
        zone_step_water = {z.zone_id: [] for z in env.zones}
        zone_step_rain = {z.zone_id: [] for z in env.zones}
        zone_step_soil = {z.zone_id: [] for z in env.zones}

        # Re-simulate sensor readings to feed the feature buffers exactly like
        # the runtime edge (noise scales: water sigma, soil 0.5*sigma, rain 0.3*sigma;
        # water clipped to [0, 2] and outlier-clipped with max delta 0.3).
        for step in range(steps):
            for zone in env.zones:
                zs = all_states[step][zone.zone_id]
                zone_id = zone.zone_id
                active_waters, active_rains, active_soils = [], [], []
                last_clipped = zone_clipper_last[zone_id]
                for s_idx in range(sensors_per_zone):
                    if rng.random() < dropout_rate:
                        continue
                    noisy_water = zs['water'] + rng.normal(0, noise_std)
                    noisy_water = np.clip(noisy_water, 0, 2.0)
                    clipped = noisy_water
                    if last_clipped is not None:
                        clipped = min(max(noisy_water, last_clipped - outlier_clip),
                                      last_clipped + outlier_clip)
                    last_clipped = clipped
                    active_waters.append(clipped)
                    active_rains.append(zs['rain'] + rng.normal(0, noise_std * 0.3))
                    active_soils.append(zs['soil'] + rng.normal(0, noise_std * 0.5))
                zone_clipper_last[zone_id] = last_clipped
                zone_step_water[zone_id].append(
                    float(np.mean(active_waters)) if active_waters else 0.0)
                zone_step_rain[zone_id].append(
                    float(np.mean(active_rains)) if active_rains else 0.0)
                zone_step_soil[zone_id].append(
                    float(np.mean(active_soils)) if active_soils else 0.0)

        # Compute future flood labels
        for zone in env.zones:
            for t in range(steps):
                future_steps = all_states[t + 1: t + 1 + self.horizon_T]
                will_flood = any(s[zone.zone_id]['flooded'] for s in future_steps)
                future_flood[zone.zone_id].append(will_flood)

        # Build feature records
        records = []
        min_warmup = 20

        for zone in env.zones:
            water_buf = RingBuffer(50)
            rain_buf = RingBuffer(50)
            soil_buf = RingBuffer(50)

            for step in range(steps - self.horizon_T):
                water_buf.append(zone_step_water[zone.zone_id][step])
                rain_buf.append(zone_step_rain[zone.zone_id][step])
                soil_buf.append(zone_step_soil[zone.zone_id][step])

                if step < min_warmup:
                    continue

                meta = zone_step_meta[zone.zone_id][step]

                features = {
                    'episode_id': episode_id,
                    'step': step,
                    'zone_id': zone.zone_id,
                    'scenario': scenario,
                    'dropout_rate': dropout_rate,
                    'noise_std': noise_std,
                    'water_mean_5': water_buf.mean(5),
                    'water_slope_5': water_buf.slope(5),
                    'water_max_10': water_buf.max(10),
                    'rain_sum_20': rain_buf.sum(20),
                    'rain_mean_10': rain_buf.mean(10),
                    'soil_mean_10': soil_buf.mean(10),
                    'consensus': meta['consensus'],
                    'health': meta['health'],
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
