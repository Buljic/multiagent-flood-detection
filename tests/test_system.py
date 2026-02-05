"""
Basic system tests for FloodMAS.
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml


def load_config():
    """Load default configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TestEnvironment:
    """Tests for FloodEnvironment."""
    
    def test_environment_init(self):
        from sim.environment import FloodEnvironment
        config = load_config()
        env = FloodEnvironment(config, seed=42)
        
        assert env.grid_size == config['simulation']['grid_size']
        assert len(env.zones) == config['simulation']['num_zones']
        assert env.water_level.shape == (env.grid_size, env.grid_size)
    
    def test_environment_step(self):
        from sim.environment import FloodEnvironment
        config = load_config()
        env = FloodEnvironment(config, seed=42)
        env.generate_random_rainfall('normal')
        
        state = env.step()
        
        assert 'water_level' in state
        assert 'soil_saturation' in state
        assert 'rainfall' in state
        assert env.current_step == 1
    
    def test_zone_state(self):
        from sim.environment import FloodEnvironment
        config = load_config()
        env = FloodEnvironment(config, seed=42)
        
        zone_state = env.get_zone_state(0)
        
        assert 'water_mean' in zone_state
        assert 'soil_mean' in zone_state
        assert 0 <= zone_state['water_mean'] <= 2.0


class TestGuardrails:
    """Tests for guardrails system."""
    
    def test_state_machine_init(self):
        from sim.guardrails import AlertStateMachine, GuardrailsConfig, AlertState
        
        config = GuardrailsConfig()
        sm = AlertStateMachine(config)
        
        assert sm.state == AlertState.NORMAL
        assert sm.consecutive_up == 0
    
    def test_state_machine_transitions(self):
        from sim.guardrails import AlertStateMachine, GuardrailsConfig, AlertState
        
        config = GuardrailsConfig(K_UP=2, K_DOWN=2)
        sm = AlertStateMachine(config)
        
        sm.update(risk=0.7, consensus=0.6, health=1.0)
        sm.update(risk=0.7, consensus=0.6, health=1.0)
        
        assert sm.state == AlertState.SUSPECTED
    
    def test_ring_buffer(self):
        from sim.guardrails import RingBuffer
        
        buf = RingBuffer(10)
        for i in range(15):
            buf.append(float(i))
        
        assert len(buf) == 10
        assert buf.mean(5) == 12.0
    
    def test_feature_extractor(self):
        from sim.guardrails import FeatureExtractor
        
        fe = FeatureExtractor()
        for i in range(10):
            fe.update(water=0.1 * i, rain=0.05, soil=0.3)
        
        features = fe.extract(consensus=0.5, health=1.0)
        
        assert 'water_mean_5' in features
        assert 'consensus' in features
        assert features['health'] == 1.0


class TestAgents:
    """Tests for Mesa agents."""
    
    def test_flood_model_init(self):
        from sim.model import FloodModel
        config = load_config()
        
        model = FloodModel(config, seed=42)
        
        assert model.coordinator is not None
        assert len(model.edges) == config['simulation']['num_zones']
        assert len(model.sensors) > 0
    
    def test_flood_model_run(self):
        from sim.model import FloodModel
        config = load_config()
        
        model = FloodModel(config, seed=42)
        model.generate_random_rainfall('normal')
        
        logs = model.run(50)
        
        assert isinstance(logs, pd.DataFrame)
        assert len(logs) > 0
        assert 'risk' in logs.columns
        assert 'state' in logs.columns


class TestBaseline:
    """Tests for threshold baseline."""
    
    def test_baseline_init(self):
        from baseline.threshold import ThresholdBaseline
        
        baseline = ThresholdBaseline()
        
        assert baseline.current_state == 'NORMAL'
    
    def test_baseline_update(self):
        from baseline.threshold import ThresholdBaseline
        
        baseline = ThresholdBaseline()
        
        for _ in range(25):
            baseline.update(water=0.6, rain=0.2, step=0)
        
        assert baseline.current_state == 'ALERT'


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_detection_metrics(self):
        from eval.metrics import MetricsCalculator
        
        calc = MetricsCalculator()
        y_true = np.array([0, 0, 1, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1])
        
        metrics = calc.compute_detection_metrics(y_true, y_pred)
        
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
    
    def test_stability_metrics(self):
        from eval.metrics import MetricsCalculator
        
        calc = MetricsCalculator()
        state_history = ['NORMAL', 'NORMAL', 'SUSPECTED', 'ALERT', 'ALERT', 'NORMAL']
        
        metrics = calc.compute_stability_metrics(state_history)
        
        assert metrics.total_state_changes == 3


def run_quick_integration_test():
    """Run a quick integration test of the full pipeline."""
    print("Running quick integration test...")
    
    from sim.model import FloodModel
    from baseline.threshold import ZonedThresholdBaseline
    from eval.metrics import MetricsCalculator
    
    config = load_config()
    
    model = FloodModel(config, seed=42)
    model.generate_random_rainfall('extreme')
    
    baseline = ZonedThresholdBaseline(
        num_zones=config['simulation']['num_zones'],
        config=config
    )
    
    for step in range(100):
        model.step()
        
        zone_readings = {}
        for zone_id, edge in model.edges.items():
            zone_readings[zone_id] = {
                'water': edge.current_features.get('water_mean_5', 0),
                'rain': edge.current_features.get('rain_sum_20', 0) / 20
            }
        baseline.update(zone_readings, step)
    
    logs = model.get_logs()
    calc = MetricsCalculator()
    metrics = calc.compute_from_logs(logs)
    
    print(f"  Simulation steps: 100")
    print(f"  Log entries: {len(logs)}")
    print(f"  Metrics computed: {list(metrics.keys())}")
    print("Integration test PASSED!")
    
    return True


if __name__ == '__main__':
    run_quick_integration_test()
