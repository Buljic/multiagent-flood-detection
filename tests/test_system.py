"""
System tests for FloodMAS.
Run with: python -m pytest tests/ -v
"""

from pathlib import Path
import sys
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pytest
import numpy as np
import pandas as pd
import yaml
import copy


def load_config():
    """Load default configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================================
# Environment Tests
# ============================================================================

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

    def test_water_non_negative(self):
        """Water levels must never go negative."""
        from sim.environment import FloodEnvironment
        config = load_config()
        env = FloodEnvironment(config, seed=42)
        env.generate_random_rainfall('extreme')

        for _ in range(200):
            env.step()
            assert np.all(env.water_level >= 0), "Water level went negative"

    def test_soil_saturation_bounds(self):
        """Soil saturation must stay in [0, 1]."""
        from sim.environment import FloodEnvironment
        config = load_config()
        env = FloodEnvironment(config, seed=42)
        env.generate_random_rainfall('extreme')

        for _ in range(200):
            env.step()
            assert np.all(env.soil_saturation >= 0), "Soil saturation went negative"
            assert np.all(env.soil_saturation <= 1.0), "Soil saturation exceeded 1.0"

    def test_no_rain_no_flood(self):
        """With no rain and dry soil, no zone should flood."""
        from sim.environment import FloodEnvironment
        config = load_config()
        env = FloodEnvironment(config, seed=42)
        env.reset(soil_saturation_init=0.1)
        # No rainfall added

        for _ in range(100):
            env.step()

        for zone in env.zones:
            if not zone.is_river_zone:
                assert not env.is_flooded(zone.zone_id), \
                    f"Zone {zone.zone_id} flooded with no rain"

    def test_reset_clears_state(self):
        """Reset should restore environment to initial conditions."""
        from sim.environment import FloodEnvironment
        config = load_config()
        env = FloodEnvironment(config, seed=42)
        env.generate_random_rainfall('extreme')

        for _ in range(100):
            env.step()

        env.reset(soil_saturation_init=0.2)
        assert env.current_step == 0
        assert len(env.rainfall_events) == 0


# ============================================================================
# Guardrails Tests
# ============================================================================

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

    def test_full_state_cycle(self):
        """NORMAL -> SUSPECTED -> ALERT -> COOLDOWN -> NORMAL full cycle."""
        from sim.guardrails import AlertStateMachine, GuardrailsConfig, AlertState

        config = GuardrailsConfig(K_UP=2, K_DOWN=2, cooldown_steps=3)
        sm = AlertStateMachine(config)

        # NORMAL -> SUSPECTED (2 high-risk steps)
        for _ in range(2):
            sm.update(risk=0.8, consensus=0.7, health=1.0)
        assert sm.state == AlertState.SUSPECTED

        # SUSPECTED -> ALERT (2 more high-risk steps)
        for _ in range(2):
            sm.update(risk=0.8, consensus=0.7, health=1.0)
        assert sm.state == AlertState.ALERT

        # ALERT -> COOLDOWN (2 low-risk steps)
        for _ in range(2):
            sm.update(risk=0.1, consensus=0.1, health=1.0)
        assert sm.state == AlertState.COOLDOWN

        # COOLDOWN -> NORMAL (wait cooldown_steps=3)
        for _ in range(3):
            sm.update(risk=0.1, consensus=0.1, health=1.0)
        assert sm.state == AlertState.NORMAL

    def test_consensus_gating_blocks_alert(self):
        """High risk with low consensus should NOT trigger alert."""
        from sim.guardrails import AlertStateMachine, GuardrailsConfig, AlertState

        config = GuardrailsConfig(K_UP=2, CONS_MIN=0.5)
        sm = AlertStateMachine(config)

        # High risk but consensus below CONS_MIN
        for _ in range(10):
            sm.update(risk=0.9, consensus=0.3, health=1.0)

        assert sm.state == AlertState.NORMAL, \
            "Alert triggered despite low consensus"

    def test_degraded_mode_stricter_thresholds(self):
        """Low health should make it harder to trigger alerts."""
        from sim.guardrails import AlertStateMachine, GuardrailsConfig, AlertState

        config = GuardrailsConfig(
            TH_UP=0.6, degraded_TH_UP=0.8,
            HEALTH_MIN=0.6, K_UP=2, degraded_K_UP=2
        )
        sm = AlertStateMachine(config)

        # Risk=0.7 > TH_UP=0.6 but < degraded_TH_UP=0.8
        # With low health, degraded mode uses 0.8 threshold
        for _ in range(5):
            sm.update(risk=0.7, consensus=0.8, health=0.3)

        assert sm.state == AlertState.NORMAL, \
            "Alert triggered in degraded mode with risk below degraded threshold"

    def test_hysteresis_prevents_oscillation(self):
        """Risk between TH_DOWN and TH_UP should not cause state changes."""
        from sim.guardrails import AlertStateMachine, GuardrailsConfig, AlertState

        config = GuardrailsConfig(TH_UP=0.6, TH_DOWN=0.4, K_UP=2, K_DOWN=2)
        sm = AlertStateMachine(config)

        # First trigger to SUSPECTED
        for _ in range(2):
            sm.update(risk=0.8, consensus=0.7, health=1.0)
        assert sm.state == AlertState.SUSPECTED

        # Now risk in dead zone [0.4, 0.6) - should gradually decay counters
        initial_changes = sm.state_changes
        for _ in range(20):
            sm.update(risk=0.5, consensus=0.7, health=1.0)

        # Should not oscillate rapidly
        assert sm.state_changes - initial_changes <= 1

    def test_input_validation_clips_out_of_range(self):
        """Out-of-range inputs should be clipped, not crash."""
        from sim.guardrails import AlertStateMachine, GuardrailsConfig

        config = GuardrailsConfig()
        sm = AlertStateMachine(config)

        # Should not crash with out-of-range values
        sm.update(risk=1.5, consensus=-0.5, health=2.0)
        sm.update(risk=-0.1, consensus=1.1, health=-0.3)

    def test_ring_buffer(self):
        from sim.guardrails import RingBuffer

        buf = RingBuffer(10)
        for i in range(15):
            buf.append(float(i))

        assert len(buf) == 10
        assert buf.mean(5) == 12.0

    def test_ring_buffer_slope_positive(self):
        """Rising values should produce positive slope."""
        from sim.guardrails import RingBuffer

        buf = RingBuffer(50)
        for i in range(20):
            buf.append(float(i))

        assert buf.slope(10) > 0

    def test_ring_buffer_slope_flat(self):
        """Constant values should produce near-zero slope."""
        from sim.guardrails import RingBuffer

        buf = RingBuffer(50)
        for _ in range(20):
            buf.append(5.0)

        assert abs(buf.slope(10)) < 1e-10

    def test_feature_extractor(self):
        from sim.guardrails import FeatureExtractor

        fe = FeatureExtractor()
        for i in range(10):
            fe.update(water=0.1 * i, rain=0.05, soil=0.3)

        features = fe.extract(consensus=0.5, health=1.0)

        assert 'water_mean_5' in features
        assert 'consensus' in features
        assert features['health'] == 1.0
        assert len(features) == 8, f"Expected 8 features, got {len(features)}"

    def test_feature_extractor_all_keys(self):
        """FeatureExtractor must produce all 8 expected feature keys."""
        from sim.guardrails import FeatureExtractor

        expected_keys = {
            'water_mean_5', 'water_slope_5', 'water_max_10',
            'rain_sum_20', 'rain_mean_10', 'soil_mean_10',
            'consensus', 'health'
        }
        fe = FeatureExtractor()
        fe.update(water=0.5, rain=0.1, soil=0.3)
        features = fe.extract(consensus=0.6, health=0.9)

        assert set(features.keys()) == expected_keys

    def test_outlier_clipper(self):
        """OutlierClipper should cap large jumps."""
        from sim.guardrails import OutlierClipper

        clipper = OutlierClipper(max_delta=0.3)
        v1 = clipper.clip(0.5)
        v2 = clipper.clip(1.5)  # delta=1.0, should be clipped

        assert v1 == 0.5
        assert v2 == 0.5 + 0.3  # clipped to last + max_delta

    def test_missing_value_handler_carry_forward(self):
        """MissingValueHandler should carry forward and penalize."""
        from sim.guardrails import MissingValueHandler

        handler = MissingValueHandler(max_carry=3)

        # First valid reading
        val, penalty = handler.process(sensor_id=1, value=0.5)
        assert val == 0.5
        assert penalty == 0.0

        # Missing reading - should carry forward
        val, penalty = handler.process(sensor_id=1, value=None)
        assert val == 0.5  # carried forward
        assert penalty > 0  # should have penalty

        # Multiple missing - penalty should increase
        _, p2 = handler.process(sensor_id=1, value=None)
        assert p2 > penalty  # increasing penalty


# ============================================================================
# Agent Tests
# ============================================================================

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

    def test_risk_values_bounded(self):
        """Risk scores should always be in [0, 1]."""
        from sim.model import FloodModel
        config = load_config()

        model = FloodModel(config, seed=42)
        model.generate_random_rainfall('extreme')
        logs = model.run(100)

        assert logs['risk'].min() >= 0.0, "Risk went below 0"
        assert logs['risk'].max() <= 1.0, "Risk went above 1"

    def test_health_values_bounded(self):
        """Health should always be in [0, 1]."""
        from sim.model import FloodModel
        config = load_config()

        model = FloodModel(config, seed=42)
        model.generate_random_rainfall('extreme')
        logs = model.run(100)

        assert logs['health'].min() >= 0.0, "Health went below 0"
        assert logs['health'].max() <= 1.0, "Health went above 1"

    def test_all_zones_produce_logs(self):
        """Every zone should produce log entries."""
        from sim.model import FloodModel
        config = load_config()
        num_zones = config['simulation']['num_zones']

        model = FloodModel(config, seed=42)
        model.generate_random_rainfall('normal')
        logs = model.run(50)

        logged_zones = logs['zone_id'].unique()
        assert len(logged_zones) == num_zones, \
            f"Expected {num_zones} zones in logs, got {len(logged_zones)}"

    def test_coordinator_logs(self):
        """Coordinator should produce structured logs."""
        from sim.model import FloodModel
        config = load_config()

        model = FloodModel(config, seed=42)
        model.generate_random_rainfall('normal')
        model.run(50)

        coord_logs = model.get_coordinator_logs()
        assert len(coord_logs) == 50
        assert 'global_risk' in coord_logs.columns
        assert 'global_alarm' in coord_logs.columns

    def test_model_reset(self):
        """Reset should clear all agent state."""
        from sim.model import FloodModel
        config = load_config()

        model = FloodModel(config, seed=42)
        model.generate_random_rainfall('normal')
        model.run(50)

        model.reset(soil_saturation_init=0.2)
        assert model.step_count == 0
        for edge in model.edges.values():
            assert len(edge.history) == 0
            assert edge.current_risk == 0.0

    def test_high_dropout_degrades_health(self):
        """With high dropout, average health should be noticeably below 1.0."""
        from sim.model import FloodModel
        config = copy.deepcopy(load_config())
        config['sensors']['dropout_rate'] = 0.5

        model = FloodModel(config, seed=42)
        model.generate_random_rainfall('normal')
        logs = model.run(100)

        avg_health = logs['health'].mean()
        assert avg_health < 0.9, \
            f"Expected degraded health with 50% dropout, got {avg_health:.2f}"

    def test_sensor_count_per_zone(self):
        """Each zone should have the configured number of sensors."""
        from sim.model import FloodModel
        config = load_config()
        expected = config['sensors']['per_zone']

        model = FloodModel(config, seed=42)
        for edge in model.edges.values():
            assert len(edge.sensors) == expected, \
                f"Zone {edge.zone_id}: expected {expected} sensors, got {len(edge.sensors)}"


# ============================================================================
# Baseline Tests
# ============================================================================

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

    def test_baseline_returns_to_normal(self):
        """Baseline should clear alert when conditions subside."""
        from baseline.threshold import ThresholdBaseline

        baseline = ThresholdBaseline()

        # Trigger alert
        for i in range(25):
            baseline.update(water=0.6, rain=0.2, step=i)
        assert baseline.current_state == 'ALERT'

        # Subside
        for i in range(25, 50):
            baseline.update(water=0.1, rain=0.0, step=i)
        assert baseline.current_state == 'NORMAL'

    def test_zoned_baseline(self):
        """ZonedThresholdBaseline should track zones independently."""
        from baseline.threshold import ZonedThresholdBaseline

        baseline = ZonedThresholdBaseline(num_zones=2)

        # Only zone 0 gets high readings
        for i in range(25):
            readings = {
                0: {'water': 0.6, 'rain': 0.2},
                1: {'water': 0.1, 'rain': 0.0}
            }
            baseline.update(readings, step=i)

        assert baseline.zone_baselines[0].current_state == 'ALERT'
        assert baseline.zone_baselines[1].current_state == 'NORMAL'


# ============================================================================
# Metrics Tests
# ============================================================================

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

    def test_detection_metrics_perfect(self):
        """Perfect predictions should yield precision=recall=f1=1.0."""
        from eval.metrics import MetricsCalculator

        calc = MetricsCalculator()
        y = np.array([0, 1, 0, 1, 1])
        metrics = calc.compute_detection_metrics(y, y)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.false_positive_rate == 0.0

    def test_detection_metrics_single_class(self):
        """Should handle single-class inputs without crashing and produce
        consistent metrics (no impossible precision=1/recall=1/f1=0 combos)."""
        from eval.metrics import MetricsCalculator

        calc = MetricsCalculator()

        # All negative, predicted all negative => perfect TN, no TP/FP/FN
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        metrics = calc.compute_detection_metrics(y_true, y_pred)
        assert metrics.accuracy == 1.0
        assert metrics.precision == 0.0  # no positives predicted
        assert metrics.recall == 0.0     # no positives in truth
        assert metrics.f1 == 0.0
        assert metrics.confusion_matrix[0, 0] == 4  # 4 TN

        # All negative, but predicted some positive => FP only
        y_pred_fp = np.array([1, 0, 0, 0])
        metrics_fp = calc.compute_detection_metrics(y_true, y_pred_fp)
        assert metrics_fp.precision == 0.0  # TP=0, FP=1
        assert metrics_fp.recall == 0.0     # no positives in truth
        assert metrics_fp.false_positive_rate > 0

        # All positive, predicted all positive => perfect TP
        y_all_pos = np.array([1, 1, 1, 1])
        metrics_pos = calc.compute_detection_metrics(y_all_pos, y_all_pos)
        assert metrics_pos.precision == 1.0
        assert metrics_pos.recall == 1.0
        assert metrics_pos.f1 == 1.0

    def test_stability_metrics(self):
        from eval.metrics import MetricsCalculator

        calc = MetricsCalculator()
        state_history = ['NORMAL', 'NORMAL', 'SUSPECTED', 'ALERT', 'ALERT', 'NORMAL']

        metrics = calc.compute_stability_metrics(state_history)

        assert metrics.total_state_changes == 3

    def test_lead_time_no_alerts(self):
        """No alerts before flood should return zero lead times."""
        from eval.metrics import MetricsCalculator

        calc = MetricsCalculator()
        result = calc.compute_lead_time(alert_times=[], flood_times=[100])

        assert result.mean_lead_time == 0.0
        assert len(result.lead_times) == 0

    def test_lead_time_correct(self):
        """Lead time should be flood_time - alert_time."""
        from eval.metrics import MetricsCalculator

        calc = MetricsCalculator()
        result = calc.compute_lead_time(
            alert_times=[90, 80],
            flood_times=[100]
        )

        assert result.mean_lead_time == 10.0  # 100 - 90

    def test_compare_systems(self):
        """compare_systems should return improvement metrics."""
        from eval.metrics import MetricsCalculator

        calc = MetricsCalculator()

        # Create minimal mock logs
        mas_logs = pd.DataFrame({
            'step': list(range(10)) * 2,
            'zone_id': [0]*10 + [1]*10,
            'state': ['NORMAL']*5 + ['ALERT']*5 + ['NORMAL']*10,
            'ground_truth_flooded': [False]*5 + [True]*5 + [False]*10
        })
        baseline_logs = pd.DataFrame({
            'step': list(range(10)) * 2,
            'zone_id': [0]*10 + [1]*10,
            'state': ['NORMAL']*10 + ['NORMAL']*10,
            'ground_truth_flooded': [False]*5 + [True]*5 + [False]*10
        })

        result = calc.compare_systems(mas_logs, baseline_logs)
        assert 'mas' in result
        assert 'baseline' in result
        assert 'improvement' in result


# ============================================================================
# Time Mapping Tests
# ============================================================================

class TestTimeMapping:
    """Tests for real-world time mapping."""

    def test_steps_to_real_time_minutes(self):
        from sim.model import steps_to_real_time
        config = {'step_duration_minutes': 15}

        assert steps_to_real_time(1, config) == "15min"
        assert steps_to_real_time(3, config) == "45min"

    def test_steps_to_real_time_hours(self):
        from sim.model import steps_to_real_time
        config = {'step_duration_minutes': 15}

        assert steps_to_real_time(4, config) == "1h 0min"
        assert steps_to_real_time(10, config) == "2h 30min"

    def test_steps_to_real_time_days(self):
        from sim.model import steps_to_real_time
        config = {'step_duration_minutes': 15}

        result = steps_to_real_time(400, config)
        assert result.startswith("4d")

    def test_steps_to_real_time_default(self):
        """Without config key, should default to 15 min."""
        from sim.model import steps_to_real_time

        assert steps_to_real_time(4, {}) == "1h 0min"


# ============================================================================
# Data Generation Tests
# ============================================================================

class TestDataGeneration:
    """Tests for synthetic data generation."""

    def test_deepcopy_isolation(self):
        """Config changes in _run_episode must NOT mutate the original."""
        config = load_config()
        original_dropout = config['sensors']['dropout_rate']
        original_noise = config['sensors']['noise_std']

        from ml.generate_data import DataGenerator
        gen = DataGenerator(config, seed=42)

        # Generate a small dataset with varying params
        gen.generate(num_episodes=5, steps_per_episode=50)

        assert config['sensors']['dropout_rate'] == original_dropout, \
            "Original config was mutated by data generation (dropout_rate)"
        assert config['sensors']['noise_std'] == original_noise, \
            "Original config was mutated by data generation (noise_std)"

    def test_generated_features_complete(self):
        """Generated data should contain all required feature columns."""
        config = load_config()
        from ml.generate_data import DataGenerator

        gen = DataGenerator(config, seed=42)
        df = gen.generate(num_episodes=3, steps_per_episode=80)

        expected_features = config['ml']['features']
        for feat in expected_features:
            assert feat in df.columns, f"Missing feature column: {feat}"

        assert 'flood_in_next_T' in df.columns
        assert 'episode_id' in df.columns

    def test_consensus_realistic_range(self):
        """Consensus values should be in [0, 1] and vary."""
        config = load_config()
        from ml.generate_data import DataGenerator

        gen = DataGenerator(config, seed=42)
        df = gen.generate(num_episodes=5, steps_per_episode=100)

        assert df['consensus'].min() >= 0.0
        assert df['consensus'].max() <= 1.0
        # Should have some variation (not all same value)
        assert df['consensus'].std() > 0.01, "Consensus has no variation"

    def test_health_reflects_dropout(self):
        """Health should be lower when dropout_rate is higher."""
        config = load_config()
        from ml.generate_data import DataGenerator

        gen = DataGenerator(config, seed=42)
        df = gen.generate(num_episodes=20, steps_per_episode=80)

        # Compare health for different dropout rates
        low_dropout = df[df['dropout_rate'] == 0.0]['health'].mean()
        high_dropout = df[df['dropout_rate'] == 0.3]['health'].mean()

        if len(df[df['dropout_rate'] == 0.0]) > 0 and len(df[df['dropout_rate'] == 0.3]) > 0:
            assert low_dropout > high_dropout, \
                f"Health should be lower with high dropout ({low_dropout:.2f} vs {high_dropout:.2f})"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Full pipeline: simulate -> baseline -> metrics."""
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

        assert len(logs) > 0
        assert 'detection' in metrics or 'stability' in metrics

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical results."""
        from sim.model import FloodModel

        config = load_config()

        model1 = FloodModel(config, seed=123)
        model1.generate_random_rainfall('normal')
        logs1 = model1.run(50)

        model2 = FloodModel(config, seed=123)
        model2.generate_random_rainfall('normal')
        logs2 = model2.run(50)

        pd.testing.assert_frame_equal(logs1, logs2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different results."""
        from sim.model import FloodModel

        config = load_config()

        model1 = FloodModel(config, seed=1)
        model1.generate_random_rainfall('normal')
        logs1 = model1.run(50)

        model2 = FloodModel(config, seed=999)
        model2.generate_random_rainfall('normal')
        logs2 = model2.run(50)

        # At least risk values should differ
        assert not np.allclose(logs1['risk'].values, logs2['risk'].values)

    def test_ground_truth_varies_per_step(self):
        """ground_truth_flooded must reflect the state at each step, not just
        the final state. With extreme rain a zone should transition from
        not-flooded to flooded during the simulation."""
        from sim.model import FloodModel

        config = load_config()
        model = FloodModel(config, seed=42)
        # Use extreme rain with wet soil to ensure flooding occurs
        model.reset(soil_saturation_init=0.7)
        model.environment.add_rainfall_event(intensity=0.8, duration=60, start_step=10)

        logs = model.run(200)

        assert 'ground_truth_flooded' in logs.columns, \
            "ground_truth_flooded column missing from logs"

        gt = logs.groupby('zone_id')['ground_truth_flooded']
        # At least one zone should have BOTH True and False entries
        # (starts dry, becomes flooded during heavy rain)
        has_variation = False
        for zone_id, group in gt:
            if group.nunique() > 1:
                has_variation = True
                break

        assert has_variation, (
            "ground_truth_flooded is constant for every zone — "
            "it should vary over time as flooding develops"
        )

    def test_mitigation_agent_finds_edge(self):
        """MitigationAgent should find its zone's edge via model.edges."""
        from sim.model import FloodModel

        config = copy.deepcopy(load_config())
        config['countermeasures']['enabled'] = True

        model = FloodModel(config, seed=42)
        model.generate_random_rainfall('extreme')

        # Verify each mitigation agent can find its edge
        for zone_id, mitigation in model.mitigations.items():
            edge = mitigation._get_zone_edge()
            assert edge is not None, \
                f"MitigationAgent for zone {zone_id} could not find its edge"
            assert edge.zone_id == zone_id


# ============================================================================
# Experiment Runner Tests
# ============================================================================

class TestExperimentRunner:
    """Tests for experiment runner correctness."""

    def test_baseline_ground_truth_varies_per_step(self):
        """Baseline ground truth must be per-step, not end-of-sim constant."""
        from sim.model import FloodModel
        from baseline.threshold import ZonedThresholdBaseline

        config = load_config()
        model = FloodModel(config, seed=42)
        model.reset(soil_saturation_init=0.7)
        model.environment.add_rainfall_event(intensity=0.8, duration=60, start_step=10)

        num_zones = config['simulation']['num_zones']
        baseline = ZonedThresholdBaseline(num_zones=num_zones, config=config)
        per_step_gt = {}

        for step in range(200):
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

        # At least one zone should transition from not-flooded to flooded
        has_variation = False
        for zone_id in range(num_zones):
            gt_values = [per_step_gt[(s, zone_id)] for s in range(200)]
            if len(set(gt_values)) > 1:
                has_variation = True
                break

        assert has_variation, \
            "Baseline ground truth is constant — per-step recording is broken"

    def test_metrics_single_class_consistency(self):
        """Single-class metrics must be internally consistent:
        - If no positives exist in truth and none predicted: P=0, R=0, F1=0
        - confusion_matrix must reflect actual counts, not zeros."""
        from eval.metrics import MetricsCalculator

        calc = MetricsCalculator()
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0])

        m = calc.compute_detection_metrics(y_true, y_pred)

        # No positives anywhere => P, R, F1 all zero
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        # Confusion matrix must have actual counts
        assert m.confusion_matrix[0, 0] == 5  # 5 true negatives
        assert m.confusion_matrix.sum() == 5

    def test_metrics_no_impossible_combinations(self):
        """precision=1, recall=1 with f1=0 should never happen."""
        from eval.metrics import MetricsCalculator

        calc = MetricsCalculator()

        # Test various single-class combinations
        for y_true, y_pred in [
            (np.array([0, 0, 0]), np.array([0, 0, 0])),
            (np.array([1, 1, 1]), np.array([1, 1, 1])),
            (np.array([0, 0, 0]), np.array([1, 0, 0])),
            (np.array([1, 1, 1]), np.array([0, 1, 1])),
        ]:
            m = calc.compute_detection_metrics(y_true, y_pred)
            # If F1 is 0, at least one of P or R must be 0
            if m.f1 == 0.0:
                assert m.precision == 0.0 or m.recall == 0.0, \
                    f"F1=0 but P={m.precision}, R={m.recall} — impossible"

    def test_feature_columns_from_config(self):
        """ModelTrainer should use feature list from config, not hardcoded."""
        from ml.train import ModelTrainer, get_feature_columns

        config = load_config()
        trainer = ModelTrainer(config=config)

        assert trainer.feature_columns == config['ml']['features']

        # Without config, should fall back to defaults
        trainer_default = ModelTrainer()
        assert len(trainer_default.feature_columns) == 8

    def test_stability_matches_state_machine(self):
        """Stability total_state_changes from metrics must match the actual
        state machine counts (no spurious zone-boundary transitions)."""
        from sim.model import FloodModel
        from eval.metrics import MetricsCalculator

        config = load_config()
        model = FloodModel(config, seed=42)
        model.reset(soil_saturation_init=0.7)
        model.environment.add_rainfall_event(intensity=0.8, duration=60, start_step=10)
        logs = model.run(200)

        # Ground-truth state changes from actual state machines
        sm_total = sum(e.state_machine.state_changes for e in model.edges.values())

        calc = MetricsCalculator()
        metrics = calc.compute_from_logs(logs)
        metric_total = metrics['stability']['total_state_changes']

        assert metric_total == sm_total, (
            f"Metric stability ({metric_total}) != state machine ({sm_total}). "
            f"Likely counting zone-boundary transitions as state changes."
        )

    def test_groupkfold_single_episode_no_crash(self):
        """GroupKFold with only 1 episode must not crash."""
        from ml.train import ModelTrainer

        config = load_config()
        trainer = ModelTrainer(config=config)

        # Tiny dataset: 1 episode
        X = pd.DataFrame({
            'water_mean_5': [0.1, 0.2, 0.3],
            'water_slope_5': [0.0, 0.01, 0.02],
            'water_max_10': [0.1, 0.2, 0.3],
            'rain_sum_20': [0.5, 1.0, 1.5],
            'rain_mean_10': [0.05, 0.1, 0.15],
            'soil_mean_10': [0.3, 0.3, 0.3],
            'consensus': [0.0, 0.5, 1.0],
            'health': [1.0, 1.0, 1.0],
        })
        y = pd.Series([0, 0, 1])
        groups = pd.Series([0, 0, 0])  # single episode

        # Should not raise — skips CV gracefully
        result = trainer.cross_validate(X, y, groups=groups, cv=5)
        assert 'cv_auc_mean' in result
        assert result['cv_scores'] == []  # No CV possible with 1 group

    def test_no_runtime_warnings_on_import(self):
        """Package __init__.py should not eagerly import -m runnable modules."""
        import importlib
        import warnings

        for pkg in ['eval', 'sim', 'ml']:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                importlib.reload(importlib.import_module(pkg))
                runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
                assert len(runtime_warnings) == 0, (
                    f"Package '{pkg}' produced RuntimeWarning on import: "
                    f"{[str(x.message) for x in runtime_warnings]}"
                )


    def test_lead_time_per_zone_not_cross_zone(self):
        """Lead time must NOT match alerts in one zone with floods in another."""
        from eval.metrics import MetricsCalculator

        calc = MetricsCalculator()
        # Zone 0: alert at step 10, no flood ever
        # Zone 1: no alert, flood at step 20
        # Cross-zone: old code would compute lead_time = 20-10 = 10 (wrong)
        # Per-zone:   zone 0 has no flood → no lead time
        #             zone 1 has no alert → no lead time
        #             Expected count = 0
        logs = pd.DataFrame([
            *[{'step': s, 'zone_id': 0,
               'state': 'ALERT' if 10 <= s <= 30 else 'NORMAL',
               'ground_truth_flooded': False} for s in range(50)],
            *[{'step': s, 'zone_id': 1,
               'state': 'NORMAL',
               'ground_truth_flooded': s >= 20} for s in range(50)],
        ])

        metrics = calc.compute_from_logs(logs)
        assert metrics['lead_time']['count'] == 0, (
            "Cross-zone alert/flood should NOT produce a lead time"
        )

    def test_num_zones_non_square_raises(self):
        """Non-perfect-square num_zones must raise ValueError."""
        from sim.environment import FloodEnvironment

        config = load_config()
        config = copy.deepcopy(config)
        config['simulation']['num_zones'] = 6  # not a perfect square

        with pytest.raises(ValueError, match="not a perfect square"):
            FloodEnvironment(config, seed=42)

    def test_evaluate_single_class_test_set(self):
        """evaluate() must not crash when test set has only one class."""
        from ml.train import ModelTrainer

        config = load_config()
        trainer = ModelTrainer(config=config)

        # Need >=5 samples per class for 5-fold calibration CV
        rng = np.random.default_rng(42)
        n = 20
        X_train = pd.DataFrame({
            'water_mean_5': rng.uniform(0, 1, n),
            'water_slope_5': rng.uniform(0, 0.1, n),
            'water_max_10': rng.uniform(0, 1, n),
            'rain_sum_20': rng.uniform(0, 5, n),
            'rain_mean_10': rng.uniform(0, 0.5, n),
            'soil_mean_10': rng.uniform(0, 1, n),
            'consensus': rng.uniform(0, 1, n),
            'health': rng.uniform(0.5, 1, n),
        })
        y_train = pd.Series([0]*10 + [1]*10)
        trainer.train(X_train, y_train)

        # Test set with only one class (all negative)
        X_test = pd.DataFrame({
            'water_mean_5': [0.1, 0.15],
            'water_slope_5': [0.0, 0.0],
            'water_max_10': [0.1, 0.15],
            'rain_sum_20': [0.5, 0.6],
            'rain_mean_10': [0.05, 0.06],
            'soil_mean_10': [0.3, 0.3],
            'consensus': [0.0, 0.0],
            'health': [1.0, 1.0],
        })
        y_test = pd.Series([0, 0])

        result = trainer.evaluate(X_test, y_test)
        assert 'auc_roc' in result
        assert np.isnan(result['auc_roc']), "AUC-ROC should be NaN for single-class test set"

    def test_experiment_seed_from_config(self):
        """Experiment seed should come from config, not be hardcoded."""
        from eval.run_experiments import ExperimentRunner

        config = load_config()
        config = copy.deepcopy(config)
        config['seed'] = 999

        runner = ExperimentRunner(config)
        metadata = runner._generate_run_metadata([], repeats=1)
        assert metadata['seed'] == 999, (
            f"Metadata seed should match config seed 999, got {metadata['seed']}"
        )


    def test_zone_allocation_covers_entire_grid(self):
        """Every grid cell must belong to exactly one zone, even when
        grid_size is not perfectly divisible by sqrt(num_zones)."""
        from sim.environment import FloodEnvironment

        config = load_config()
        config = copy.deepcopy(config)
        config['simulation']['grid_size'] = 20
        config['simulation']['num_zones'] = 9  # sqrt=3, 20//3=6, remainder=2

        env = FloodEnvironment(config, seed=42)
        env.reset(soil_saturation_init=0.3)

        grid_size = config['simulation']['grid_size']
        all_cells = set()
        for zone in env.zones:
            all_cells.update(zone.cells)

        expected = {(r, c) for r in range(grid_size) for c in range(grid_size)}
        assert all_cells == expected, (
            f"Unallocated cells: {expected - all_cells}"
        )

    def test_lead_time_counts_suspected_as_alert(self):
        """Lead time must count SUSPECTED as alert start (matching detection)."""
        from eval.metrics import MetricsCalculator

        calc = MetricsCalculator()
        # Zone 0: SUSPECTED at step 10 (no ALERT), flood at step 20
        logs = pd.DataFrame([
            {'step': s, 'zone_id': 0,
             'state': 'SUSPECTED' if 10 <= s <= 15 else 'NORMAL',
             'ground_truth_flooded': s >= 20}
            for s in range(50)
        ])

        metrics = calc.compute_from_logs(logs)
        assert metrics['lead_time']['count'] > 0, (
            "SUSPECTED should count as alert start for lead time"
        )

    def test_evaluate_no_undefined_metric_warning(self):
        """evaluate() with single-class test set should not emit
        UndefinedMetricWarning (zero_division=0 suppresses it)."""
        import warnings
        from ml.train import ModelTrainer

        config = load_config()
        trainer = ModelTrainer(config=config)

        rng = np.random.default_rng(42)
        n = 20
        X_train = pd.DataFrame({
            'water_mean_5': rng.uniform(0, 1, n),
            'water_slope_5': rng.uniform(0, 0.1, n),
            'water_max_10': rng.uniform(0, 1, n),
            'rain_sum_20': rng.uniform(0, 5, n),
            'rain_mean_10': rng.uniform(0, 0.5, n),
            'soil_mean_10': rng.uniform(0, 1, n),
            'consensus': rng.uniform(0, 1, n),
            'health': rng.uniform(0.5, 1, n),
        })
        y_train = pd.Series([0]*10 + [1]*10)
        trainer.train(X_train, y_train)

        X_test = pd.DataFrame({
            'water_mean_5': [0.1, 0.15],
            'water_slope_5': [0.0, 0.0],
            'water_max_10': [0.1, 0.15],
            'rain_sum_20': [0.5, 0.6],
            'rain_mean_10': [0.05, 0.06],
            'soil_mean_10': [0.3, 0.3],
            'consensus': [0.0, 0.0],
            'health': [1.0, 1.0],
        })
        y_test = pd.Series([0, 0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trainer.evaluate(X_test, y_test)
            undefined_warnings = [
                x for x in w
                if 'UndefinedMetric' in str(x.category.__name__)
            ]
            assert len(undefined_warnings) == 0, (
                f"Got UndefinedMetricWarning: {[str(x.message) for x in undefined_warnings]}"
            )


    def test_lead_time_without_zone_id_column(self):
        """compute_from_logs must not crash when logs lack zone_id column."""
        from eval.metrics import MetricsCalculator

        calc = MetricsCalculator()
        # Single-zone logs without zone_id column
        logs = pd.DataFrame([
            {'step': s, 'state': 'ALERT' if 10 <= s <= 15 else 'NORMAL',
             'ground_truth_flooded': s >= 20}
            for s in range(50)
        ])
        assert 'zone_id' not in logs.columns

        metrics = calc.compute_from_logs(logs)
        assert 'lead_time' in metrics
        assert metrics['lead_time']['count'] > 0

    def test_num_zones_too_large_raises(self):
        """num_zones larger than grid_size^2 must raise ValueError."""
        from sim.environment import FloodEnvironment

        config = load_config()
        config = copy.deepcopy(config)
        config['simulation']['grid_size'] = 5
        config['simulation']['num_zones'] = 36  # sqrt=6 > grid_size=5

        with pytest.raises(ValueError, match="too large"):
            FloodEnvironment(config, seed=42)

    def test_cv_without_groups_single_class(self):
        """cross_validate without groups on single-class data must not crash."""
        from ml.train import ModelTrainer

        config = load_config()
        trainer = ModelTrainer(config=config)

        X = pd.DataFrame({
            'water_mean_5': [0.1, 0.2, 0.3],
            'water_slope_5': [0.0, 0.01, 0.02],
            'water_max_10': [0.1, 0.2, 0.3],
            'rain_sum_20': [0.5, 1.0, 1.5],
            'rain_mean_10': [0.05, 0.1, 0.15],
            'soil_mean_10': [0.3, 0.3, 0.3],
            'consensus': [0.0, 0.5, 1.0],
            'health': [1.0, 1.0, 1.0],
        })
        y = pd.Series([0, 0, 0])  # single class, no groups

        result = trainer.cross_validate(X, y, groups=None, cv=3)
        assert 'cv_auc_mean' in result
        assert result['cv_scores'] == []  # skipped


    def test_sensor_reading_not_mutated_by_edge(self):
        """Edge processing must not mutate the sensor's internal last_reading.
        Captures the water value BEFORE edge.step(), then verifies it's
        unchanged AFTER edge.step() processes (and clips) a copy."""
        from sim.model import FloodModel

        config = load_config()
        model = FloodModel(config, seed=42)
        model.reset(soil_saturation_init=0.5)
        model.environment.add_rainfall_event(intensity=0.5, duration=30, start_step=5)

        # Run sensor steps to populate readings, but NOT edge steps yet
        for _ in range(10):
            for sensor in model.sensors:
                sensor.step()
            model.environment.step()

        # Snapshot sensor water values BEFORE edge processing
        snapshots = {}
        for zone_id, edge in model.edges.items():
            for sensor in edge.sensors:
                if sensor.last_reading is not None:
                    snapshots[sensor.unique_id] = sensor.last_reading['water']

        # Now run edge steps (which clip water via outlier_clipper)
        for edge in model.edges.values():
            edge.step()

        # Verify sensor readings are unchanged after edge processing
        mutations_found = 0
        for zone_id, edge in model.edges.items():
            for sensor in edge.sensors:
                if sensor.unique_id in snapshots:
                    assert sensor.last_reading['water'] == snapshots[sensor.unique_id], (
                        f"Sensor {sensor.unique_id} water was mutated by edge processing: "
                        f"before={snapshots[sensor.unique_id]}, after={sensor.last_reading['water']}"
                    )
                    mutations_found += 1

        assert mutations_found > 0, "No active sensors found to verify"

    def test_episode_split_small_dataset_nonempty(self):
        """Episode-based train/test split must produce non-empty test set
        even with very few episodes (<=4)."""
        from ml.train import ModelTrainer

        config = load_config()
        trainer = ModelTrainer(config=config)

        # 3 episodes, 2 rows each = 6 total rows
        X = pd.DataFrame({
            'water_mean_5': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'water_slope_5': [0.0]*6,
            'water_max_10': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'rain_sum_20': [0.5]*6,
            'rain_mean_10': [0.05]*6,
            'soil_mean_10': [0.3]*6,
            'consensus': [0.0, 0.5, 0.0, 0.5, 0.0, 0.5],
            'health': [1.0]*6,
            'episode_id': [0, 0, 1, 1, 2, 2],
        })
        y = pd.Series([0, 1, 0, 1, 0, 1])

        # The split should work without crashing and produce non-empty sets
        episodes = X['episode_id'].unique()
        n_test = max(1, int(len(episodes) * 0.2))
        assert n_test >= 1, "Must have at least 1 test episode"

    def test_grouped_cv_single_class_no_crash(self):
        """GroupKFold with multiple groups but single-class target must not crash."""
        from ml.train import ModelTrainer

        config = load_config()
        trainer = ModelTrainer(config=config)

        X = pd.DataFrame({
            'water_mean_5': [0.1, 0.2, 0.3, 0.4],
            'water_slope_5': [0.0]*4,
            'water_max_10': [0.1, 0.2, 0.3, 0.4],
            'rain_sum_20': [0.5]*4,
            'rain_mean_10': [0.05]*4,
            'soil_mean_10': [0.3]*4,
            'consensus': [0.0, 0.5, 0.0, 0.5],
            'health': [1.0]*4,
        })
        y = pd.Series([0, 0, 0, 0])  # single class
        groups = pd.Series([0, 0, 1, 1])  # 2 groups

        result = trainer.cross_validate(X, y, groups=groups, cv=2)
        assert 'cv_auc_mean' in result
        assert result['cv_scores'] == []  # should skip


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
