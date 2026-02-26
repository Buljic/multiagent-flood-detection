"""
System tests for FloodMAS.
Run with: python -m pytest tests/ -v
"""

from pathlib import Path
import sys
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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
        """Should handle single-class inputs without crashing."""
        from eval.metrics import MetricsCalculator

        calc = MetricsCalculator()
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        metrics = calc.compute_detection_metrics(y_true, y_pred)
        assert metrics.accuracy == 1.0

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


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
