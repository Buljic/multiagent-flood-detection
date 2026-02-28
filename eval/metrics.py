"""
Evaluation metrics for flood detection systems.
Computes precision, recall, F1, false positive rate, lead time, and stability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


@dataclass
class DetectionMetrics:
    """Container for detection performance metrics."""
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    true_positive_rate: float
    accuracy: float
    confusion_matrix: np.ndarray


@dataclass
class StabilityMetrics:
    """Container for system stability metrics."""
    total_state_changes: int
    avg_state_changes_per_zone: float
    flapping_rate: float
    time_in_alert: float


@dataclass
class LeadTimeMetrics:
    """Container for lead time analysis."""
    mean_lead_time: float
    median_lead_time: float
    std_lead_time: float
    min_lead_time: float
    max_lead_time: float
    lead_times: List[int]


class MetricsCalculator:
    """
    Calculates comprehensive metrics for flood detection evaluation.
    """
    
    def __init__(self):
        pass
    
    def compute_detection_metrics(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> DetectionMetrics:
        """
        Compute detection performance metrics.
        
        Args:
            y_true: Ground truth labels (0/1)
            y_pred: Predicted labels (0/1)
            
        Returns:
            DetectionMetrics with precision, recall, F1, etc.
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        
        # Use labels=[0,1] to always get a 2x2 matrix, even with single-class data
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = recall
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        return DetectionMetrics(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            false_positive_rate=float(fpr),
            true_positive_rate=float(tpr),
            accuracy=float(accuracy),
            confusion_matrix=cm
        )
    
    def compute_stability_metrics(self, state_history: List[str],
                                   num_zones: int = 1) -> StabilityMetrics:
        """
        Compute stability metrics from state history.
        
        Args:
            state_history: List of state strings over time
            num_zones: Number of zones for averaging
            
        Returns:
            StabilityMetrics with state changes, flapping rate, etc.
        """
        state_changes = 0
        for i in range(1, len(state_history)):
            if state_history[i] != state_history[i-1]:
                state_changes += 1
        
        window_size = 20
        flapping_count = 0
        for i in range(window_size, len(state_history)):
            window = state_history[i-window_size:i]
            changes_in_window = sum(1 for j in range(1, len(window)) 
                                   if window[j] != window[j-1])
            if changes_in_window > 3:
                flapping_count += 1
        
        flapping_rate = flapping_count / max(1, len(state_history) - window_size)
        
        alert_count = sum(1 for s in state_history if s in ['ALERT', 'SUSPECTED'])
        time_in_alert = alert_count / max(1, len(state_history))
        
        return StabilityMetrics(
            total_state_changes=state_changes,
            avg_state_changes_per_zone=state_changes / max(1, num_zones),
            flapping_rate=float(flapping_rate),
            time_in_alert=float(time_in_alert)
        )
    
    def compute_lead_time(self, alert_times: List[int],
                          flood_times: List[int]) -> LeadTimeMetrics:
        """
        Compute lead time between alerts and actual floods.
        
        Args:
            alert_times: List of steps when alert was raised
            flood_times: List of steps when flood actually occurred
            
        Returns:
            LeadTimeMetrics with lead time statistics
        """
        lead_times = []
        
        for flood_time in flood_times:
            prior_alerts = [a for a in alert_times if a < flood_time]
            if prior_alerts:
                lead_time = flood_time - max(prior_alerts)
                lead_times.append(lead_time)
        
        if not lead_times:
            return LeadTimeMetrics(
                mean_lead_time=0.0,
                median_lead_time=0.0,
                std_lead_time=0.0,
                min_lead_time=0.0,
                max_lead_time=0.0,
                lead_times=[]
            )
        
        return LeadTimeMetrics(
            mean_lead_time=float(np.mean(lead_times)),
            median_lead_time=float(np.median(lead_times)),
            std_lead_time=float(np.std(lead_times)),
            min_lead_time=float(np.min(lead_times)),
            max_lead_time=float(np.max(lead_times)),
            lead_times=lead_times
        )
    
    def compute_from_logs(self, logs: pd.DataFrame) -> Dict:
        """
        Compute all metrics from simulation logs.
        
        Args:
            logs: DataFrame with columns: step, zone_id, state, ground_truth_flooded
            
        Returns:
            Dict with all computed metrics
        """
        results = {}
        
        if 'state' in logs.columns and 'ground_truth_flooded' in logs.columns:
            y_true = logs['ground_truth_flooded'].astype(int).values
            y_pred = logs['state'].apply(
                lambda x: 1 if x in ['ALERT', 'SUSPECTED'] else 0
            ).values
            
            detection = self.compute_detection_metrics(y_true, y_pred)
            results['detection'] = {
                'precision': detection.precision,
                'recall': detection.recall,
                'f1': detection.f1,
                'false_positive_rate': detection.false_positive_rate,
                'accuracy': detection.accuracy,
                'confusion_matrix': detection.confusion_matrix.tolist()
            }
        
        if 'state' in logs.columns:
            # Compute stability PER ZONE then aggregate — avoids counting
            # zone-boundary transitions as spurious state changes.
            if 'zone_id' in logs.columns:
                num_zones = logs['zone_id'].nunique()
                total_changes = 0
                weighted_flapping = 0.0
                weighted_alert = 0.0
                total_entries = 0

                for zone_id in sorted(logs['zone_id'].unique()):
                    zone_logs = logs[logs['zone_id'] == zone_id].sort_values('step')
                    zone_history = zone_logs['state'].tolist()
                    zm = self.compute_stability_metrics(zone_history, num_zones=1)
                    total_changes += zm.total_state_changes
                    weighted_flapping += zm.flapping_rate * len(zone_history)
                    weighted_alert += zm.time_in_alert * len(zone_history)
                    total_entries += len(zone_history)

                results['stability'] = {
                    'total_state_changes': total_changes,
                    'avg_state_changes_per_zone': total_changes / max(1, num_zones),
                    'flapping_rate': weighted_flapping / max(1, total_entries),
                    'time_in_alert': weighted_alert / max(1, total_entries),
                }
            else:
                state_history = logs['state'].tolist()
                stability = self.compute_stability_metrics(state_history)
                results['stability'] = {
                    'total_state_changes': stability.total_state_changes,
                    'avg_state_changes_per_zone': stability.avg_state_changes_per_zone,
                    'flapping_rate': stability.flapping_rate,
                    'time_in_alert': stability.time_in_alert
                }
        
        if 'state' in logs.columns and 'ground_truth_flooded' in logs.columns:
            # Compute lead time PER ZONE then aggregate — avoids matching
            # alerts in one zone with floods in a different zone.
            all_lead_times = []

            for zone_id in sorted(logs['zone_id'].unique()):
                zone_logs = logs[logs['zone_id'] == zone_id].sort_values('step')

                zone_alerts = []
                prev_state = 'NORMAL'
                for _, row in zone_logs.iterrows():
                    if row['state'] in ('ALERT', 'SUSPECTED') and prev_state not in ('ALERT', 'SUSPECTED'):
                        zone_alerts.append(row['step'])
                    prev_state = row['state']

                zone_floods = []
                prev_flood = False
                for _, row in zone_logs.iterrows():
                    if row['ground_truth_flooded'] and not prev_flood:
                        zone_floods.append(row['step'])
                    prev_flood = row['ground_truth_flooded']

                zone_lt = self.compute_lead_time(zone_alerts, zone_floods)
                all_lead_times.extend(zone_lt.lead_times)

            if all_lead_times:
                results['lead_time'] = {
                    'mean': float(np.mean(all_lead_times)),
                    'median': float(np.median(all_lead_times)),
                    'std': float(np.std(all_lead_times)),
                    'min': float(np.min(all_lead_times)),
                    'max': float(np.max(all_lead_times)),
                    'count': len(all_lead_times)
                }
            else:
                results['lead_time'] = {
                    'mean': 0.0, 'median': 0.0, 'std': 0.0,
                    'min': 0.0, 'max': 0.0, 'count': 0
                }
        
        return results
    
    def compare_systems(self, mas_logs: pd.DataFrame,
                        baseline_logs: pd.DataFrame) -> Dict:
        """
        Compare MAS and baseline system performance.
        
        Args:
            mas_logs: Logs from MAS system
            baseline_logs: Logs from baseline system
            
        Returns:
            Dict with comparison metrics
        """
        mas_metrics = self.compute_from_logs(mas_logs)
        baseline_metrics = self.compute_from_logs(baseline_logs)
        
        comparison = {
            'mas': mas_metrics,
            'baseline': baseline_metrics,
            'improvement': {}
        }
        
        if 'detection' in mas_metrics and 'detection' in baseline_metrics:
            for metric in ['precision', 'recall', 'f1', 'accuracy']:
                mas_val = mas_metrics['detection'].get(metric, 0)
                base_val = baseline_metrics['detection'].get(metric, 0)
                comparison['improvement'][f'detection_{metric}'] = mas_val - base_val
        
        if 'stability' in mas_metrics and 'stability' in baseline_metrics:
            mas_changes = mas_metrics['stability'].get('total_state_changes', 0)
            base_changes = baseline_metrics['stability'].get('total_state_changes', 0)
            comparison['improvement']['stability_reduction'] = base_changes - mas_changes
        
        return comparison
