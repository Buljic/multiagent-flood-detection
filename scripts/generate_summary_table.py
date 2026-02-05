"""
Generate summary table from experiment results.
Creates CSV/Markdown table with key metrics per scenario.
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def load_results(results_path: str) -> dict:
    """Load experiment results JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def generate_summary_table(results: dict) -> pd.DataFrame:
    """Generate summary table from experiment results."""
    rows = []
    
    for scenario in results.get('scenarios', []):
        scenario_name = scenario['scenario_name']
        agg = scenario.get('aggregated', {})
        
        # MAS metrics
        mas_detection = agg.get('mas', {}).get('detection', {})
        mas_stability = agg.get('mas', {}).get('stability', {})
        mas_leadtime = agg.get('mas', {}).get('lead_time', {})
        
        rows.append({
            'scenario': scenario_name,
            'mode': 'MAS+ML',
            'f1': mas_detection.get('f1', {}).get('mean', 0),
            'f1_std': mas_detection.get('f1', {}).get('std', 0),
            'precision': mas_detection.get('precision', {}).get('mean', 0),
            'recall': mas_detection.get('recall', {}).get('mean', 0),
            'leadtime_mean': mas_leadtime.get('mean', {}).get('mean', 0),
            'leadtime_std': mas_leadtime.get('std', {}).get('mean', 0),
            'state_changes_mean': mas_stability.get('total_state_changes', {}).get('mean', 0),
            'state_changes_std': mas_stability.get('total_state_changes', {}).get('std', 0),
            'flapping_rate': mas_stability.get('flapping_rate', {}).get('mean', 0)
        })
        
        # Baseline metrics
        baseline_detection = agg.get('baseline', {}).get('detection', {})
        baseline_stability = agg.get('baseline', {}).get('stability', {})
        baseline_leadtime = agg.get('baseline', {}).get('lead_time', {})
        
        rows.append({
            'scenario': scenario_name,
            'mode': 'Baseline',
            'f1': baseline_detection.get('f1', {}).get('mean', 0),
            'f1_std': baseline_detection.get('f1', {}).get('std', 0),
            'precision': baseline_detection.get('precision', {}).get('mean', 0),
            'recall': baseline_detection.get('recall', {}).get('mean', 0),
            'leadtime_mean': baseline_leadtime.get('mean', {}).get('mean', 0),
            'leadtime_std': baseline_leadtime.get('std', {}).get('mean', 0),
            'state_changes_mean': baseline_stability.get('total_state_changes', {}).get('mean', 0),
            'state_changes_std': baseline_stability.get('total_state_changes', {}).get('std', 0),
            'flapping_rate': baseline_stability.get('flapping_rate', {}).get('mean', 0)
        })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Generate summary table from experiment results')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to experiment results JSON')
    parser.add_argument('--out', type=str, required=True,
                        help='Output path for summary table (CSV or MD)')
    
    args = parser.parse_args()
    
    results = load_results(args.results)
    df = generate_summary_table(results)
    
    # Round for readability
    df = df.round(3)
    
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.csv':
        df.to_csv(output_path, index=False)
    elif output_path.suffix == '.md':
        df.to_markdown(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    print(f"Summary table saved to {output_path}")
    print("\nPreview:")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
