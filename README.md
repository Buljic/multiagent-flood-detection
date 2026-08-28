# FloodMAS - Multi-Agent Flood Early-Warning System

A multi-agent system for early flood detection using Mesa simulation framework, machine learning, and robust guardrails for stable decision-making.

## Overview

FloodMAS implements an end-to-end flood detection pipeline:

- **Simulation**: Realistic hydro-meteorological signal generation (rainfall, water levels, soil saturation)
- **Multi-Agent Architecture**: Sensor → Edge → Coordinator (distributed decisions)
- **ML Model**: Tabular classifier for flood risk prediction
- **Guardrails**: State machine with hysteresis, debouncing, consensus gating, and health-aware degradation
- **Baseline Comparison**: Threshold-based heuristic for benchmarking
- **BI Dashboard**: Interactive Streamlit visualization

## Project Structure

```
FloodMAS/
├── configs/
│   ├── default.yaml        # Main configuration
│   └── scenarios.yaml      # Experiment scenarios
├── sim/
│   ├── environment.py      # Flood simulation environment
│   ├── agents.py           # Mesa agents (Sensor, Edge, Coordinator, Mitigation)
│   ├── guardrails.py       # State machine, hysteresis, consensus
│   └── model.py            # Main Mesa model
├── ml/
│   ├── generate_data.py    # Synthetic data generation
│   └── train.py            # ML model training
├── baseline/
│   └── threshold.py        # Threshold baseline system
├── eval/
│   ├── metrics.py          # Evaluation metrics
│   └── run_experiments.py  # Experiment runner
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── outputs/
│   ├── datasets/           # Generated training data
│   ├── models/             # Trained models
│   ├── logs/               # Simulation logs
│   └── experiments/        # Experiment results
├── requirements.txt
└── README.md
```

## Installation

1. Create virtual environment:
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Training Data

```bash
python -m ml.generate_data --config configs/default.yaml \
    --episodes 2000 --steps 400 \
    --out outputs/datasets/sim.parquet
```

### 2. Train ML Model

```bash
python -m ml.train --data outputs/datasets/sim.parquet \
    --model rf --out outputs/models/risk_model.pkl \
    --report outputs/models/train_report.json
```

### 3. Run Simulation

```bash
python -m sim.model --config configs/default.yaml \
    --model outputs/models/risk_model.pkl \
    --log outputs/logs/run_001.parquet \
    --scenario extreme_wet
```

Valid scenarios: `normal_wet`, `normal_dry`, `extreme_wet`, `extreme_dry`, `extreme_dropout_10`, `extreme_dropout_30`, `extreme_dropout_50`, `extreme_noisy`

### 4. Run Experiments

```bash
python -m eval.run_experiments --config configs/default.yaml \
    --scenarios-config configs/scenarios.yaml \
    --model outputs/models/risk_model.pkl \
    --out outputs/experiments/results.json
```

### 5. Generate Figures

```bash
python -m eval.make_figures --results outputs/experiments/results.json \
    --output outputs/figures
```

### 6. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Architecture

### Multi-Agent System

```
┌─────────────────────────────────────────────────────────────┐
│                     CoordinatorAgent                         │
│           (Global fusion, alarm management)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
│ EdgeAggregator│ │ EdgeAggregator│ │ EdgeAggregator│
│   (Zone 0)    │ │   (Zone 1)    │ │   (Zone N)    │
│ ML + Guards   │ │ ML + Guards   │ │ ML + Guards   │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
   ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
   │Sensors  │       │Sensors  │       │Sensors  │
   │(noisy)  │       │(noisy)  │       │(noisy)  │
   └─────────┘       └─────────┘       └─────────┘
```

### Guardrails State Machine

```
    NORMAL ─────► SUSPECTED ─────► ALERT
      ▲               │               │
      │               │               │
      └───────────────┴───────◄───────┘
                          COOLDOWN
```

**Stability mechanisms:**
- **Hysteresis**: Different thresholds for up/down transitions (TH_UP > TH_DOWN)
- **Debouncing**: K consecutive steps required before state change
- **Consensus Gating**: Minimum sensor agreement required
- **Health-Aware Degradation**: Stricter thresholds when sensors fail

## Configuration

All parameters in `configs/default.yaml`:

| Section | Parameter | Description |
|---------|-----------|-------------|
| `simulation` | `grid_size` | Environment grid dimensions |
| `simulation` | `num_zones` | Number of monitoring zones |
| `sensors` | `noise_std` | Sensor noise level |
| `sensors` | `dropout_rate` | Sensor failure probability |
| `ml` | `horizon_T` | Prediction horizon (steps) |
| `guardrails` | `TH_UP/TH_DOWN` | Hysteresis thresholds |
| `guardrails` | `K_UP/K_DOWN` | Debounce counters |
| `guardrails` | `CONS_MIN` | Minimum consensus |
| `guardrails` | `HEALTH_MIN` | Health degradation threshold |

## Metrics

| Metric | Description |
|--------|-------------|
| **Precision** | Correct alerts / Total alerts |
| **Recall** | Detected floods / Total floods |
| **F1** | Harmonic mean of precision/recall |
| **FPR** | False alarms / Total non-floods |
| **Lead Time** | Steps between alert and flood |
| **Stability** | State changes (flapping indicator) |

## Experimental Results

The MAS system typically shows:
- **Higher F1**: Better detection accuracy vs threshold baseline
- **Lower flapping**: Guardrails reduce unnecessary state changes
- **Better robustness**: Maintains performance under sensor dropout

## ML Features

| Feature | Description |
|---------|-------------|
| `water_mean_5` | Rolling mean water level (5 steps) |
| `water_slope_5` | Water level trend |
| `water_max_10` | Maximum water (10 steps) |
| `rain_sum_20` | Cumulative rainfall (20 steps) |
| `rain_mean_10` | Mean rainfall (10 steps) |
| `soil_mean_10` | Mean soil saturation |
| `consensus` | Fraction of sensors with rising trend |
| `health` | Fraction of operational sensors |

## Countermeasures (Optional)

Enable in config to add mitigation agents:
- **Pump**: Reduces water level in alert zones
- **Gate**: Reduces upstream inflow

```yaml
countermeasures:
  enabled: true
  pump_capacity: 0.05
  gate_reduction: 0.3
```

## License

MIT License

## Citation

```bibtex
@inproceedings{buljic2026multiagent,
  title     = {Using Multi-Agent Systems For Predicting and Preventing Natural Disasters},
  author    = {Bulji{'c}, Imad and Goran, Nermin and Hod{\v z}i{\'c}, Mujo},
  booktitle = {Proceedings of the International Symposium on Innovative and Interdisciplinary Applications of Advanced Technologies (IAT 2026)},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```

## Citation

If you use this repository in your research, please cite:

> Imad Buljić, Nermin Goran, and Mujo Hodžić. "Using Multi-Agent Systems For Predicting and Preventing Natural Disasters." In *Proceedings of the International Symposium on Innovative and Interdisciplinary Applications of Advanced Technologies (IAT) 2026*. Springer (to appear).

- Repository: https://github.com/Buljic/multiagent-flood-detection
- Authors:
  - Imad Buljić - [ORCID 0009-0009-2723-2315](https://orcid.org/0009-0009-2723-2315) - [imadbuljic.com](https://imadbuljic.com)
  - Nermin Goran - [ORCID 0000-0002-0905-0843](https://orcid.org/0000-0002-0905-0843)
  - Mujo Hodžić - [ORCID 0000-0002-0015-8268](https://orcid.org/0000-0002-0015-8268)

The paper will be linked here with its Springer DOI once published.
