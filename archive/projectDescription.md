# FloodMAS: Multi-Agent Flood Early-Warning System

## Project Description (Internal Documentation)

**Authors:** Imad, Goran, Nermin  
**Purpose:** IEEE Publication / Scientific Research  
**Status:** Complete Implementation

---

## 1. Project Goal

Build an **end-to-end Multi-Agent Flood Early-Warning System** that:
1. Simulates realistic hydro-meteorological conditions (rainfall, water level, soil saturation)
2. Uses distributed Multi-Agent architecture (Sensor → Edge → Coordinator)
3. Employs ML models for flood risk prediction
4. Implements robust guardrails to prevent false alarms and system instability
5. Provides measurable comparison against a baseline heuristic
6. Generates reproducible results suitable for IEEE publication

---

## 2. What Was Implemented

### 2.1 Core Simulation Environment (`sim/environment.py`)

**FloodEnvironment** class simulates:
- **Grid-based terrain** with elevation and river path
- **Water dynamics**: spreading, evaporation, upstream inflow
- **Soil saturation**: infiltration, drainage
- **Rainfall events**: configurable scenarios (normal/extreme)
- **Zone management**: divides grid into monitoring zones

**Why:** Provides realistic synthetic data generation without needing real sensor deployments. The simulation captures key physical dynamics that affect flood formation.

### 2.2 Multi-Agent System (`sim/agents.py`, `sim/model.py`)

Four agent types implemented using **Mesa 3.x** framework:

| Agent | Role | Key Functions |
|-------|------|---------------|
| **SensorAgent** | Emit noisy readings | Simulates sensor noise, dropout, trend detection |
| **EdgeAggregatorAgent** | Zone-level processing | Feature extraction, ML inference, guardrails state machine |
| **CoordinatorAgent** | Global fusion | Aggregates zone risks, manages global alarm |
| **MitigationAgent** | Countermeasures | Pump/gate actions when in ALERT (optional) |

**Why:** Distributed architecture mirrors real IoT deployments. Edge processing reduces latency and enables local decision-making even if network fails.

### 2.3 Guardrails System (`sim/guardrails.py`)

Implements stability mechanisms to prevent "flapping" and false alarms:

| Mechanism | Implementation | Purpose |
|-----------|----------------|---------|
| **Hysteresis** | TH_UP=0.6, TH_DOWN=0.4 | Different thresholds for up/down transitions |
| **Debouncing** | K_UP=3, K_DOWN=5 consecutive steps | Requires sustained signal before state change |
| **Consensus Gating** | CONS_MIN=0.5 | Minimum sensor agreement required |
| **Health-Aware Degradation** | HEALTH_MIN=0.6 | Stricter thresholds when sensors fail |
| **Outlier Clipping** | OutlierClipper class | Limits unrealistic jumps (max_delta=0.3) |
| **Missing Value Handling** | MissingValueHandler class | Last-value carry-forward with health penalty |

**State Machine:**
```
NORMAL → SUSPECTED → ALERT → COOLDOWN → NORMAL
```

**Why:** Critical for IEEE acceptance. Raw ML predictions fluctuate too much. Guardrails ensure operational stability and reduce false positive rate.

### 2.4 Synthetic Data Generation (`ml/generate_data.py`)

**DataGenerator** creates training datasets by:
1. Running N episodes (default 2000) of simulation
2. Randomizing scenario parameters per episode (storm type, soil, dropout)
3. Extracting features at each timestep
4. Computing ground truth label: `flood_in_next_T` (T=10 steps)

**Feature Set (8 features):**
- `water_mean_5`: 5-step rolling mean of water level
- `water_slope_5`: 5-step trend slope
- `water_max_10`: 10-step rolling max
- `rain_sum_20`: 20-step cumulative rainfall
- `rain_mean_10`: 10-step rain average
- `soil_mean_10`: 10-step soil saturation average
- `consensus`: Fraction of sensors showing rising trend
- `health`: Fraction of operational sensors

**No Data Leakage:** Features use only past data (ring buffers). Label computed from future states (steps t+1 to t+T).

**Class Balancing:** 60% extreme scenarios to ensure sufficient positive samples.

**Why:** Synthetic data allows controlled experiments. Real data (USGS, FlowDB) can be added later for validation.

### 2.5 ML Training Pipeline (`ml/train.py`)

**ModelTrainer** supports:
- **RandomForestClassifier** (default): robust, interpretable
- **GradientBoostingClassifier**: alternative option
- **Isotonic Calibration** (5-fold CV): ensures predicted probabilities are well-calibrated
- **class_weight='balanced'**: handles class imbalance
- **Stratified split**: 80/20 train/test with same class ratio

**Output Metrics:**
- AUC-ROC, F1, Precision, Recall, Accuracy
- **Brier Score**: measures calibration quality (lower is better)
- Feature importance ranking
- Cross-validation scores

**Why:** Calibrated probabilities are essential for threshold-based alerting. Brier score proves calibration isn't overfit.

### 2.6 Baseline System (`baseline/threshold.py`)

**ThresholdBaseline** with **minimal guardrails** for fair comparison:
- Simple threshold logic: ALERT if water > TH and rain > TH
- **Minimal hysteresis**: TH_DOWN = TH_UP - 0.1 (fair comparison)
- **Minimal debouncing**: 2 consecutive steps (fair comparison)
- No consensus, no health-awareness (simpler than MAS)

**Why:** Baseline must have minimal guardrails or reviewers will say comparison is unfair. MAS advantage comes from ML + full guardrails + consensus.

### 2.7 Evaluation Metrics (`eval/metrics.py`)

**MetricsCalculator** computes:

| Category | Metrics | Definition |
|----------|---------|------------|
| **Detection** | Precision, Recall, F1, FPR, Accuracy | Standard classification metrics |
| **Lead Time** | Mean, Median, Std, Min, Max | Time from first ALERT to actual flood |
| **Stability** | State changes, Flapping rate, Time in alert | Measures system stability |

**Lead Time Definition:** "Time from first ALERT state to the moment ground truth flood occurs" - clearly defined for IEEE.

**Flapping Detection:** Counts windows with >3 state changes in 20 steps.

**Why:** These metrics directly address IEEE reviewer concerns about practical deployment.

### 2.8 Experiment Runner (`eval/run_experiments.py`)

**ExperimentRunner** orchestrates:
1. Runs multiple scenarios (normal, extreme, dropout 0/10/30/50%, noisy)
2. Multiple repeats per scenario (default 3) for statistical significance
3. Compares MAS vs baseline on same data
4. **Logs run metadata** for reproducibility:
   - Timestamp, seed, config hash, model hash
   - All guardrails and baseline parameters
   - Sensor configuration

**Why:** Reproducibility is #1 IEEE criticism. Full metadata ensures any reviewer can replicate results.

### 2.9 BI Dashboard (`dashboard/app.py`)

**Streamlit dashboard** provides:
- Timeline visualization (risk, state, ground truth)
- Confusion matrix display
- F1/PR curves
- Lead time distribution
- MAS vs Baseline comparison charts
- Zone-level filtering

**Why:** Visual results are essential for paper figures and presentations.

---

## 3. How to Use

### Quick Start (Training Pipeline)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic training data (runs simulation episodes)
python -m ml.generate_data --episodes 500 --steps 300 --out outputs/datasets/sim.parquet

# 3. Train ML model (produces risk_model.pkl + train_report.json)
python -m ml.train --data outputs/datasets/sim.parquet --out outputs/models/risk_model.pkl

# 4. Run experiments comparing MAS vs baseline
python -m eval.run_experiments --model outputs/models/risk_model.pkl

# 5. Launch dashboard to visualize results
streamlit run dashboard/app.py
```

**Yes, you can train the model without running the dashboard.** Steps 2 and 3 are standalone.

### Configuration

All parameters in `configs/default.yaml`:
- `seed`: Random seed for reproducibility
- `simulation`: Grid size, zones, steps
- `sensors`: Noise, dropout, outlier clipping
- `guardrails`: All hysteresis/debounce parameters
- `ml`: Prediction horizon T
- `baseline`: Threshold values

Scenarios in `configs/scenarios.yaml`:
- 8 predefined scenarios for robustness testing
- Vary rainfall type, soil saturation, dropout rate, noise level

---

## 4. IEEE Checklist Verification

### ✅ 1. Reproducibility
- [x] `seed` in config, used everywhere (numpy, Mesa)
- [x] Run metadata logged (timestamp, seed, config hash, model hash)
- [x] `run_experiments.py` produces identical results with same seed

### ✅ 2. Dataset Quality
- [x] Label `flood_in_next_T` precisely defined (T=10 steps lookahead)
- [x] No data leakage (features from ring buffers, label from future)
- [x] Class balancing via `class_weight='balanced'` + stratified sampling
- [x] Isotonic calibration with 5-fold CV (not overfit)
- [x] **Brier score** in training report proves calibration quality

### ✅ 3. Guardrails
- [x] Hysteresis (TH_UP ≠ TH_DOWN)
- [x] Debouncing (K_UP, K_DOWN)
- [x] Consensus gating (CONS_MIN)
- [x] Health-aware degradation (HEALTH_MIN)
- [x] Missing value handling (carry-forward + penalty)
- [x] Outlier clipping (max_delta limit)

### ✅ 4. Fair Baseline Comparison
- [x] Baseline has **minimal hysteresis** (TH_DOWN = TH_UP - 0.1)
- [x] Baseline has **minimal debouncing** (2 steps)
- [x] Baseline is simpler but not trivially broken

### ✅ 5. Metrics Methodology
- [x] Lead time: clearly defined as "time from first ALERT to flood"
- [x] Stability: state changes per episode/zone, flapping rate
- [x] Robustness: tested across dropout 0/10/30/50%

---

## 5. Key Design Decisions

### Why Mesa Framework?
- Industry-standard agent-based modeling
- Easy to extend with new agent types
- Built-in scheduling and data collection

### Why RandomForest over Deep Learning?
- Tabular data (8 features) - RF excels here
- Interpretable feature importance for paper
- Fast training, no GPU required
- Calibration works well with ensemble methods

### Why Synthetic Data?
- Full control over ground truth labels
- Can generate unlimited training samples
- Reproducible experiments
- Real data can be added later for validation

### Why 4-State Machine (not 2)?
- SUSPECTED provides early warning without full alert
- COOLDOWN prevents rapid re-triggering after alert ends
- More gradual transitions = fewer false alarms

---

## 6. File Structure

```
MultiAgent_Flood_Detection/
├── configs/
│   ├── default.yaml          # Main configuration
│   └── scenarios.yaml        # Experiment scenarios
├── sim/
│   ├── environment.py        # FloodEnvironment simulation
│   ├── agents.py             # All Mesa agents
│   ├── guardrails.py         # State machine, buffers, clippers
│   └── model.py              # FloodModel orchestration
├── ml/
│   ├── generate_data.py      # Synthetic dataset generation
│   └── train.py              # ML training pipeline
├── baseline/
│   └── threshold.py          # Threshold baseline with minimal guardrails
├── eval/
│   ├── metrics.py            # Detection, stability, lead time metrics
│   └── run_experiments.py    # Multi-scenario experiment runner
├── dashboard/
│   └── app.py                # Streamlit visualization
├── tests/
│   └── test_system.py        # Integration tests
├── outputs/
│   ├── datasets/             # Generated .parquet files
│   ├── models/               # Trained .pkl models + reports
│   ├── logs/                 # Simulation logs
│   └── experiments/          # Experiment results JSON
├── requirements.txt
├── README.md                 # Public documentation
└── projectDescription.md     # This file (internal)
```

---

## 7. Future Work

1. **VisionAgent**: Add satellite/drone imagery processing
2. **Real-World Validation**: Test with USGS/FlowDB data
3. **Edge Deployment**: Optimize for Raspberry Pi / edge devices
4. **Federated Learning**: Train across distributed zones
5. **Explainability**: SHAP values for ML decisions

---

## 8. Known Limitations

1. **Synthetic Only**: Currently no real sensor data validation
2. **2D Grid**: Simplified terrain model (no 3D hydraulics)
3. **Single River**: One river path per grid
4. **No Cascading**: Zone alerts don't influence neighboring zones

---

*This document is for internal use and should be in .gitignore for the public repository.*
