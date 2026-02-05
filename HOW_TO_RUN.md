# How to Run FloodMAS - Complete Guide

## Prerequisites

```powershell
# Verify Python 3.10+
python --version

# Install dependencies
pip install -r requirements.txt
```

---

## Option A: Smoke Test (5 minutes)

Quick validation that entire pipeline works.

```powershell
.\scripts\smoke_test.ps1
```

**Expected outputs:**
- `outputs/datasets/smoke.parquet` (dataset)
- `outputs/models/smoke_model.pkl` (trained model)
- `outputs/models/smoke_report.json` (training metrics)
- `outputs/logs/smoke_normal_wet.parquet` (simulation log)
- `outputs/experiments/smoke_results.json` (experiment results)

---

## Option B: Full Scenario Test (30-60 minutes)

Run all 8 scenarios for publication-quality results.

```powershell
.\scripts\full_test.ps1
```

**Expected outputs:**
- `outputs/datasets/full_train.parquet` (1000 episodes)
- `outputs/models/full_model.pkl` (production model)
- `outputs/models/full_report.json` (AUC, F1, Brier score, feature importance)
- `outputs/experiments/full_results.json` (all scenarios, 5 repeats each)
- `outputs/experiments/summary_table.csv` (comparison table)

**Summary table columns:**
- scenario, mode (MAS+ML / Baseline)
- f1, f1_std, precision, recall
- leadtime_mean, leadtime_std
- state_changes_mean, state_changes_std
- flapping_rate

---

## Option C: Manual Step-by-Step

### Step 1: Generate Training Data

```powershell
python -m ml.generate_data --episodes 1000 --steps 300 --out outputs/datasets/train_data.parquet --seed 42
```

**Parameters:**
- `--episodes`: Number of simulation runs (more = better model, 1000-2000 recommended)
- `--steps`: Steps per episode (300-400 typical)
- `--seed`: Random seed for reproducibility

**Output:** Parquet file with columns:
- Features: water_mean_5, water_slope_5, water_max_10, rain_sum_20, rain_mean_10, soil_mean_10, consensus, health
- Label: flood_in_next_T (binary)
- Metadata: episode_id, step, zone_id, scenario, dropout_rate, noise_std

### Step 2: Train ML Model

```powershell
python -m ml.train --data outputs/datasets/train_data.parquet --model rf --out outputs/models/risk_model.pkl --report outputs/models/train_report.json --seed 42
```

**Parameters:**
- `--model`: rf (RandomForest) or gb (GradientBoosting)
- `--no-calibrate`: Skip isotonic calibration (not recommended)
- `--seed`: Random seed

**Output:**
- `risk_model.pkl`: Trained + calibrated model
- `train_report.json`: Metrics including:
  - AUC-ROC, F1, Precision, Recall
  - Brier score (calibration quality)
  - Feature importance
  - Cross-validation scores

**Expected metrics:**
- AUC-ROC: > 0.95
- F1: > 0.90
- Brier: < 0.05 (excellent calibration)

### Step 3: Run Single Simulation

```powershell
python -m sim.model --config configs/default.yaml --model outputs/models/risk_model.pkl --scenario extreme_wet --steps 400 --log outputs/logs/run_extreme_wet.parquet
```

**Valid scenarios:**
- `normal_wet`, `normal_dry` (operational)
- `extreme_wet`, `extreme_dry` (natural stress)
- `extreme_dropout_10`, `extreme_dropout_30`, `extreme_dropout_50` (robustness)
- `extreme_noisy` (sensor degradation)

**Output:**
- `run_extreme_wet.parquet`: Per-step logs (zone_id, step, risk, state, features, ground_truth)
- `run_extreme_wet_coordinator.parquet`: Global coordinator logs

### Step 4: Run Experiments (All Scenarios)

```powershell
python -m eval.run_experiments --config configs/default.yaml --scenarios-config configs/scenarios.yaml --model outputs/models/risk_model.pkl --out outputs/experiments/results.json --steps 400 --repeats 5
```

**Parameters:**
- `--repeats`: Number of runs per scenario (5 recommended for std calculation)
- `--steps`: Simulation length per run

**Output:** `results.json` structure:
```json
{
  "run_metadata": {
    "timestamp": "...",
    "seed": 42,
    "config_hash": "...",
    "model_hash": "...",
    "guardrails_params": {...},
    "baseline_params": {...}
  },
  "scenarios": [
    {
      "scenario_name": "normal_wet",
      "runs": [...],
      "aggregated": {
        "mas": {"detection": {...}, "stability": {...}, "lead_time": {...}},
        "baseline": {...}
      }
    },
    ...
  ],
  "summary": {
    "mas_avg_f1": 0.94,
    "baseline_avg_f1": 0.85,
    "f1_improvement": 0.09,
    "mas_avg_state_changes": 8,
    "baseline_avg_state_changes": 35,
    "stability_improvement": 27
  }
}
```

### Step 5: Generate Summary Table

```powershell
python scripts/generate_summary_table.py --results outputs/experiments/results.json --out outputs/experiments/summary_table.csv
```

**Output:** CSV with one row per (scenario, mode) pair.

### Step 6: Generate Publication Figures

```powershell
python -m eval.make_figures --results outputs/experiments/results.json --logs-dir outputs/logs --output outputs/figures
```

**Output:** 5 figures + TXT explanations:
- `Fig1_timeline_normal_wet.png` (+ .txt)
- `Fig1_timeline_extreme_wet.png` (+ .txt)
- `Fig1_timeline_extreme_dropout_50.png` (+ .txt)
- `Fig2_confusion_hero.png` (+ .txt)
- `Fig3_leadtime_boxplot.png` (+ .txt)
- `Fig4_robustness_dropout.png` (+ .txt)
- `Fig5_flapping_stability.png` (+ .txt)

**Note:** Fig1 timeline requires individual scenario logs. To generate them:

```powershell
# Run hero scenarios individually and save logs
python -m sim.model --model outputs/models/risk_model.pkl --scenario normal_wet --steps 400 --log outputs/logs/normal_wet_mas.parquet

python -m sim.model --model outputs/models/risk_model.pkl --scenario extreme_wet --steps 400 --log outputs/logs/extreme_wet_mas.parquet

python -m sim.model --model outputs/models/risk_model.pkl --scenario extreme_dropout_50 --steps 400 --log outputs/logs/extreme_dropout_50_mas.parquet
```

For baseline comparison logs, modify baseline/threshold.py to save logs or run experiments with `--save-logs` flag.

### Step 7: Launch Dashboard

```powershell
streamlit run dashboard/app.py
```

Browser opens at `http://localhost:8501`

**Features:**
- Load experiment results JSON
- View timeline plots per scenario
- Compare MAS vs Baseline metrics
- Filter by zone, scenario
- Export figures

---

## Troubleshooting

### Issue: "Invalid scenario 'extreme'"

**Fix:** Use full scenario name from `configs/scenarios.yaml`:
- ✅ `extreme_wet`
- ❌ `extreme`

List valid scenarios:
```powershell
python -m sim.model --scenario INVALID 2>&1 | Select-String "Valid scenarios"
```

### Issue: "No module named 'mesa'"

**Fix:** Install dependencies:
```powershell
pip install -r requirements.txt
```

### Issue: Training fails with "Only one class"

**Cause:** Dataset has no positive samples (no floods in short runs)

**Fix:** Increase episodes or steps:
```powershell
python -m ml.generate_data --episodes 500 --steps 300 ...
```

Ensure positive rate > 5% in output:
```
Positive rate: 0.208  ✓ Good (20.8%)
Positive rate: 0.000  ✗ Bad (need more episodes)
```

### Issue: Figures not generating

**Cause:** Missing logs for hero scenarios

**Fix:** Run individual simulations for each hero scenario first (see Step 6 above)

### Issue: "Agent.__init__() missing 1 required positional argument"

**Cause:** Old Mesa 2.x code

**Fix:** Already fixed in this version. Update your local repo.

---

## Understanding Output Files

### Training Report (`train_report.json`)

```json
{
  "auc_roc": 0.9952,          // Discrimination ability (higher = better)
  "f1": 0.9701,               // Balanced accuracy (higher = better)
  "precision": 0.9910,        // Alert correctness (higher = fewer false alarms)
  "recall": 0.9501,           // Flood detection rate (higher = fewer misses)
  "brier_score": 0.0106,      // Calibration quality (lower = better, <0.05 excellent)
  "feature_importance": {
    "water_max_10": 0.264,    // Most important feature
    "rain_sum_20": 0.241,
    "rain_mean_10": 0.178,
    "water_slope_5": 0.173,
    ...
  },
  "cv_auc_mean": 0.9915,      // Cross-validation score
  "cv_auc_std": 0.0018        // Consistency across folds
}
```

### Experiment Results Summary

```json
{
  "summary": {
    "mas_avg_f1": 0.94,                    // MAS average F1 across scenarios
    "baseline_avg_f1": 0.85,               // Baseline average F1
    "f1_improvement": 0.09,                // +9% improvement
    "mas_avg_state_changes": 8,            // MAS stability (fewer = better)
    "baseline_avg_state_changes": 35,      // Baseline stability
    "stability_improvement": 27            // 27 fewer state changes
  }
}
```

**Key insights:**
- F1 improvement > 0.05 → significant
- Stability improvement > 20 → much more stable
- Lead time > 15 steps → adequate warning

### Summary Table (`summary_table.csv`)

| scenario | mode | f1 | precision | recall | leadtime_mean | state_changes_mean | flapping_rate |
|----------|------|-----|-----------|--------|---------------|-------------------|---------------|
| normal_wet | MAS+ML | 0.92 | 0.95 | 0.89 | 23.4 | 6 | 0.02 |
| normal_wet | Baseline | 0.87 | 0.82 | 0.93 | 21.1 | 28 | 0.15 |
| extreme_dropout_50 | MAS+ML | 0.82 | 0.88 | 0.77 | 18.2 | 12 | 0.08 |
| extreme_dropout_50 | Baseline | 0.45 | 0.52 | 0.40 | 8.3 | 52 | 0.42 |

**Reading the table:**
- MAS maintains F1 > 0.8 even at 50% dropout
- Baseline collapses to F1 = 0.45 at 50% dropout
- MAS has 50-80% fewer state changes (more stable)

---

## Configuration Tuning

### Guardrails (`configs/default.yaml` → `guardrails:`)

| Parameter | Default | Purpose | Tuning Guide |
|-----------|---------|---------|-------------|
| `TH_UP` | 0.6 | Alert trigger threshold | ↑ = fewer false alarms, ↓ = more sensitive |
| `TH_DOWN` | 0.4 | Alert clear threshold | Must be < TH_UP (hysteresis gap) |
| `K_UP` | 3 | Steps before ALERT | ↑ = more stable, ↓ = faster response |
| `K_DOWN` | 5 | Steps before clearing | ↑ = longer alerts, ↓ = faster recovery |
| `CONS_MIN` | 0.5 | Min sensor consensus | ↑ = require more agreement |
| `HEALTH_MIN` | 0.6 | Health degradation threshold | Below this → stricter rules |

**Trade-off:** Stability ↔ Lead Time
- More debouncing (↑ K_UP) = more stable but ↓ lead time
- Less debouncing (↓ K_UP) = more lead time but ↑ flapping

**Recommended tuning process:**
1. Run experiments with default values
2. If F1 < 0.9: decrease TH_UP to 0.5
3. If flapping_rate > 0.1: increase K_UP to 5
4. If lead_time < 15: decrease TH_UP or K_UP
5. Re-run experiments, iterate

### Baseline (`configs/default.yaml` → `baseline:`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `water_threshold` | 0.5 | Water level trigger |
| `rain_threshold` | 2.0 | Rainfall trigger |
| `hysteresis_margin` | 0.1 | TH_DOWN = TH_UP - margin |
| `debounce_steps` | 2 | Minimal debouncing |

Keep baseline simple for fair comparison.

---

## Reproducibility Checklist

Before submitting results to IEEE:

- [ ] All commands use `--seed 42` (or document seed in paper)
- [ ] `run_metadata` in results.json includes config_hash
- [ ] requirements.txt has exact versions (`pip freeze > requirements.txt`)
- [ ] README.md lists exact commands used
- [ ] Training report shows episode-based split (no data leakage)
- [ ] Brier score < 0.05 (proves calibration quality)
- [ ] Summary table shows mean ± std (not just mean)
- [ ] Figures include error bars (if multiple repeats)
- [ ] CONCEPTS.md explains all design choices

---

## Quick Reference

**Generate data:**
```powershell
python -m ml.generate_data --episodes 1000 --steps 300 --out outputs/datasets/data.parquet --seed 42
```

**Train model:**
```powershell
python -m ml.train --data outputs/datasets/data.parquet --out outputs/models/model.pkl --seed 42
```

**Run simulation:**
```powershell
python -m sim.model --model outputs/models/model.pkl --scenario extreme_wet --steps 400 --log outputs/logs/run.parquet
```

**Run experiments:**
```powershell
python -m eval.run_experiments --model outputs/models/model.pkl --out outputs/experiments/results.json --repeats 5
```

**Generate figures:**
```powershell
python -m eval.make_figures --results outputs/experiments/results.json --output outputs/figures
```

**Generate summary table:**
```powershell
python scripts/generate_summary_table.py --results outputs/experiments/results.json --out outputs/experiments/summary.csv
```

**Launch dashboard:**
```powershell
streamlit run dashboard/app.py
```

---

## Expected Runtimes (approximate)

| Task | Episodes/Steps | Runtime | Output Size |
|------|---------------|---------|-------------|
| Smoke test | 50 eps × 100 steps | 2 min | ~10 MB |
| Training data | 1000 eps × 300 steps | 10 min | ~100 MB |
| Model training | - | 2 min | ~5 MB |
| Single simulation | 1 run × 400 steps | 5 sec | ~1 MB |
| Experiments (8 scenarios, 5 repeats) | 40 runs × 400 steps | 5 min | ~20 MB |
| Figure generation | - | 10 sec | ~5 MB (PNGs) |
| Full test | All above | 30-60 min | ~150 MB total |

*Runtimes on typical laptop (i5/i7, 8GB RAM)*

---

## Contact & Support

For bugs or questions:
1. Check CONCEPTS.md for conceptual explanations
2. Check projectDescription.md for implementation details
3. Review error messages in terminal
4. Verify all dependencies installed: `pip list | Select-String "mesa|sklearn|streamlit"`

**Common mistakes:**
- ❌ Using `--scenario extreme` (too vague)
- ✅ Using `--scenario extreme_wet` (specific)
- ❌ Forgetting `--seed` (not reproducible)
- ✅ Always using `--seed 42`
- ❌ Short episodes (<100 steps) → no floods
- ✅ Use 300-400 steps per episode
