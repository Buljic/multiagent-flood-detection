# CHANGELOG — MultiAgent Flood Detection System

## v2.12 - Noise-Consistency Re-Run & Camera-Ready Re-Sync (2026-08-27)

A noise-consistency fix in `ml/generate_data.py` (training-time health and
consensus features aligned with the runtime edge agents) required a retrain,
so the full deterministic pipeline was re-run (2,000 episodes x 400 steps,
seed 42; 8 scenarios x 3 repeats, seeds 42/1042/2042; hero sims). The fresh
canonical artifacts are in `FINAL_RESULTS/` (results.json 2026-08-27T18:38,
final_report.json, hero parquets, final_model.pkl). Every paper number was
re-derived from these artifacts and the docx/PDF/figures re-synced
(CORRECTIONS.md sixth pass):

1. **Model metrics (Table 2)**: ROC-AUC 0.999 -> 0.998, F1 0.992 -> 0.990,
   Recall 0.990 -> 0.985, Brier 0.007 -> 0.009, CV 0.998 +/- 0.000,
   Accuracy 0.989; confusion matrix TN 265,460 / FP 1,654 / FN 5,018 /
   TP 319,868; FPR 0.62%, miss rate 1.54%.
2. **Feature importance (Table 3)**: water_max_10 0.442, water_mean_5
   0.326, water_slope_5 0.094 (3rd), consensus 0.087 (4th), soil_mean_10
   0.028, rain_mean_10 0.012, rain_sum_20 0.011, health 0.000.
3. **Scenario results (Table 4)**: all eight rows + average row updated
   from the fresh results.json (average row 0.376 / 0.140 / 0.972 / 0.362 /
   2.8). Baseline columns unchanged (F1 0.175-0.196, precision 1.000,
   recall ~10.4%, 8.0 state changes).
4. **Headlines**: F1 advantage 2.6x -> 2.7x (0.38 vs 0.14), recall
   advantage 3.4x -> 3.5x (36.2% vs 10.4%), precision ~97%, mean lead time
   2.8 steps (~40 min, was 4.0/~60), early warnings 53 (8.8 per scenario),
   normal_wet FPR 2.6% / 7.8% worst, dropout_50 state changes 12.0 vs 14.0.
5. **Dropout lead-time narrative rewritten**: degraded-mode guardrails fire
   far fewer early warnings under 50% dropout (2 vs 12 across repeats) and
   mean lead time of those warnings falls 3.3 -> 1.7 steps.
6. **Honest-disclosure additions**: repeat-variability sentence (per-scenario
   F1 stds 0.19-0.23; MAS advantage holds in all 18/18 runs); per-zone-step
   metric-construct sentence at the Sec 5.2 opening; repository URL
   github.com/Buljic/multiagent-flood-detection in the reproducibility
   statement.
7. **Figures**: Fig. 2 / Fig. 3 re-embedded from the fresh artifacts
   (`generate_paper_figures.py` annotation now data-driven, 2.7x); Fig. 3
   narrative verified unchanged (45 min before / 240 min after). PDF
   regenerated (19 pages), zips rebuilt.

## v2.11 — Final Quality Audit: Citation Renumbering & Typography (2026-08-27)

Deep line-by-line verification of the paper against configs/code/artifacts
(CORRECTIONS.md fifth pass). All implementation claims verified correct.
Fixes:

1. **References renumbered to first-citation order** (LNCS convention):
   [29,30]->[4,5] (Matsuda, Watanabe), [31,32]->[15,16] (Rafanelli,
   Avula), [33]->[19] (Tamascelli), sklearn docs moved to [27]/[32]/[33],
   etc.; all in-text markers rewritten and reference paragraphs reordered.
   Verified: first-appearance order strictly 1..33.
2. **Sec 3.1**: "X time steps" -> "T time steps".
3. **Wording**: intro "and that improves" -> "which improves";
   "operational stability, turning noisy signals"; mitigation bullet
   "respond to sustained alerts".
4. **RQ mapping**: "Sections 5.3 and 5.4 address RQ2-RQ3".
5. **Typography**: " - " -> " — " em dashes (9 body spots), "TH_UP -
   TH_DOWN" -> "TH_UP − TH_DOWN", reference title dashes -> en dashes,
   abstract dashes -> em dashes.
6. Positioning-table headers shortened ("Prob. calib.", "Alarm guard.")
   with autofit for clean word-boundary wraps.

No code or measured numbers changed. PDF regenerated (19 pages),
17/17 verification checks pass, zips rebuilt.

## v2.10 — Self-Critical Audit & Positioning Table (2026-08-27)

Fourth-pass audit of the camera-ready paper (docs: `4_cameraReady/CORRECTIONS.md`
fourth pass). The 3rd-version baseline was confirmed broken ("9 zones" x3,
"zero false alarms", "18 early warnings", "Chapter" x5); the 4th version is
now fully consistent:

1. **Citation fix (Sec. 3.5)**: `[21,22]` -> `[22,28]` (was citing
   Friedman's gradient-boosting paper for sklearn calibration docs).
2. **New Table 1** (Related Work positioning table, ADDITIONS_DRAFT C) with
   follow-up novelty sentence; Tables 1-3 renumbered to 2-4 (captions +
   all in-text references + reproducibility statement).
3. **Typography**: "±" restored in the CV row; "2.6x/3.4x/20x20/3x3" ->
   "2.6×/3.4×/20×20/3×3"; em dashes in agent bullets; spaces after "):";
   en dash in Table 4 caption.
4. **Accuracy**: AUC-circularity note now cites Sections 5.2 and 5.4.
5. Positioning table rendered with autofit layout (word-boundary wraps);
   PDF regenerated (19 pages); zips rebuilt.

No code or measured numbers changed in this pass — text/typography/structure
only.

## v2.9 — Full Compute Re-Run & Paper Re-Sync (2026-08-27)

Complete deterministic pipeline re-run (`python run_pipeline.py`, ~59 min):
tests (74/74) -> data gen (2,000 episodes x 400 steps, seed 42) -> training
(seed 42) -> hero sims (3) -> experiments (8 scenarios x 3 repeats, seeds
42/1042/2042) -> figures -> FINAL_RESULTS copy. Environment uses newer
libraries than the original pins (numpy 2.5.2 / pandas 3.0.5 / sklearn
1.9.0 vs 1.26.2 / 2.1.3 / 1.3.2), so results drifted in the 3rd decimal.

### What was verified (all hold)

- Training: AUC 0.999, F1 0.992, precision 0.995, Brier 0.007, CV 0.999;
  test split 592,000 rows / 400 held-out episodes from 2,960,000-row
  dataset (exact).
- Headlines: MAS vs baseline F1 2.62x, recall 3.42x, mean lead 4.0 steps
  (~60 min), 13.3 vs 14.0 state changes, baseline 8.0, 56 early warnings
  (18.7/scenario), normal_wet 2.7% avg FPR (8.0% worst repeat, 0 in two).
- Fig. 3 narrative identical: MAS 45 min before onset, baseline 240 min
  after (extreme_dropout_50, zone 1).

### Paper updated to fresh artifacts (all in CORRECTIONS.md, third pass)

- Table 1: Recall 0.990; CM 265,524/1,590/3,387/321,499; FPR 0.60%;
  miss 1.04%.
- Table 2: FI 0.462/0.333/0.096/0.046/0.041/0.011/0.012/0.000; text
  79.5% combined, consensus 9.6%.
- Table 3: extreme_dry 0.450/0.919/5.2; extreme_wet 0.965/3.4; dropout_10
  0.969/2.7; dropout_50 0.514/0.374; noisy 0.976; averages unchanged.
- Sec 5.2: 56 warnings, 18.7/scenario; Sec 5.3: 2.7%/8.0% FPR, 3.4->7.0
  steps lead under dropout, F1 0.514.
- Fig. 2 (FigA) and Fig. 3 (FigB) regenerated from fresh results/logs and
  re-embedded; `generate_paper_figures.py` now reads F1 lists from
  `results.json` (AGENTS.md hazard #2 — no more hand-copied values).
- Zips rebuilt: `96_Buljic.zip`, `Figures.zip`; PDF regenerated and
  verified (18 pages).

## v2.8 — Camera-Ready Finalization Pass (2026-08-27)

Text-only improvements applied to the camera-ready paper
(`Final Submission Documents/4_cameraReady/96_Buljic/96_Buljic_paper.docx`),
per the plan in `HANDOFF_TO_NEXT_AGENT.md` Section 2. No experiment code or
measured numbers changed; the PDF was regenerated and the submission
archives rebuilt.

### Paper text (docx)

1. **New citations (5)** — trust/alert-fatigue claim in Sec. 1 now cites
   Matsuda, Kotani & Onishi (WCAS 17(4), 2025) and Watanabe et al.
   (J. Meteorol. Soc. Japan 104, 2026); Sec. 2 positions Rafanelli,
   Costantini & De Gasperis (Intelligenza Artificiale 17(1), 2023),
   Avula et al. (IEEE eScience 2025, "Flood Watch"), and Tamascelli,
   Paltrinieri & Cozzani (Comput. Chem. Eng. 143, 2020) as references
   [29]-[33]. Note: the chattering-alarm paper previously identified as
   "Zhou et al. (2020)" is authored by Tamascelli, Paltrinieri & Cozzani
   (verified via Crossref).
2. **Zone layout disclosure (Sec. 4.1)** — the 20x20 grid / four 10x10
   zones / central river corridor / upstream-inflow asymmetry is now
   described; the 3x3 (river vs land zones) scalability study is framed
   as future work (Sec. 6, item 6).
3. **Unverifiable claims softened (Sec. 4.3)** — the TH_UP = 0.6
   "majority-class flood probability" justification is reworded to
   "selected by inspecting the calibrated model's score distribution",
   and the deadband observation is marked as an informal design
   observation.
4. **AUC circularity note (Sec. 5.1)** — part of the near-perfect
   ROC-AUC reflects the causal water-level/label link; operational value
   lies in the pre-threshold (lead-time) regime.
5. **Lead-time definition (Sec. 4.4)** — states precisely how lead time
   is measured (per zone, last warning-state transition before onset;
   floods without a preceding warning contribute none). Metric code
   unchanged.
6. **Reproducibility statement** — added before Acknowledgments
   (ADDITIONS_DRAFT section A, with `[REPOSITORY-URL]` placeholder for
   the authors).
7. **Research questions RQ1-RQ3** — added at the end of Sec. 1
   (ADDITIONS_DRAFT section B; section mapping matches the existing
   paper structure). The positioning table (section C) was NOT added to
   avoid renumbering Tables 1-3.
8. **Abstract wording** — "detects even 2.6x better" -> "achieves 2.6x
   higher detection F1" (overstated "detects" removed).
9. **Reference list normalized** — "Accessed" dates unified to
   "Accessed DD Mon YYYY", internal spaces in URLs removed (refs 2, 18,
   24), page-range dashes normalized (refs 4, 7: en dashes).

### Packaging & verification

- PDF regenerated from the edited docx (LibreOffice headless); verified
  against the handoff checklist: "4 zones" >= 3x, "9 zones" 0x, new
  citations [29]-[33] present in text and reference list, "zero false
  alarms" absent, normal_wet 2.6%/7.9% FPR wording present, "55 early
  warnings ... 18.3 per scenario" present, tables/figures unchanged.
- `96_Buljic.zip` rebuilt from `96_Buljic/` (docx + pdf);
  `Figures.zip` rebuilt as Fig.1.jpg + Fig.2_regenerated.png +
  Fig.3_regenerated.png per the handoff packaging rules.
- `python -m pytest tests/ -q`: 74 passed (requires the pinned
  `mesa==2.1.4`; mesa 3.x breaks the `Agent(unique_id, model)` API).

## v2.7 — Camera-Ready Consistency Fixes (2026-08-27)

Aligns code and paper claims with the archived experiment artifacts.
All evaluations ran with 4 zones (the config default), while the
paper stated 9 zones; the figure title and data-generator logic are
corrected accordingly.

### High

#### 39. Paper and figure stated 9 zones; experiments used 4

**File:** `generate_paper_figures.py`

`configs/default.yaml` sets `num_zones: 4`, and every archived artifact
(`outputs/experiments/results.json`, hero parquet logs, the training
dataset) contains 4 zones. The FigA title now reports the true setup.

#### 40. Training health/consensus features diverged from runtime

**File:** `ml/generate_data.py`

Health is now computed as `max(0, (active - missing_penalty) / total)`
with the same carry-forward penalty schedule as `MissingValueHandler`,
and the per-sensor previous reading is cleared on dropout, matching
`SensorAgent` trend logic. The shipped model and archived results are
unchanged (health has near-zero feature importance).

### Low

#### 41. Figure description files written in platform encoding

**File:** `generate_paper_figures.py`

`write_text` now passes `encoding="utf-8"` so the FigA/FigB description
files are portable across platforms.

## v2.6 — Mutation Safety, Split Guards & Figure Resilience (2026-02-28)

Fixes sensor reading mutation by reference, episode split edge cases,
grouped CV single-class guard, brittle figure generation, and dashboard
lead-time column access.

Test coverage: 71 → 74 tests.

---

### High

#### 38. Edge processing mutated sensor's internal last_reading

**File:** `sim/agents.py`

`sensor.get_reading()` returned `self.last_reading` by reference.
Edge code then mutated `reading['water']` (outlier clipping), which
altered the sensor's internal state used for trend detection on the
next step.

**Fix:** Copy the reading dict before processing:
`reading = dict(reading)`.

### Medium

#### 39. Episode split produced empty test set for small datasets

**File:** `ml/train.py`

`n_test = int(len(episodes) * 0.2)` evaluates to 0 for ≤4 episodes,
causing an empty test set and downstream crashes.

**Fix:** `n_test = max(1, int(len(episodes) * 0.2))`.

#### 40. GroupKFold single-class guard missing in grouped branch

**File:** `ml/train.py`

The ungrouped CV branch had a single-class guard (v2.5 #37), but the
grouped branch did not. `cross_val_score(..., scoring='roc_auc')` with
`GroupKFold` and single-class `y` produces NaN scores and warnings.

**Fix:** Added `elif len(np.unique(y)) < 2` guard in the grouped branch.

### Low

#### 41. Figure generation crashed when expected scenarios were missing

**File:** `eval/make_figures.py`

Fig4 (dropout robustness) and Fig5 (scenario comparison) hardcoded
expected scenario names. If any were missing from results, `KeyError`
or empty plots resulted.

**Fix:** Track only present scenarios; skip figure with warning if
none found.

#### 42. Dashboard lead-time assumed zone_id column exists

**File:** `dashboard/app.py`

`render_lead_time_distribution()` accessed `logs['zone_id']` without
checking the column exists, crashing on logs without zone information.

**Fix:** Added `if 'zone_id' not in logs.columns: return` guard.

---

## v2.5 — Edge-Case Guards (2026-02-28)

Hardens remaining edge cases in metrics, environment validation, and CV.

Test coverage: 68 → 71 tests.

---

### Low

#### 35. Lead-time crashed on logs without zone_id column

**File:** `eval/metrics.py`

`compute_from_logs()` accessed `logs['zone_id']` without checking the
column exists. Logs without `zone_id` (e.g. manually constructed or
single-zone data) would raise `KeyError`.

**Fix:** Check `'zone_id' in logs.columns`; if absent, treat all rows
as a single zone (fallback `zone_id=0`).

#### 36. num_zones validation missed upper bound

**File:** `sim/environment.py`

`num_zones=900` with `grid_size=20` would pass the perfect-square check
but create `sqrt(900)=30 > 20` zones per row, producing empty zones with
no grid cells (logically broken state).

**Fix:** Added check that `sqrt(num_zones) <= grid_size`.

#### 37. cross_val_score without groups crashed on single-class data

**File:** `ml/train.py`

The `else` branch (no groups) called `cross_val_score(..., scoring='roc_auc')`
without checking if both classes exist. Single-class `y` causes sklearn to
return NaN scores and emit warnings.

**Fix:** Guard with `len(np.unique(y)) < 2` check; skip CV and return
NaN scores. Also cap `cv` to `len(y)` to prevent folds > samples.

---

## v2.4 — Dashboard Robustness, Zone Coverage & Consistency (2026-02-27)

This release fixes dashboard crashes on non-MAS log files, ensures complete
grid coverage for zone allocation, aligns alarm definitions across metrics,
suppresses sklearn warnings, and fixes documentation errors.

Test coverage: 65 → 68 tests.

---

### Medium

#### 29. Dashboard crashed when loading coordinator or baseline parquet files

**File:** `dashboard/app.py`

Sidebar offered all `*.parquet` files including `*_coordinator.parquet` and
`*_baseline.parquet`, but timeline expected `zone_id` and `risk` columns
that these files lack, causing `KeyError` crashes.

**Fix:** Filter sidebar to exclude coordinator/baseline files. Added
defensive column checks before accessing `zone_id` and `risk`.

#### 30. Zone allocation left grid cells unassigned for non-divisible sizes

**File:** `sim/environment.py`

With `grid_size=20` and `num_zones=9`, `zone_rows=20//3=6` covered only
18 of 20 rows. Rows 18-19 were never assigned to any zone, creating
unmonitored "dead zones".

**Fix:** Last zone in each row/column now extends to the grid boundary,
absorbing the remainder cells.

### Low

#### 31. Inconsistent alarm definition between detection and lead-time

**File:** `eval/metrics.py`, `dashboard/app.py`

Detection metrics counted both `ALERT` and `SUSPECTED` as positive, but
lead-time only counted `ALERT` transitions. A `SUSPECTED→NORMAL` sequence
counted as detection-positive but generated zero lead-time.

**Fix:** Lead-time now counts `ALERT` or `SUSPECTED` as alert start,
matching the detection definition. Same fix applied in dashboard.

#### 32. UndefinedMetricWarning during single-class evaluation

**File:** `ml/train.py`

`f1_score`, `precision_score`, `recall_score` called without
`zero_division=0`, producing noisy warnings on imbalanced test sets.

**Fix:** Added `zero_division=0` to all three calls.

#### 33. Wrong command example in dashboard

**File:** `dashboard/app.py`

Example used `--config configs/scenarios.yaml` but `--config` expects
`default.yaml`. The correct flag for scenarios is `--scenarios-config`.

**Fix:** Replaced with correct single-line command using both flags.

#### 34. make_paper_figures.py legend label still said "ground_truth"

**File:** `make_paper_figures.py`

Column reference was fixed in v2.3, but the plot legend label and TXT
description still used the old name.

**Fix:** Updated label and description text to `ground_truth_flooded`.

---

## v2.3 — Lead-Time Fix, Input Validation & Edge Cases (2026-02-27)

This release fixes cross-zone lead-time calculation, adds input validation
for zone configuration, hardens ML evaluation against single-class test sets,
fixes seed reproducibility in experiments, and updates stale column references.

Test coverage: 61 → 65 tests.

---

### High

#### 23. Lead-time mixed alerts and floods from different zones

**File:** `eval/metrics.py`

`compute_from_logs()` collected `alert_starts` and `flood_starts` globally
across all zones, then passed the mixed lists to `compute_lead_time`. An alert
in zone 0 could be matched with a flood in zone 1, producing misleading
lead-time values.

**Fix:** Lead time is now computed per zone then aggregated — same pattern as
the stability fix in v2.2. Each zone's alerts and floods are matched
independently, and the resulting lead times are combined.

### Medium

#### 24. num_zones silently produced wrong zone count for non-perfect-squares

**File:** `sim/environment.py`

`_init_zones()` used `int(sqrt(num_zones))` to compute the grid layout. For
non-square values like 6, `int(sqrt(6))=2`, creating a 2×2=4 zone grid
silently instead of the requested 6 zones.

**Fix:** Added validation at the start of `_init_zones()` that raises
`ValueError` if `num_zones` is not a perfect square.

#### 25. roc_auc_score crashed for single-class test set

**File:** `ml/train.py`

`evaluate()` called `roc_auc_score(y_test, y_prob)` without checking if
`y_test` contained both classes. With an all-negative or all-positive test
set, sklearn raises `ValueError: Only one class present in y_true`.

**Fix:** Guard the call: if `len(np.unique(y_test)) < 2`, return `NaN` for
AUC-ROC with a warning instead of crashing.

### Low

#### 26. Experiment seed was hardcoded, metadata reported wrong seed

**File:** `eval/run_experiments.py`

The repeat loop used `seed = 42 + rep * 1000` (hardcoded) while metadata
reported `self.config.get('seed', 42)`. If the config seed was changed,
metadata would lie about which seed was actually used. Also removed unused
`--baseline` CLI argument.

**Fix:** Base seed now comes from `self.config.get('seed', 42)`. Removed
dead `--baseline` argument.

#### 27. make_paper_figures.py used stale column name `ground_truth`

**File:** `make_paper_figures.py`

Referenced `ground_truth` column but the system produces
`ground_truth_flooded`. This caused the script to silently skip ground-truth
overlays in timeline figures and produce wrong lead-time summaries.

**Fix:** Updated all 4 references to `ground_truth_flooded`.

#### 28. Unprotected accuracy division by zero in metrics

**File:** `eval/metrics.py`

`accuracy = (tp + tn) / (tp + tn + fp + fn)` had no zero-division guard.
Other metrics (precision, recall) were protected but accuracy was not.

**Fix:** Added `if total > 0 else 0.0` guard, matching the pattern used for
other metrics.

---

## v2.2 — Metrics Correctness & Package Quality (2026-02-27)

This release fixes stability metrics, hardens edge cases in the ML training
pipeline, and eliminates runtime warnings from Python package imports.

---

### Medium

#### 18. Stability metric counted zone-boundary transitions as state changes

**File:** `eval/metrics.py`

`compute_from_logs()` flattened `logs['state']` into a single list across all
zones. When zone 0's last state was "ALERT" and zone 1's first state was
"NORMAL", the boundary counted as a spurious state change. This inflated
`total_state_changes` (e.g., metrics reported 22 but the state machines
counted 20).

**Fix:** Stability is now computed per zone then aggregated. Each zone's
`state_history` is extracted via `logs[logs['zone_id'] == zone_id].sort_values('step')`,
and `compute_stability_metrics()` is called per zone. Totals are summed,
rates are weighted-averaged.

#### 19. GroupKFold crashed with fewer than 2 episodes

**File:** `ml/train.py`

`GroupKFold(n_splits=1)` is invalid in sklearn. With a single-episode dataset,
`actual_cv = min(cv, n_groups)` would produce `n_splits=1`, crashing.

**Fix:** When `n_groups < 2`, CV is skipped with a warning and `cv_auc_mean=NaN`
is returned. This only affects edge-case testing, not the default pipeline
(200+ episodes).

---

### Low

#### 20. `python -m` runtime warnings from package `__init__.py`

**Files:** `eval/__init__.py`, `sim/__init__.py`, `ml/__init__.py`

All three packages eagerly imported modules that are also run with `python -m`
(e.g., `eval/__init__.py` imported `run_experiments`). This caused Python's
"found in sys.modules after import of package" RuntimeWarning.

**Fix:** Removed eager imports of `-m` runnable modules. Only stable
dependencies (e.g., `MetricsCalculator`, `FloodEnvironment`) remain in
`__init__.py`. No code relied on the removed convenience imports.

#### 21. Variable shadowing in `generate_data.py`

**File:** `ml/generate_data.py`

Inner loop variable `step` shadowed the outer `steps` parameter in
`_run_episode()`. Not a runtime bug but confusing for readers.

**Fix:** Renamed inner loop variable from `step` to `t`.

---

### Test Coverage: 58 → 61 tests

| Category | Before | After | Key additions |
|----------|--------|-------|---------------|
| Experiment Runner | 4 | 7 | stability matches state machine, GroupKFold single-episode, no import warnings |

---

## v2.1 — Experiment & Evaluation Correctness (2026-02-26)

This release fixes issues in the experiment runner, metrics, figure generation,
and ML training pipeline identified during external review. Results from v2.0
pipeline runs should be regenerated.

---

### Critical

#### 13. Baseline ground truth was end-of-simulation constant

**File:** `eval/run_experiments.py`

`is_flooded(zone_id)` was called **after** the simulation loop, so every baseline
log row received the same ground truth (final state). This made all baseline
F1/recall/precision comparisons invalid (baseline showed F1=0.0 artificially).

**Fix:** Ground truth is now recorded **inside** the simulation loop at each step,
using a `per_step_ground_truth` dict keyed by `(step, zone_id)`.

---

### High

#### 14. Single-class metric branch produced invalid stats

**File:** `eval/metrics.py`

When `y_true` had only one class, hardcoded values produced impossible combinations
(e.g., precision=1.0, recall=1.0, but F1=0.0; confusion matrix all zeros).

**Fix:** Removed the special-case branch entirely. `confusion_matrix(labels=[0,1])`
always returns a 2x2 matrix, and the existing division guards (`if (tp+fp) > 0`)
handle single-class inputs correctly.

#### 15. Fig1 timeline was always skipped (filename mismatch)

**Files:** `eval/make_figures.py`, `sim/model.py`

Figure code expected `{scenario}_mas.parquet` and `{scenario}_baseline.parquet`,
but the pipeline produced `hero_{scenario}.parquet` only (no baseline logs).

**Fix:**
- `sim/model.py` main() now runs baseline in parallel with MAS and saves
  `hero_{scenario}_baseline.parquet` alongside MAS logs
- `make_figures.py` updated to look for `hero_{scenario}.parquet` (MAS) and
  `hero_{scenario}_baseline.parquet` (baseline)

---

### Medium

#### 16. Cross-validation AUC was optimistic (non-grouped CV)

**File:** `ml/train.py`

`cross_val_score` used plain KFold on concatenated rows, allowing rows from the
same episode to appear in both train and validation folds (temporal leakage).

**Fix:** CV now uses `GroupKFold` grouped by `episode_id`, ensuring all rows
from an episode stay in the same fold.

#### 17. Config feature list was unused in training

**File:** `ml/train.py`

`FEATURE_COLUMNS` was hardcoded and used directly. The config-based feature list
(`config['ml']['features']`) was ignored, risking silent desync between training
and inference.

**Fix:** `ModelTrainer` now accepts a `config` parameter and reads feature columns
via `get_feature_columns(config)`. All internal uses of `FEATURE_COLUMNS` replaced
with `self.feature_columns`. Pipeline passes `--config` to train step.

---

### Test Coverage: 54 → 58 tests

| Category | Before | After | Key additions |
|----------|--------|-------|---------------|
| Experiment Runner | 0 | 4 | baseline GT varies, single-class consistency, no impossible combos, config features |

---

## v2.0 — Code Quality & Correctness Fixes (2026-02-26)

This release fixes several critical bugs that affected training data integrity
and evaluation metrics. **All previous results (in `outputs/_deprecated_v1/` and
`FINAL_RESULTS/`) were generated with buggy code and should be considered
invalid.** Re-generation instructions are provided below.

---

### Critical Bug Fixes

#### 1. `ground_truth_flooded` was always the FINAL state (not per-step)

**Files:** `sim/model.py`

`get_logs()` called `environment.is_flooded()` **after** the simulation ended,
so every log entry for a zone received the **same** ground truth value (the
flood status at the last step). This means all detection metrics (precision,
recall, F1, lead time) were computed against incorrect labels.

**Fix:** Ground truth is now recorded **inside `step()`** at each simulation
step, so every log entry captures the actual flood state at that moment.

#### 2. Shallow copy mutated the original config across episodes

**Files:** `ml/generate_data.py`, `eval/run_experiments.py`

`config.copy()` is a shallow copy — nested dicts like `config['sensors']` were
shared references. Each call to `_run_episode()` mutated the original config's
`dropout_rate` and `noise_std`, causing parameter bleed between episodes.

**Fix:** Replaced with `copy.deepcopy(self.config)`.

#### 3. Consensus feature in training data was synthetic, not realistic

**File:** `ml/generate_data.py`

Training data used a `tanh()` heuristic to generate consensus values:
```python
consensus = 0.5 + 0.5 * np.tanh(water_slope * 10) + noise
```
But at runtime, consensus is computed from actual sensor readings:
```python
consensus = rising_count / active_count
```
This distribution mismatch meant the ML model was trained on data that didn't
match what it sees during evaluation.

**Fix:** `_run_episode()` now simulates actual sensor noise and dropout per step,
computing consensus and health identically to the runtime `EdgeAggregatorAgent`.

---

### Functional Fixes

#### 4. MissingValueHandler penalty was computed but never used

**File:** `sim/agents.py`

The carry-forward health penalty was calculated but discarded. Health was only
computed as `active_count / total_sensors`.

**Fix:** Penalties are now accumulated and subtracted from base health:
```python
self.health = max(0.0, base_health - avg_penalty)
```

#### 5. MitigationAgent used nonexistent `self.model.schedule.agents`

**File:** `sim/agents.py`

`_get_zone_edge()` iterated over `self.model.schedule.agents`, but `FloodModel`
doesn't use a Mesa scheduler. This would crash if countermeasures were enabled.

**Fix:** Replaced with `self.model.edges.get(self.zone_id)`.

#### 6. Gate activation hardcoded to zone 0 only

**File:** `sim/agents.py`

`if edge.zone_id == 0: self._activate_gate()` — only zone 0 could activate
gates, regardless of river proximity.

**Fix:** Changed to `if zone.is_river_zone:` which checks the actual zone
property.

---

### Code Quality Improvements

#### 7. Centralized feature columns from config

**Files:** `sim/agents.py`, `ml/train.py`

Feature list was hardcoded in 3 places. Now reads from `config['ml']['features']`
with a fallback to the default list.

#### 8. Input validation in guardrails state machine

**File:** `sim/guardrails.py`

`risk`, `consensus`, `health` are now clipped to [0, 1] at the start of
`update()` to prevent undefined behaviour from out-of-range values.

#### 9. `np.random.seed()` replaced with `default_rng()`

**File:** `ml/train.py`

Episode-based data split used the global `np.random.seed()` which is
deprecated and affects global state. Now uses `np.random.default_rng()`.

#### 10. Coordinator logs store structured data

**File:** `sim/agents.py`

Coordinator logged `str(list)` and `str(dict)` which loses structure and makes
analysis difficult. Now stores native Python dicts and lists.

#### 11. `sys.path.insert` with guard

**Files:** `ml/generate_data.py`, `eval/run_experiments.py`, `tests/test_system.py`

Added `if _project_root not in sys.path:` guard to avoid duplicate path entries.

---

### New Features

#### 12. Real-world time mapping

**Files:** `configs/default.yaml`, `sim/model.py`

Added `step_duration_minutes: 15` to config with detailed documentation mapping
simulation steps to real-world time:

| Parameter | Steps | Real-world time |
|-----------|-------|----------------|
| Sensor interval | 1 | 15 min |
| Debounce K_UP=3 | 3 | 45 min |
| Debounce K_DOWN=5 | 5 | 1h 15min |
| Prediction horizon T=10 | 10 | 2h 30min |
| Normal rainfall | ~20 | ~5 hours |
| Extreme rainfall | ~40 | ~10 hours |
| Full episode | 400 | ~4.2 days |

Added `steps_to_real_time(steps, config)` utility function.

---

### Test Coverage: 14 → 54 tests

| Category | Before | After | Key additions |
|----------|--------|-------|---------------|
| Environment | 3 | 7 | water non-negative, soil bounds, no-rain-no-flood, reset |
| Guardrails | 4 | 14 | full cycle, consensus gating, degraded mode, hysteresis, validation, clipper, missing handler |
| Agents | 2 | 9 | risk/health bounds, all zones logged, coordinator, reset, dropout, sensors |
| Baseline | 2 | 4 | returns to normal, zoned independence |
| Metrics | 2 | 7 | perfect prediction, single class, lead time, compare systems |
| Time Mapping | 0 | 4 | minutes, hours, days, default fallback |
| Data Gen | 0 | 4 | deepcopy isolation, features, consensus range, health vs dropout |
| Integration | 1 | 5 | determinism, seed divergence, ground truth per-step, mitigation edge |

---

### Deprecated Outputs

Previous outputs have been moved to `outputs/_deprecated_v1/`. They were
generated with buggy code (incorrect ground truth, config mutation, synthetic
consensus) and should NOT be used for publication.

The `FINAL_RESULTS/` directory also contains results from the buggy version.
These should be regenerated using the instructions below.

---

## How to Regenerate Results

After the code fixes, you must regenerate the full pipeline. Run these commands
**from the project root directory**:

```bash
# ──────────────────────────────────────────────────────────────
# Step 1: Generate new training data (~10 min for 2000 episodes)
# ──────────────────────────────────────────────────────────────
python -m ml.generate_data \
    --config configs/default.yaml \
    --episodes 2000 \
    --steps 400 \
    --out outputs/datasets/sim.parquet \
    --seed 42

# ──────────────────────────────────────────────────────────────
# Step 2: Train the ML model (~2 min)
# ──────────────────────────────────────────────────────────────
python -m ml.train \
    --data outputs/datasets/sim.parquet \
    --model rf \
    --out outputs/models/risk_model.pkl \
    --report outputs/models/train_report.json \
    --seed 42

# ──────────────────────────────────────────────────────────────
# Step 3: Run hero simulations for figures (~1 min each)
# ──────────────────────────────────────────────────────────────
python -m sim.model \
    --config configs/default.yaml \
    --model outputs/models/risk_model.pkl \
    --scenario normal_wet \
    --log outputs/logs/hero_normal_wet.parquet

python -m sim.model \
    --config configs/default.yaml \
    --model outputs/models/risk_model.pkl \
    --scenario extreme_wet \
    --log outputs/logs/hero_extreme_wet.parquet

python -m sim.model \
    --config configs/default.yaml \
    --model outputs/models/risk_model.pkl \
    --scenario extreme_dropout_50 \
    --log outputs/logs/hero_extreme_dropout_50.parquet

# ──────────────────────────────────────────────────────────────
# Step 4: Run full experiments across all scenarios (~30 min)
# ──────────────────────────────────────────────────────────────
python -m eval.run_experiments \
    --config configs/default.yaml \
    --scenarios-config configs/scenarios.yaml \
    --model outputs/models/risk_model.pkl \
    --out outputs/experiments/results.json \
    --steps 400 \
    --repeats 3

# ──────────────────────────────────────────────────────────────
# Step 5: Generate publication figures
# ──────────────────────────────────────────────────────────────
python -m eval.make_figures \
    --results outputs/experiments/results.json \
    --output outputs/figures

# ──────────────────────────────────────────────────────────────
# Step 6: Run tests to verify everything works
# ──────────────────────────────────────────────────────────────
python -m pytest tests/ -v
```

### Expected Changes in Results

After regeneration, expect these differences compared to v1:

- **F1, Precision, Recall** — likely to change because ground truth is now
  correct per-step (was previously constant per zone)
- **Lead time** — will now be meaningful (previously was unreliable)
- **Stability metrics** — should remain similar (unaffected by ground truth)
- **Consensus feature importance** — may shift because consensus is now
  realistic instead of synthetic

The **architectural conclusions** of the paper (MAS outperforms baseline,
guardrails improve stability) should hold or strengthen, since the fixes
make the evaluation more honest.

### Updating FINAL_RESULTS

After regeneration, copy the new outputs to `FINAL_RESULTS/`:

```bash
# Copy new model
cp outputs/models/risk_model.pkl FINAL_RESULTS/model/final_model.pkl
cp outputs/models/train_report.json FINAL_RESULTS/model/final_report.json

# Copy new training data
cp outputs/datasets/sim.parquet FINAL_RESULTS/training/train_data.parquet

# Copy new simulations
cp outputs/logs/hero_*.parquet FINAL_RESULTS/simulations/

# Copy new experiment results
cp outputs/experiments/results.json FINAL_RESULTS/experiments/
```
