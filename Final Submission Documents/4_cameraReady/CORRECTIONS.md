# Camera-Ready Corrections — 96_Buljic

Date: 2026-08-27
Files: `96_Buljic_paper.docx` (authoritative), `96_Buljic_paper.pdf` (converted copy)

All corrections below were made to the paper text so that it matches the
archived experiment artifacts (`outputs/experiments/results.json`, the training
report `outputs/models/train_report.json`, the training dataset
`outputs/datasets/sim.parquet`, and the hero logs `outputs/logs/hero_*.parquet`).
No measured number in the paper was invented or changed; only statements that
contradicted the recorded data were corrected.

---

## 1. Zone count: "9 zones" -> "4 zones" (3 places + figure)

The paper stated 9 zones in three places. Every artifact shows 4 zones:
`configs/default.yaml` (`num_zones: 4`), the confusion matrices in
`results.json` (1600 = 4 zones x 400 steps per run), the training dataset
(2,000 episodes x 4 zones), and the hero logs (zone IDs 0-3). The "9 zones"
statements were corrected to "4 zones":

- Sec. 5.1: "2,000 simulated episodes (400 steps each, 4 zones)"
- Sec. 5.1 / Fig. 2 paragraph: "averaged over 3 repeated runs with 4 zones and 400 steps each"
- Sec. 5.2: "each with 4 zones over 400 steps"

The embedded Fig. 2 (bar chart) was regenerated from
`generate_paper_figures.py` so its subtitle no longer reads "9 zones".

## 2. Table 1 caption / test-set provenance

- "test set: 592,000 zone-step samples from 2,000 episodes" ->
  "test set: 592,000 zone-step samples from 400 held-out episodes"
- "(total 592,000 zone-step samples from 2,000 simulated episodes)" ->
  "(total 592,000 zone-step samples from 400 held-out episodes)"

Rationale: the dataset has 2,960,000 rows (2,000 episodes x 4 zones x 370
labeled zone-steps); the 20% episode-based test split yields 592,000 rows
drawn from 400 episodes, not 2,000.

## 3. normal_wet "zero false alarms" claim corrected (Sec. 5.3)

The old text claimed both systems "correctly remain silent" and that
FloodMAS "produces zero false alarms" in normal_wet. The archived run data
show FloodMAS produced 126 false-positive zone-steps in one of the three
repeats (0 and 0 in the other two; 7.9% FPR in that repeat, 2.6% average).
The text now reports this honestly while keeping the guardrails argument.

## 4. "18 early warning alerts" claim corrected (Sec. 5.2)

The old text said FloodMAS "issued 18 early warning alerts". The recorded
lead-time counts sum to 55 pre-flood warning events across the six flood
scenarios and three repeats (a mean of 18.3 per scenario over its repeats).
The text now states exactly that.

## 5. State-change figure clarified (Sec. 5.3)

"an average of 13.3 state changes vs. 14.0" now reads
"13.3 total state changes (summed across the four zones) vs. 14.0".

## 6. Citation fix (Sec. 3.4)

The Gradient Boosting sentence cited [20] (Breiman, Random Forests); it now
cites [21] (Friedman, gradient boosting). The Random Forest sentence keeps [20].

## 7. "Chapter N" -> "Section N" (Sec. 1)

The outline paragraph referred to "Chapter 2 ... Chapter 6"; corrected to
"Section 2 ... Section 6" to match the section numbering used in the paper.

## 8. Health indicator description aligned with the implementation (Sec. 4.3)

The paper described health as the "fraction of sensors currently operational";
the runtime edge agent additionally subtracts a penalty for sensors with
missing readings (MissingValueHandler carry-forward schedule). The text now
describes the implemented formula. `ml/generate_data.py` was updated so the
training-time health and consensus features follow the same logic as the
runtime edge agents (the shipped model and results are unchanged; health has
near-zero feature importance).

## 9. Minor polish

- "(vibrations,turbulence and similar)" -> "(vibrations, turbulence, and similar)"
- "not to maximize hydrologic realism but still," -> "not to maximize hydrologic realism; still,"
- Fig. 2 paragraph: "omitted as neither system issues any alarms (correctly)" ->
  "omitted because they contain no flooding (detection F1 is 0.0 for both systems)"

## Verified unchanged (consistent with the archived data)

- Table 1 (ROC-AUC 0.999, F1 0.992, Precision 0.995, Recall 0.989, Brier 0.007,
  CV 0.999 +/- 0.000, Accuracy 0.992, confusion matrix TN 265,552 / FP 1,562 /
  FN 3,433 / TP 321,453)
- Table 2 feature importances (water_max_10 0.454, water_mean_5 0.329,
  consensus 0.108, water_slope_5 0.045, soil_mean_10 0.041, rain_mean_10 0.012,
  rain_sum_20 0.011, health 0.000)
- Table 3 (all 8 scenarios, averages, 2.6x F1 advantage, 3.4x recall, 4.0-step
  mean lead time)
- Fig. 3 claims: FloodMAS alerts 45 min before flood onset; baseline alerts
  240 min after onset (verified against hero_extreme_dropout_50 logs, zone 1)
- Abstract claims (2.6x, ~60 min lead time, ROC-AUC/F1/Brier)

## Regeneration note

The corrected PDF in this folder was converted from the corrected DOCX with
LibreOffice for convenience. The DOCX is the authoritative camera-ready file;
the publisher will re-typeset from it. If the publisher requires a PDF
produced by a specific toolchain, regenerate it from this DOCX.

`Fig.2_regenerated.png` and `Fig.3_regenerated.png` are the re-rendered
versions of the bar-chart and timeline figures produced by the corrected
`generate_paper_figures.py` (the Fig. 2 subtitle now reads "4 zones"). If the
conference also asks for separate figure files (the `Figures.zip` in this
submission contains the older versions), replace `Fig.2.jpg` with
`Fig.2_regenerated.png`; `Fig.3` is unchanged in content and can be kept or
replaced. `Fig.1` (architecture) was not affected.
