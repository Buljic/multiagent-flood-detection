# Camera-Ready Corrections — 96_Buljic

Date: 2026-08-27
Files: `96_Buljic_paper.docx` (authoritative), `96_Buljic_paper.pdf` (converted copy)

## Sixth pass (2026-08-27) - noise-consistency re-run, full number re-sync

A noise-consistency fix in `ml/generate_data.py` (training-time health and
consensus features aligned with the runtime edge-agent logic) required a
retrain, so the full deterministic pipeline was re-run (2,000 episodes x
400 steps, seed 42; training seed 42; 8 scenarios x 3 repeats, seeds
42/1042/2042; hero sims). The fresh canonical artifacts are
`FINAL_RESULTS/experiments/results.json` (timestamp 2026-08-27T18:38),
`FINAL_RESULTS/model/final_report.json`, the hero parquets, and
`final_model.pkl`. Every paper number was re-derived from these artifacts
and the docx/PDF/figures were re-synced:

- **Paper Table 2 (model metrics)**: ROC-AUC 0.999 -> 0.998; F1 0.992 ->
  0.990; Precision 0.995 (unchanged); Recall 0.990 -> 0.985; Brier 0.007 ->
  0.009; 5-fold CV 0.999 +/- 0.000 -> 0.998 +/- 0.000; Accuracy 0.992 ->
  0.989. Confusion matrix TN 265,524 / FP 1,590 / FN 3,387 / TP 321,499 ->
  TN 265,460 / FP 1,654 / FN 5,018 / TP 319,868; FPR 0.60% -> 0.62%; miss
  rate 1.04% -> 1.54%. Test-set size 592,000 zone-steps unchanged.
- **Paper Table 3 (feature importance)**: water_max_10 0.462 -> 0.442;
  water_mean_5 0.333 -> 0.326; water_slope_5 0.046 -> 0.094 (now third);
  consensus 0.096 -> 0.087 (now fourth); soil_mean_10 0.041 -> 0.028;
  rain_mean_10 0.011 -> 0.012; rain_sum_20 0.012 -> 0.011; health 0.000
  (unchanged). Text: consensus "third (9.6%)" -> "fourth (8.7%)"; "79.5%
  combined" -> "76.8% combined".
- **Paper Table 4 (scenario results)**: extreme_dry 0.450/0.919/0.323/5.2
  -> 0.461/0.929/0.332/4.8; extreme_wet 0.497/0.965/0.358/3.4 ->
  0.513/0.967/0.370/3.3; dropout_10 0.488/0.350/2.7 -> 0.502/0.362/2.8;
  dropout_30 0.502/0.959/0.366/4.7 -> 0.511/0.981/0.369/2.5; dropout_50
  0.987/0.374/7.0 -> 0.995/0.371/1.7; noisy 0.492/0.976/0.355/1.4 ->
  0.509/0.992/0.366/1.4. Baseline columns unchanged (F1 0.175-0.196,
  precision 1.000, recall 0.096-0.109). Average row 0.368/0.141/0.963/
  0.354/4.0 -> 0.376/0.140/0.972/0.362/2.8.
- **Headline claims**: F1 advantage 2.6x -> 2.7x (0.38 vs 0.14); recall
  advantage 3.4x -> 3.5x (36.2% vs 10.4%); precision ~96% -> ~97%; mean
  lead time 4.0 steps (~60 min) -> 2.8 steps (~40 min); early warnings 56
  -> 53 (8.8 per scenario across its three repeats); normal_wet FPR 2.7% /
  8.0% worst -> 2.6% / 7.8% worst; dropout_50 total state changes 13.3 ->
  12.0 (vs 14.0 for extreme_wet); extreme_wet F1 in the dropout paragraph
  0.497 -> 0.513; lead-time range 1.4-7.0 -> 1.4-4.8 steps (extreme_dry).
- **Dropout lead-time narrative rewritten** (Sec 5.3): in the fresh run the
  degraded-mode guardrails fire far fewer early warnings under 50% dropout
  (2 across the three repeats vs 12 in extreme_wet) and the mean lead time
  of those warnings falls from 3.3 to 1.7 steps; the old "increases from
  3.4 to 7.0" claim no longer matches the data.
- **Fig. 2 / Fig. 3 re-embedded** from the fresh artifacts. Fig. 3
  narrative verified unchanged (FloodMAS 45 min before onset, baseline
  240 min after, extreme_dropout_50 zone 1). `generate_paper_figures.py`
  now computes the 2.7x annotation from the results summary instead of a
  hardcoded value.
- **Three honest-disclosure additions**: repeat-variability sentence after
  the F1-average sentence in the Fig. 2 paragraph (per-scenario F1
  standard deviations 0.19-0.23 across the three repeats; MAS advantage
  holds in every scenario and every repeat); per-zone-step metric-construct
  sentence at the opening of Sec 5.2; repository URL
  github.com/Buljic/multiagent-flood-detection filled into the
  reproducibility statement.

Verification: MAS F1 exceeds baseline F1 in all 6 flood scenarios and all
3 repeats (18/18 runs); per-scenario MAS F1 standard deviations range
0.189-0.226; PDF regenerated (19 pages); zips rebuilt.

---

## Fifth pass (2026-08-27) - final quality audit (deep line-by-line verification)

Deep audit of every claim in the paper against `configs/`, `sim/`, `ml/`,
`eval/`, `baseline/` and the fresh artifacts. All implementation claims
verified correct (guardrail params TH_UP/K_UP/CONS_MIN/HEALTH_MIN + degraded
mode, health/consensus formulas, baseline thresholds + hysteresis + debounce,
river geometry 3-col band / 4 zones / upstream inflow on the top row, 8
scenarios, T=10, 15-min steps, flood threshold 0.7). Fixes applied:

1. **Reference list renumbered to first-citation order** (LNCS convention).
   Previously refs [29]-[33] (Matsuda, Watanabe, Rafanelli, Avula,
   Tamascelli) were appended at the end even though they are first cited in
   Sections 1-2, ahead of the sklearn-docs refs [26]-[28] (Sec 5.1). All
   in-text markers rewritten (e.g. [29,30]->[4,5], [31,32]->[15,16],
   [33]->[19], [4,10]->[6,12], [20]->[25], [21]->[26], [22]->[27],
   [17,18]->[22,23]); reference paragraphs reordered to match. Verified:
   first-appearance order is strictly 1..33.
2. **Sec 3.1**: variable "X time steps" -> "T time steps" (was inconsistent
   with T used everywhere else).
3. **Grammar/wording**: intro "device availability and that improves" ->
   "device availability, which improves"; "operational stability where we
   turn noisy signals" -> "operational stability, turning noisy signals";
   mitigation-agent bullet "react to sustained alerts so that they somewhat
   react to the sensors" -> "respond to sustained alerts".
4. **RQ-to-section mapping** (Sec 1): lead-time analysis lives in Sec 5.4,
   so the mapping now reads "Sections 5.3 and 5.4 address RQ2-RQ3".
5. **Typography**: parenthetical hyphens " - " -> em dashes " — " (9 body
   spots); "TH_UP - TH_DOWN" -> "TH_UP − TH_DOWN" (minus sign); reference
   title dashes -> en dashes (4 spots); abstract dash pair -> em dashes.
6. Positioned-table header cells shortened for clean word-boundary wraps
   ("Prob. calib.", "Alarm guard.", etc.) with autofit layout.

Verification: 17/17 final checks on the regenerated PDF (19 pages) —
tables 1-4, all fresh numbers, citations in order, typography, headline
claims (2.6x, 3.4x, 60-min lead, 45/240-min Fig. 3 narrative).

---

## Fourth pass (2026-08-27) — self-critical audit ("super-fixed" v4)

Full self-critical audit of the 4th version against the 3rd-version flaws,
all instruction docs (HANDOFF / ADDITIONS_DRAFT / RESEARCH_FINDINGS) and
the fresh artifacts. Fixes applied:

1. **Citation bug (Sec. 3.5)**: calibration-curves/Brier sentence cited
   `[21,22]`, where [21] is Friedman's gradient boosting. Now `[22,28]`
   (sklearn CalibratedClassifierCV + brier_score_loss docs).
2. **Positioning table added (Sec. 2, new Table 1)**: the related-work
   positioning table from ADDITIONS_DRAFT section C (7 rows: physics-based
   [7], ML reviews [8], data-driven [9], WSN+MAS [4,10], human-agent
   collectives [11], agentic AI [12], FloodMAS) plus the follow-up
   novelty sentence. All existing tables renumbered 1->2, 2->3, 3->4
   (captions and in-text references); reproducibility statement now says
   "Tables 2-4".
3. **Typography**: restored the missing "±" in Table 2 (was "0.999  0.000"
   -> "0.999 ± 0.000"); normalized ASCII "x" to "×" (2.6x -> 2.6×,
   3.4x -> 3.4×, 20x20 -> 20×20, 3x3/4x4 -> 3×3/4×4); em dashes for the
   agent bullets; spaces after "):" in the guardrail mechanism list;
   en dash in the Table 4 caption.
4. **Accuracy**: AUC-circularity note now points to Sections 5.2 and 5.4
   (where the lead-time results actually live).
5. **Figures/PDF**: positioning table rendered with clean column layout
   (autofit, word-boundary wraps); PDF regenerated (19 pages); zips
   rebuilt. All 12 verification checks pass (numbers, tables, citations
   [1]-[33], "4 zones" x3, no "9 zones"/"zero false alarms"/"Chapter").

Verified against the 3rd version (the broken baseline): 3rd had "9 zones"
x3, "zero false alarms", "18 early warning alerts", "Chapter" x5, no
honest FPR/warning numbers — all fixed in the 4th version.

---

## Third pass (2026-08-27) — full compute re-run, paper re-synced to fresh artifacts

A complete deterministic pipeline re-run was executed on 2026-08-27
(`python run_pipeline.py`: 2,000 episodes x 400 steps, seed 42; training
seed 42; 8 scenarios x 3 repeats, seeds 42/1042/2042; hero sims; figures;
copy to FINAL_RESULTS). Library stack differs from the original pins
(numpy 2.5.2 / pandas 3.0.5 / sklearn 1.9.0 vs pinned 1.26.2 / 2.1.3 /
1.3.2), so results drifted in the 3rd decimal. All paper numbers were
re-derived from the FRESH artifacts and the paper was updated to match:

- **Table 1**: Recall 0.989 -> 0.990; confusion matrix TN 265,552 /
  FP 1,562 / FN 3,433 / TP 321,453 -> TN 265,524 / FP 1,590 / FN 3,387 /
  TP 321,499; FPR 0.58% -> 0.60%; miss rate 1.06% -> 1.04%.
- **Table 2**: water_max_10 0.454 -> 0.462; water_mean_5 0.329 -> 0.333;
  consensus 0.108 -> 0.096; water_slope_5 0.045 -> 0.046; rain_mean_10
  0.012 -> 0.011; rain_sum_20 0.011 -> 0.012. Text: "78.3% combined" ->
  "79.5% combined"; "consensus ... (10.8%)" -> "(9.6%)".
- **Table 3**: extreme_dry 0.449/0.920/5.0 -> 0.450/0.919/5.2;
  extreme_wet 0.967/3.2 -> 0.965/3.4; dropout_10 0.967/2.8 -> 0.969/2.7;
  dropout_50 0.515/0.375 -> 0.514/0.374; noisy 0.978 -> 0.976. Averages
  unchanged (0.368 / 0.141 / 0.963 / 0.354 / 4.0).
- **Sec. 5.2**: 55 early warnings -> 56; 18.3 per scenario -> 18.7.
- **Sec. 5.3**: normal_wet FPR 2.6% -> 2.7% avg, 7.9% -> 8.0% worst repeat
  (0 / 0 / 128 FP zone-steps; was 126); dropout lead-time story 3.2 -> 3.4
  steps; extreme_dropout_50 F1 0.515 -> 0.514 (still > extreme_wet 0.497).
- **Fig. 2 / Fig. 3 re-embedded**: regenerated from fresh results and hero
  logs; Fig. 3 claims verified unchanged (MAS 45 min before onset, baseline
  240 min after, extreme_dropout_50 zone 1).
- `generate_paper_figures.py` now reads the F1 lists from
  `outputs/experiments/results.json` (no hand-copied values; AGENTS.md
  hazard #2).
- Headline claims verified unchanged: 2.6x F1 advantage (fresh 2.62),
  3.4x recall (3.42), mean lead time 4.0 steps (~60 min), 13.3 vs 14.0
  state changes, baseline 8.0 changes, no flooding in normal scenarios.

Unchanged in the re-run: hero Fig.3 narrative, stability metrics, zone
layout, baseline behaviour (BL F1 ~0.19, precision 1.000, recall ~10%),
test-set provenance (592,000 rows / 400 held-out episodes).

`FINAL_RESULTS/experiments/results.json`, `FINAL_RESULTS/model/final_report.json`
and `FINAL_RESULTS/simulations/*.parquet` now hold the fresh (canonical) run.

---

## Second pass (2026-08-27, text only — no numbers changed)

Applied the improvements listed in `HANDOFF_TO_NEXT_AGENT.md` Section 2:

- **5 new citations [29]-[33]**: Matsuda/Kotani/Onishi (WCAS 17(4), 2025)
  and Watanabe et al. (JMSJ 104, 2026) for the intro trust/alert-fatigue
  claim; Rafanelli et al. (IA 17(1), 2023) and Avula et al. (IEEE eScience
  2025) positioned in Sec. 2; Tamascelli et al. (Comput. Chem. Eng. 143,
  2020) for the chattering-alarm motivation. (The "Zhou et al. 2020" paper
  is actually authored by Tamascelli, Paltrinieri & Cozzani — verified via
  Crossref.)
- **Zone layout disclosure (Sec. 4.1)**: 20x20 grid, four 10x10 zones,
  central river corridor, upstream-inflow asymmetry; 3x3 river-vs-land
  study added as future work (Sec. 6, item 6).
- **Sec. 4.3**: TH_UP justification reworded (score-distribution
  inspection), deadband observation marked informal.
- **Sec. 5.1**: AUC circularity caveat added.
- **Sec. 4.4**: precise lead-time definition added (metric code unchanged).
- **Sec. 1**: RQ1-RQ3 paragraph added (ADDITIONS_DRAFT B).
- **Sec. 6**: reproducibility statement added before Acknowledgments
  (ADDITIONS_DRAFT A) with `[REPOSITORY-URL]` placeholder — **authors must
  fill in the repo/Zenodo URL**.
- **References**: normalized (Accessed dates unified, URL internal spaces
  removed, en dashes in page ranges).
- Abstract "detects even 2.6x better" -> "achieves 2.6x higher detection F1".

The positioning table (ADDITIONS_DRAFT C) was intentionally NOT added to
avoid renumbering Tables 1-3. Page count went 17 -> 18; if the venue hard-
limits at 17, the cheapest trims are the RQ paragraph or the reproducibility
statement.

Verification: PDF text re-checked (4 zones x3, no "9 zones", new citations
present, "zero false alarms" absent, 2.6%/7.9% and 55/18.3 wording intact);
`96_Buljic.zip` and `Figures.zip` rebuilt.

---

## First pass corrections

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
