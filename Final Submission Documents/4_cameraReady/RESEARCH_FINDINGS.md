# Self-Critical Audit: Zone Geometry, Paper Fallacies, and Related Work

Prepared: 2026-08-27 (while the 9-zone pilot runs in the background).

---

## 1. Are the zones symmetric? NO — and the paper never discloses it

Measured from `sim/environment.py` (grid 20x20, river = column 10 with
width 3, i.e. columns 9-11, elevation depressed by 2.0):

### 4 zones (what the paper actually used)
```
zone 0: rows 0-9   cols 0-9   river_zone=True   (100 cells)
zone 1: rows 0-9   cols 10-19 river_zone=True   (100 cells)
zone 2: rows 10-19 cols 0-9   river_zone=True   (100 cells)
zone 3: rows 10-19 cols 10-19 river_zone=True   (100 cells)
```
Every zone touches the river (both column bands intersect columns 9-11),
so the 4-zone setup is *roughly* homogeneous; only the top row receives
the artificial upstream inflow (`water_level[0, river_mask[0,:]]`), which
biases zones 0 and 1 slightly.

### 9 zones (the proposed rerun)
```
zone 0: 36 cells  rows 0-5   cols 0-5   river=False
zone 1: 36 cells  rows 0-5   cols 6-11  river=True   <- also receives upstream inflow
zone 2: 48 cells  rows 0-5   cols 12-19 river=False
zone 3: 36 cells  rows 6-11  cols 0-5   river=False
zone 4: 36 cells  rows 6-11  cols 6-11  river=True
zone 5: 48 cells  rows 6-11  cols 12-19 river=False
zone 6: 48 cells  rows 12-19 cols 0-5   river=False
zone 7: 48 cells  rows 12-19 cols 6-11  river=True
zone 8: 64 cells  rows 12-19 cols 12-19 river=False
```
Three asymmetries:
1. **River vs land zones**: only the middle column (zones 1, 4, 7) contains
   the river channel (lower elevation + base flow), so 6 of 9 zones can only
   flood via runoff/spreading — most of the "flood" signal lives in 3 zones.
2. **Unequal zone sizes**: 36 vs 48 vs 64 cells (bottom band gets 8 rows,
   right band gets 8 cols because 20/3 = 6 remainder 2). Zone 8 is 1.8x the
   area of zone 0, so "zone mean water level" is averaged over different
   areas.
3. **Upstream inflow**: only zone 1 (top-middle, row 0) receives
   `upstream_inflow` every step, making it flood-prone regardless of rain.

**Is it a fallacy?** It is an *undisclosed heterogeneity*, and it is the
paper's weakest methodological point. The metrics average all zones
equally, the coordinator treats all zones equally, and the text never says
that only a subset of zones contains the river. A careful reviewer can
spot this. Safe fixes (no number changes):
- Add a short paragraph in Sec. 4.1 describing the zone grid explicitly:
  "the 3x3 zone grid overlays a central river corridor, so the middle
  column of zones (river zones) carries base flow and floods first, while
  the lateral zones flood only through runoff and spreading; zone sizes
  differ because the grid does not divide evenly."
- Add one sentence in Sec. 5: results aggregate all zones; river-zone vs
  land-zone behaviour is reported where relevant.
Better fixes (only possible in a rerun):
- Report per-zone-group metrics (river vs land) in Table 3 or a new table.
- (Optional) give the model a `zone_type`/`is_river` feature or train
  per-zone-type models — but that is a bigger change; disclosing + grouped
  metrics is enough for camera-ready.

---

## 2. Fallacy / weakness audit of the paper

### A. Fixable in TEXT only (no rerun, no numbers change)
1. **Uncited trust claims** (Sec. 1): "flapping erodes operator trust,
   creates alert fatigue" has no citation. Add:
   - Matsuda & Kotani, "Causal effects of perceived false alarm ratio on
     flood protective action", Weather, Climate, and Society (2025).
     https://journals.ametsoc.org/view/journals/wcas/17/4/WCAS-D-24-0106.1.xml
   - "Emotional responses and perceptions of false alarms in flood
     warnings in Japan", Discov. Cities (Springer, 2026).
     https://link.springer.com/article/10.1007/s44394-026-00029-0
2. **Unverifiable threshold claim** (Sec. 4.3): "TH_UP = 0.6 corresponds to
   the point where the isotonic-calibrated model assigns majority-class
   flood probability" — not evidenced anywhere; soften to "thresholds were
   selected by inspecting the calibrated model's score distribution on the
   training set" or delete the justification sentence.
3. **Unreported mini-experiment** (Sec. 4.3): "Preliminary observations
   suggest that narrowing the deadband below 0.1 reintroduces flapping..."
   — a result claimed without a table/figure. Either mark it clearly as an
   informal observation ("we observed informally...") or remove.
4. **Reference list sloppiness**: mixed "Accessed Jan 20, 2026" /
   "Accessed 20 Jan 2026" / "Accessed: 3 Feb 2026"; some URLs wrapped in
   `<...>`, others not; ref [2] URL contains an internal space; ref [7]
   "85-101" hyphen style vs en-dash elsewhere. Normalize all.
5. **"Chapter 2..6"** already fixed to "Section". "detects even 2.6x
   better" -> consider "achieves 2.6x higher detection F1" (the word
   "detects" overstates).
6. **No citation for alarm-management motivation in Sec. 2** — already has
   [13,14,24]; could add: Zhou et al., "Predicting chattering alarms: a
   machine learning approach", Computers & Chemical Engineering (2020).
   https://www.sciencedirect.com/science/article/pii/S0098135420308000
   (this one is the closest industrial sibling: ML + chattering alarms).

### B. Fixable in a RERUN (code + numbers would change honestly)
7. **Train/serve noise mismatch**: training features (water/rain/soil)
   were computed from the environment's TRUE zone state, while runtime uses
   noisy sensor averages. In a fresh 9-zone rerun, make `generate_data.py`
   use noisy sensor readings for the water/rain/soil features too — this
   closes the gap and makes the model genuinely noise-robust (better paper).
8. **Lead-time metric can be inflated**: `compute_lead_time` credits the
   LAST alert before flood onset, even if that alert was an earlier
   transient blip. Fix: credit lead time only when the warning state is
   active within a short window before onset (e.g., alert at onset or
   onset-1), or report both variants. Otherwise a reviewer can manufacture
   the counterexample: "alert at step 5, flood at step 300 => 295 steps of
   lead time".
9. **Near-perfect AUC partially circular**: the label "flood within T" is
   nearly deterministic given water-level features once the threshold is
   approached, so ROC-AUC 0.999 may overstate model skill. Add one honest
   sentence: part of the discriminability comes from the strong causal link
   between water-level features and the label; the operational value is the
   pre-threshold regime measured by lead time.
10. **No variability reporting**: 3 repeats give means without std/CI.
    Add +/- std to Table 3 (the JSON already stores std per metric).
11. **Zone heterogeneity** (Section 1 above) — disclose + grouped metrics.
12. **Baseline precision = 1.0 headline**: "BL Precision 1.000" is
    presented as if it were an advantage-neutral fact; the text does
    explain it (rarely alerts), keep the explanation, but make sure the
    caption of Table 3 does not read like the baseline is "more precise".

### C. Already handled in the previous pass
- "9 zones" -> 4 (or true 9 if we rerun), normal_wet "zero false alarms",
  "18 warnings" -> 55, GBM citation [20]->[21], "592,000 from 400
  episodes", health-formula description, FigA title.

---

## 3. Related work sweep — is this a redo of something better?

Closest neighbours found (none does the full combination):

1. **Rafanelli, Costantini, De Gasperis (2023). "Neural-logic multi-agent
   system for flood event detection." Intelligenza Artificiale 17(1),
   19-35. https://doi.org/10.3233/IA-230004**
   MAS (logical agents) + deep learning (image segmentation) for flood
   event detection; alerts fire only when TWO sources agree (aerial image
   segmentation + weather reports). Closest in spirit to FloodMAS's
   consensus gating, BUT: no probabilistic calibration, no
   hysteresis/debounce state machine, no health-aware degradation, and its
   consensus is across heterogeneous sources, not redundant same-zone
   sensors. **Must be cited and positioned** (currently missing from the
   paper).

2. **Avula et al. (2025). "Flood Watch: A Multi-Agent System for Smarter
   Disaster Response." IEEE (e.g., DSAA/CEC).**
   MAS integrating geolocated social media + IoT sensors for flood
   detection/alerting. Focus is data fusion and situational awareness;
   no calibrated ML risk scores, no alarm-stability guardrails, no
   health-aware degradation. **Must be cited and positioned.**

3. **Zhou et al. (2020). "Predicting chattering alarms: a machine learning
   approach." Computers & Chemical Engineering.**
   ML prediction of chattering alarms in industrial plants (the same
   anti-chatter problem, outside the flood domain). Great citation for the
   alarm-management motivation; shows the problem is real and studied, but
   in process industry, not flood EWS.

4. **Matsuda & Kotani (2025). Weather, Climate, and Society** + the Japan
   false-alarm perception study — empirical evidence that false alarms
   erode protective behaviour. Cites the paper's opening claim properly.

5. **Mousa, Zhang, Claudel (2008/2009)** — already cited [4,10]: WSN flash
   flood warning with distributed control. No ML calibration/guardrails.

6. **Surveys**: "Enhancing Flood Risk Management: A Comprehensive Review
   on Flood Early Warning Systems" (Water 16(10), 2024, MDPI); "The role
   of AI for early warning systems: status and..." (iScience 2025) — good
   for the intro/related work positioning sentence.

**Novelty verdict:** nobody in the found literature combines
(a) calibrated ML probabilities, (b) ISA-18.2-style alarm guardrails
(hysteresis + debouncing), (c) multi-agent zone decomposition, and
(d) health-aware degradation — and evaluates *alarm stability + lead time*
as first-class metrics. The closest (Rafanelli 2023) shares the MAS +
agreement idea only. So FloodMAS is NOT a redo, but the paper currently
fails to show that: it must cite 1 and 2 and add one explicit positioning
sentence in Sec. 2 (see ADDITIONS_DRAFT.md section C for the table).
