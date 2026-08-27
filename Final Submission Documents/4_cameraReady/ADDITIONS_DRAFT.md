# Proposed Additions — to make the paper more cite-able (DRAFT for review)

Everything below is ready to paste into the corrected `96_Buljic_paper.docx`
in `4_cameraReady`. Nothing has been inserted into the paper yet — review
first, then decide. Watch the page limit: the current camera-ready is 17
pages, so additions B and A are the cheapest; C and D each add ~0.5-1 page
and C would renumber all existing tables (Table 1/2/3 would shift), which
requires fixing the cross-references in Sec. 5.

---

## A. Data / code availability & reproducibility statement (short, very cite-able)

**Where:** end of Section 6 (Conclusion and Future Work), before
"**Acknowledgments.**".

**Text:**

*Reproducibility.* The simulator, data-generation pipeline, training
pipeline, evaluation harness, and figure-generation scripts are implemented
in Python 3.12 and are available in the public repository at
[REPOSITORY-URL]. All experimental configurations (seeds, guardrail
parameters, baseline thresholds, and the eight scenario definitions) are
stored as versioned YAML files, and the trained model, the generated
datasets, the per-run experiment logs, and the results used in Tables 1-3
are archived alongside the source code. The complete evaluation can be
re-executed end-to-end with a single pipeline entry point, ensuring that
every number reported in this paper can be independently regenerated.

(Note: fill in [REPOSITORY-URL] — e.g., a GitHub/Zenodo link. A Zenodo DOI
would also be appropriate.)

---

## B. Explicit research questions (1 short paragraph, high value)

**Where:** end of Section 1 (Introduction), right after the contributions
bullet list / before the domain-transfer paragraph.

**Text:**

To make the evaluation interpretable, the study is structured around three
research questions. (RQ1) Does calibrated ML combined with explicit
guardrails improve detection quality over a minimal threshold baseline under
noisy, incomplete sensor observations? (RQ2) Do the stability mechanisms
(hysteresis, debouncing, consensus gating, and health-aware degradation)
preserve alarm stability without sacrificing early-warning lead time?
(RQ3) How does the system degrade under extreme operating conditions
(sensor dropout, high noise, and dry versus wet initial states)? Section 5.1
addresses RQ1 with offline model-quality results, Section 5.2 addresses
RQ1-RQ3 with scenario-level operational results, and Section 5.3 addresses
RQ2-RQ3 with stability and lead-time analysis.

(Note: Section numbers are already consistent with this mapping — no other
changes needed.)

---

## C. Positioning table in Related Work (cite-able, ~0.5 page)

**Where:** end of Section 2 (Related Work), as a new table after the final
paragraph ("All of this together, FloodMAS sits at the intersection...").

**Draft table (convert to the paper's table style):**

**Table 1. Positioning of FloodMAS relative to representative related work.**

| System / line of work            | Data-driven ML | Multi-agent | Probability calibration | Alarm guardrails | Health-aware degradation | Evaluation data |
|----------------------------------|----------------|-------------|-------------------------|------------------|--------------------------|-----------------|
| Physics-based flood modeling [7] | no             | no          | no                      | no               | no                       | real            |
| ML flood prediction reviews [8]  | yes            | no          | rarely                  | no               | no                       | real/synthetic  |
| Data-driven forecasting [9]      | yes            | no          | no                      | no               | no                       | real            |
| WSN + MAS flood detection [4,10] | no             | yes         | no                      | no               | partial (redundancy)     | synthetic/field |
| Human-agent collectives [11]     | partial        | yes         | no                      | no               | no                       | exercises       |
| Agentic AI disaster mgmt. [12]   | yes            | yes         | not reported            | no               | no                       | mixed           |
| **FloodMAS (this paper)**        | **yes**        | **yes**     | **yes**                 | **yes**          | **yes**                  | synthetic (8 scenarios, 3 repeats) |

**Follow-up sentence:**

As shown in Table 1, FloodMAS is distinguished from prior work not by any
single component but by the combination of all four design pillars —
distributed agents, calibrated ML risk estimation, explicit alarm-stability
guardrails, and health-aware degradation — evaluated under controlled
scenario variations. To the best of our knowledge, no prior flood
early-warning study combines probability calibration with alarm-management
guardrails in a multi-agent setting and evaluates the resulting alarm
stability quantitatively.

**WARNING:** inserting this table renumbers the current Table 1/2/3 into
2/3/4. All references in Sec. 5 ("Table 1 summarizes...", "Table 2 shows...",
"Table 3. Scenario-level...") must be renumbered, and the caption of the
current Table 1 mentions it as "Table 1". If you don't want the renumbering,
place this table as "Table 4" at the end of Section 2 instead — unusual
order but zero renumbering risk.

---

## D. Threats to validity / discussion subsection (~0.5-1 page, boosts rigor)

**Where:** new Section 5.6 (or fold into Section 6 before future work).

**Text:**

*5.6 Threats to Validity and Discussion*

*Construct validity.* Flooding is operationalized as the zone-mean water
level exceeding a fixed threshold, and detection quality is scored per
zone-step, so the F1 values in Tables 3 report the fraction of flood
zone-steps covered by the alarm state rather than event-level detection.
The chosen definition is deliberately simple to keep ground truth
reproducible, but event-level metrics (e.g., per-flood-event detection and
false-alarm rates) may be more intuitive for practitioners and are a
direction for future evaluation.

*Internal validity.* All results are obtained on synthetic simulation data,
and the ML model is a single model family (Random Forest with isotonic
calibration). The guardrail thresholds were set from calibration-set
observations and are not exhaustively swept; a systematic sensitivity
analysis over thresholds and debounce windows is left to future work. The
baseline is deliberately minimal, so the comparison demonstrates the value
of the combined design rather than of ML alone.

*External validity.* The simulator abstracts spatially correlated rainfall,
non-stationary river flow, sensor calibration drift, and communication
latency, and no real sensor-network data was available for the short-horizon
label definition used here. Consequently, the reported absolute lead times
(~60 minutes at 15 min/step) are indicative of the mechanism's anticipatory
capability rather than a claim about deployment-level warning times.
Validation on real hydrometeorological sensor deployments is required
before operational use, as discussed in Section 6.

---

## E. Optional future-work addition (one sentence)

**Where:** in Section 6, item (1) or as a new item.

**Text:**

... and a scalability study of the coordinator's fusion logic on larger zone
grids (e.g., 3 x 3 and 4 x 4 layouts) to quantify how global alarm stability
scales with the number of zones and sensors.

(Note: this also anticipates the "should we rerun with 9 zones" question —
a 9-zone run can be published as this follow-up study instead of changing
the accepted camera-ready numbers.)
