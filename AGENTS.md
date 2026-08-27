# AGENTS.md — Repository Conventions for FloodMAS

## Branch policy (mandatory)

- `main` — intentionally EMPTY (placeholder README only) until the project is finished and published.
- `dev` — the main integration/sync branch. Only clean, finished work lands here.
- `dev-imad` — the working branch. All commits happen here first; merge into `dev` only when a piece of work is verified and stable.

### Handoff rule (important)

`HANDOFF_TO_NEXT_AGENT.md` exists ONLY on `dev-imad`. It is internal working
material ("dirty code"). Before any merge `dev-imad` → `dev`, the handoff
file MUST be deleted (and the deletion committed). It must never exist on
`dev` and never on `main`.

## Commands

- Run tests: `python -m pytest tests/ -q`
- Full pipeline: `python run_pipeline.py` (steps: tests → data gen → train → hero sims → experiments → figures → copy to FINAL_RESULTS)
- Quick pipeline: `python run_pipeline.py --quick`
- Generate paper figures: `python generate_paper_figures.py`

## Determinism

Every stage is seeded (data gen, training, experiments: seeds 42 / 1042 /
2042, scenario configs in `configs/`). System load does NOT change results —
only wall-clock time. Do not edit configs while a run is in progress.

## What is gitignored (regenerable artifacts)

- `outputs/` (datasets, models, logs, experiment results, figures)
- `FINAL_RESULTS/model/*.pkl` and `FINAL_RESULTS/training/*.parquet` (large binaries)
- local agent settings, caches, venvs

`FINAL_RESULTS/experiments/*.json`, `FINAL_RESULTS/model/final_report.json`,
and `FINAL_RESULTS/simulations/*.parquet` ARE tracked (paper-number evidence).

## Layout

- `configs/` — scenario + system configuration (source of truth for all numbers)
- `sim/` — Mesa model (environment, agents, guardrails)
- `ml/` — data generation + training
- `eval/` — experiment runner, metrics, figures
- `baseline/` — threshold baseline
- `dashboard/` — Streamlit UI
- `Final Submission Documents/` — 1..4_cameraReady (4 = working copy with corrections)
- `archive/` — old paper revisions and superseded docs (reference only)

## Known paper-number hazards (for future agents)

- Zone count must match `configs/default.yaml` (`num_zones`) and the actual results.
- `generate_paper_figures.py` hardcodes scenario F1 values in `mas_f1`/`bl_f1`
  lists — after any rerun these MUST be regenerated from `results.json`, not hand-copied.
- Every in-text number (Tables 1-3, lead times, state changes, FPRs, Fig. 2/3
  annotations) must be re-derived from the JSON artifacts before submitting.
