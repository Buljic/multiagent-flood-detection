#!/usr/bin/env python
"""
Full pipeline runner for MultiAgent Flood Detection System.

Executes all steps in order:
  1. Run tests          — verify code correctness
  2. Generate data      — synthetic training episodes
  3. Train ML model     — RandomForest + isotonic calibration
  4. Run hero sims      — individual scenario simulations for figures
  5. Run experiments    — full scenario comparison (MAS vs baseline)
  6. Generate figures   — publication-quality PNG + TXT explanations
  7. Copy to FINAL_RESULTS

Usage:
    python run_pipeline.py                  # run everything
    python run_pipeline.py --skip-tests     # skip pytest step
    python run_pipeline.py --quick          # fast mode (200 episodes, 1 repeat)
    python run_pipeline.py --step 3         # run only step 3 (train) and onwards
"""

import argparse
import subprocess
import sys
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
FINAL = ROOT / "FINAL_RESULTS"

# Step durations are printed but not enforced — just for user info
HERO_SCENARIOS = ["normal_wet", "extreme_wet", "extreme_dropout_50"]


def run_cmd(description: str, cmd: list, cwd=None):
    """Run a command, print status, and abort on failure."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  > {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=cwd or str(ROOT))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        print(f"  Aborting pipeline.")
        sys.exit(result.returncode)

    print(f"\n  OK ({elapsed:.1f}s)")
    return result


def step1_tests(args):
    """Run pytest to verify code correctness."""
    if args.skip_tests:
        print("\n  [SKIPPED] Tests (--skip-tests flag)")
        return
    run_cmd(
        "Step 1/7: Running tests",
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    )


def step2_generate_data(args):
    """Generate synthetic training data."""
    episodes = 200 if args.quick else 2000
    steps = 400
    out = str(OUTPUTS / "datasets" / "sim.parquet")

    run_cmd(
        f"Step 2/7: Generating training data ({episodes} episodes x {steps} steps)",
        [sys.executable, "-m", "ml.generate_data",
         "--config", "configs/default.yaml",
         "--episodes", str(episodes),
         "--steps", str(steps),
         "--out", out,
         "--seed", "42"]
    )


def step3_train(args):
    """Train ML model."""
    run_cmd(
        "Step 3/7: Training ML model (RandomForest + isotonic calibration)",
        [sys.executable, "-m", "ml.train",
         "--data", str(OUTPUTS / "datasets" / "sim.parquet"),
         "--config", "configs/default.yaml",
         "--model", "rf",
         "--out", str(OUTPUTS / "models" / "risk_model.pkl"),
         "--report", str(OUTPUTS / "models" / "train_report.json"),
         "--seed", "42"]
    )


def step4_hero_sims(args):
    """Run hero scenario simulations for figures."""
    for scenario in HERO_SCENARIOS:
        run_cmd(
            f"Step 4/7: Hero simulation — {scenario}",
            [sys.executable, "-m", "sim.model",
             "--config", "configs/default.yaml",
             "--model", str(OUTPUTS / "models" / "risk_model.pkl"),
             "--scenario", scenario,
             "--log", str(OUTPUTS / "logs" / f"hero_{scenario}.parquet")]
        )


def step5_experiments(args):
    """Run full experiment comparison across all scenarios."""
    repeats = 1 if args.quick else 3
    run_cmd(
        f"Step 5/7: Running experiments (all scenarios x {repeats} repeats)",
        [sys.executable, "-m", "eval.run_experiments",
         "--config", "configs/default.yaml",
         "--scenarios-config", "configs/scenarios.yaml",
         "--model", str(OUTPUTS / "models" / "risk_model.pkl"),
         "--out", str(OUTPUTS / "experiments" / "results.json"),
         "--steps", "400",
         "--repeats", str(repeats)]
    )


def step6_figures(args):
    """Generate publication figures."""
    run_cmd(
        "Step 6/7: Generating figures",
        [sys.executable, "-m", "eval.make_figures",
         "--results", str(OUTPUTS / "experiments" / "results.json"),
         "--logs-dir", str(OUTPUTS / "logs"),
         "--output", str(OUTPUTS / "figures")]
    )


def step7_copy_final(args):
    """Copy outputs to FINAL_RESULTS."""
    print(f"\n{'='*60}")
    print(f"  Step 7/7: Copying to FINAL_RESULTS/")
    print(f"{'='*60}\n")

    # Create directories
    (FINAL / "model").mkdir(parents=True, exist_ok=True)
    (FINAL / "training").mkdir(parents=True, exist_ok=True)
    (FINAL / "simulations").mkdir(parents=True, exist_ok=True)
    (FINAL / "experiments").mkdir(parents=True, exist_ok=True)

    copies = [
        (OUTPUTS / "models" / "risk_model.pkl",       FINAL / "model" / "final_model.pkl"),
        (OUTPUTS / "models" / "train_report.json",     FINAL / "model" / "final_report.json"),
        (OUTPUTS / "datasets" / "sim.parquet",         FINAL / "training" / "train_data.parquet"),
        (OUTPUTS / "experiments" / "results.json",     FINAL / "experiments" / "results.json"),
    ]

    # Hero simulation logs (MAS, coordinator, and baseline)
    for scenario in HERO_SCENARIOS:
        for suffix in ["", "_coordinator", "_baseline"]:
            src = OUTPUTS / "logs" / f"hero_{scenario}{suffix}.parquet"
            dst = FINAL / "simulations" / f"hero_{scenario}{suffix}.parquet"
            if src.exists():
                copies.append((src, dst))

    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  {src.name} -> {dst.relative_to(ROOT)}")
        else:
            print(f"  [WARN] {src} not found, skipping")

    print(f"\n  OK")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full FloodMAS pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  Run tests
  2  Generate training data
  3  Train ML model
  4  Run hero simulations
  5  Run full experiments
  6  Generate figures
  7  Copy to FINAL_RESULTS

Examples:
  python run_pipeline.py                # full run (~45 min)
  python run_pipeline.py --quick        # fast run (~5 min, fewer episodes/repeats)
  python run_pipeline.py --step 4       # start from step 4 (skip data gen & training)
  python run_pipeline.py --skip-tests   # skip pytest
        """
    )
    parser.add_argument("--quick", action="store_true",
                        help="Fast mode: 200 episodes, 1 repeat (for testing)")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip the pytest verification step")
    parser.add_argument("--step", type=int, default=1, choices=range(1, 8),
                        help="Start from this step (1-7)")

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"#  FloodMAS Pipeline Runner")
    print(f"#  Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"#  Starting from step: {args.step}")
    print(f"{'#'*60}")

    steps = [
        (1, step1_tests),
        (2, step2_generate_data),
        (3, step3_train),
        (4, step4_hero_sims),
        (5, step5_experiments),
        (6, step6_figures),
        (7, step7_copy_final),
    ]

    total_start = time.time()

    for step_num, step_fn in steps:
        if step_num >= args.step:
            step_fn(args)

    total_elapsed = time.time() - total_start

    print(f"\n{'#'*60}")
    print(f"#  Pipeline complete! ({total_elapsed:.0f}s total)")
    print(f"#")
    print(f"#  Results in: outputs/")
    print(f"#  Final copy: FINAL_RESULTS/")
    print(f"#")
    print(f"#  Key files:")
    print(f"#    Model:       outputs/models/risk_model.pkl")
    print(f"#    Report:      outputs/models/train_report.json")
    print(f"#    Experiments: outputs/experiments/results.json")
    print(f"#    Figures:     outputs/figures/")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
