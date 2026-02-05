# Full test: Run all scenarios from configs/scenarios.yaml
# Usage: .\scripts\full_test.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FloodMAS Full Scenario Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

# Step 1: Generate training data
Write-Host "[1/4] Generating training data (1000 episodes)..." -ForegroundColor Yellow
python -m ml.generate_data --episodes 1000 --steps 300 --out outputs/datasets/full_train.parquet --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Data generation failed!" -ForegroundColor Red
    exit 1
}

# Step 2: Train model
Write-Host "[2/4] Training ML model..." -ForegroundColor Yellow
python -m ml.train --data outputs/datasets/full_train.parquet --out outputs/models/full_model.pkl --report outputs/models/full_report.json --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Model training failed!" -ForegroundColor Red
    exit 1
}

# Step 3: Run experiments (all scenarios, 5 repeats)
Write-Host "[3/4] Running experiments (all scenarios from scenarios.yaml)..." -ForegroundColor Yellow
Write-Host "  This will take several minutes..." -ForegroundColor Gray
python -m eval.run_experiments --config configs/default.yaml --scenarios-config configs/scenarios.yaml --model outputs/models/full_model.pkl --out outputs/experiments/full_results.json --steps 400 --repeats 5

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Experiments failed!" -ForegroundColor Red
    exit 1
}

# Step 4: Generate summary table
Write-Host "[4/4] Generating summary table..." -ForegroundColor Yellow
python scripts/generate_summary_table.py --results outputs/experiments/full_results.json --out outputs/experiments/summary_table.csv

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Summary table generation failed, but continuing..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "FULL TEST COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results:" -ForegroundColor Cyan
Write-Host "  - Model: outputs/models/full_model.pkl" -ForegroundColor White
Write-Host "  - Training report: outputs/models/full_report.json" -ForegroundColor White
Write-Host "  - Experiment results: outputs/experiments/full_results.json" -ForegroundColor White
Write-Host "  - Summary table: outputs/experiments/summary_table.csv" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review summary_table.csv for scenario performance" -ForegroundColor White
Write-Host "  2. Run: python -m eval.make_figures --results outputs/experiments/full_results.json" -ForegroundColor White
Write-Host "  3. Check outputs/figures/ for publication figures" -ForegroundColor White
