# Smoke test: Quick validation that entire pipeline works
# Usage: .\scripts\smoke_test.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FloodMAS Smoke Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

# Step 1: Generate data
Write-Host "[1/5] Generating synthetic data (50 episodes)..." -ForegroundColor Yellow
python -m ml.generate_data --episodes 50 --steps 100 --out outputs/datasets/smoke.parquet --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Data generation failed!" -ForegroundColor Red
    exit 1
}

# Step 2: Train model
Write-Host "[2/5] Training ML model..." -ForegroundColor Yellow
python -m ml.train --data outputs/datasets/smoke.parquet --out outputs/models/smoke_model.pkl --report outputs/models/smoke_report.json --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Model training failed!" -ForegroundColor Red
    exit 1
}

# Step 3: Run simulation (normal_wet scenario)
Write-Host "[3/5] Running simulation (normal_wet)..." -ForegroundColor Yellow
python -m sim.model --model outputs/models/smoke_model.pkl --scenario normal_wet --steps 150 --log outputs/logs/smoke_normal_wet.parquet

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Simulation failed!" -ForegroundColor Red
    exit 1
}

# Step 4: Run mini experiment (2 scenarios)
Write-Host "[4/5] Running mini experiment (2 scenarios)..." -ForegroundColor Yellow
# Create temp scenarios file
$tempScenarios = @"
scenarios:
  - name: normal_wet
    rainfall_type: normal
    soil_saturation_init: 0.6
    dropout_rate: 0.0
    noise_level: low
    
  - name: extreme_wet
    rainfall_type: extreme
    soil_saturation_init: 0.7
    dropout_rate: 0.0
    noise_level: low

noise_levels:
  low:
    sensor_noise_std: 0.03
"@

$tempScenarios | Out-File -FilePath "outputs/smoke_scenarios.yaml" -Encoding UTF8

python -m eval.run_experiments --config configs/default.yaml --scenarios-config outputs/smoke_scenarios.yaml --model outputs/models/smoke_model.pkl --out outputs/experiments/smoke_results.json --steps 150 --repeats 1

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Experiments failed!" -ForegroundColor Red
    exit 1
}

# Step 5: Verify outputs
Write-Host "[5/5] Verifying outputs..." -ForegroundColor Yellow

$expectedFiles = @(
    "outputs/datasets/smoke.parquet",
    "outputs/models/smoke_model.pkl",
    "outputs/models/smoke_report.json",
    "outputs/logs/smoke_normal_wet.parquet",
    "outputs/experiments/smoke_results.json"
)

$allExist = $true
foreach ($file in $expectedFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file (MISSING)" -ForegroundColor Red
        $allExist = $false
    }
}

Write-Host ""
if ($allExist) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "SMOKE TEST PASSED!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "All pipeline components working correctly." -ForegroundColor Green
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "SMOKE TEST FAILED!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Some outputs are missing. Check errors above." -ForegroundColor Red
    exit 1
}
