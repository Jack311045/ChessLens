# Phase 3.2 Local MLflow + Optuna Walkthrough

Phase 3.2 adds local experiment tracking and bounded HPO loops without launching real-data training by default.

Scope in this phase:

- local MLflow tracking backed by SQLite + local artifact root;
- explicit historical-import CLI for completed baseline/neural manifests;
- bounded Optuna study execution with resume-safe storage, identity freeze, and single-writer locking;
- validation-only neural HPO mode that avoids test-split evaluation.

## 1) Key Modules

Tracking config and import flow:

- `src/chesslens/modeling/mlflow_tracking_config.py`
- `src/chesslens/modeling/mlflow_tracking.py`
- `src/chesslens/modeling/run_mlflow_import.py`

HPO config and runner:

- `src/chesslens/modeling/optuna_study_config.py`
- `src/chesslens/modeling/run_neural_hpo.py`

Neural scope support:

- `src/chesslens/modeling/neural_training.py`
- `src/chesslens/modeling/run_neural.py`

## 2) Safety/Determinism Rules

- HPO study identity is frozen on first run from reference config + search space + budget.
- Re-running with changed identity inputs under the same study name fails fast.
- Only one writer can mutate a study at a time (`.study.lock`).
- `execution.trial_concurrency` controls Optuna trial serialization and is fixed at `1` for this implementation.
- `runtime.threads` controls PyTorch compute threads inside a trial and remains independent from trial concurrency.
- Validation-only HPO runs evaluate on validation metrics and skip test-split evaluation.
- Historical MLflow import validates artifact integrity before logging.

## 3) Configs Added

- `configs/mlflow/local_tracking.yaml`
- `configs/hpo/fixture_smoke.yaml`
- `configs/hpo/local_cpu_smoke.yaml`
- `configs/hpo/real_100k_reference.yaml`

## 4) PowerShell Runbook (A-H)

All commands below explicitly use the repository interpreter:

- `C:\vsdeeznutz\ChessLens\.venv\Scripts\python.exe`

When capturing JSON, keep stdout/stderr separate. Do not merge streams with `2>&1`.

### A. Verify interpreter and dependencies

```powershell
$py = "C:\vsdeeznutz\ChessLens\.venv\Scripts\python.exe"
if (!(Test-Path $py)) { throw "Interpreter not found: $py" }

& $py -c "import sys; print(sys.executable); print(sys.version)"
& $py -c "import mlflow, optuna, torch, yaml; print('deps_ok')"

$uvFromPath = Get-Command uv -ErrorAction SilentlyContinue
if ($uvFromPath) {
  uv --version
  uv sync --frozen
} else {
  $uvModule = & $py -c "import importlib.util; print(bool(importlib.util.find_spec('uv')))"
  if ($uvModule -eq "True") {
    & $py -m uv --version
    & $py -m uv sync --frozen
  } else {
    Write-Host "uv not detected; using the existing .venv as-is"
  }
}
```

### B. Resolve exact historical 100k manifests and prepare local config

```powershell
$refRoot = "C:/ChessLensWork/phase31_neural/data_scaling_100k"
$baselineId = (Get-Content "$refRoot/baseline_experiment_id.txt").Trim()
$neuralId = (Get-Content "$refRoot/neural_experiment_id.txt").Trim()

$baselineManifest = "$refRoot/baselines/experiments/$baselineId/_manifest.json"
$neuralManifest = "$refRoot/neural/experiments/$neuralId/_manifest.json"

if (!(Test-Path $baselineManifest)) { throw "Missing baseline manifest: $baselineManifest" }
if (!(Test-Path $neuralManifest)) { throw "Missing neural manifest: $neuralManifest" }

New-Item -ItemType Directory -Force -Path reports/local | Out-Null

& $py -m chesslens.modeling.prepare_real_hpo_config `
  --reference-root $refRoot `
  --phase32-output-root "C:/ChessLensWork/phase32_experiments" `
  --output-config configs/hpo/real_100k_reference.yaml `
  --json `
  1> reports/local/phase32_prepare_real_hpo_config.json `
  2> reports/local/phase32_prepare_real_hpo_config.stderr.txt
```

### C. Import historical baseline/neural experiments

```powershell
& $py -m chesslens.modeling.run_mlflow_import `
  --tracking-config configs/mlflow/local_tracking.yaml `
  --kind baseline `
  --manifest $baselineManifest `
  --mode historical `
  --json `
  1> reports/local/phase32_mlflow_import_baseline.json `
  2> reports/local/phase32_mlflow_import_baseline.stderr.txt

& $py -m chesslens.modeling.run_mlflow_import `
  --tracking-config configs/mlflow/local_tracking.yaml `
  --kind neural `
  --manifest $neuralManifest `
  --mode historical `
  --json `
  1> reports/local/phase32_mlflow_import_neural_1.json `
  2> reports/local/phase32_mlflow_import_neural_1.stderr.txt

# Idempotency check (same logical run, reused_existing=true)
& $py -m chesslens.modeling.run_mlflow_import `
  --tracking-config configs/mlflow/local_tracking.yaml `
  --kind neural `
  --manifest $neuralManifest `
  --mode historical `
  --json `
  1> reports/local/phase32_mlflow_import_neural_2.json `
  2> reports/local/phase32_mlflow_import_neural_2.stderr.txt
```

### D. Start local MLflow UI in a separate terminal

```powershell
# Run in a separate terminal window/tab.
& $py -m mlflow ui `
  --backend-store-uri "sqlite:///data/mlflow/tracking.db" `
  --default-artifact-root "data/mlflow/artifacts" `
  --host 127.0.0.1 `
  --port 5000
```

### E. Real HPO dry-run (expected counts/fingerprints/budget)

```powershell
& $py -m chesslens.modeling.run_neural_hpo `
  --config configs/hpo/real_100k_reference.yaml `
  --dry-run `
  --json `
  1> reports/local/phase32_real_hpo_dry_run.json `
  2> reports/local/phase32_real_hpo_dry_run.stderr.txt
```

Expected from the JSON payload:

- `max_total_trials = 4`
- `max_new_trials_per_invocation = 1`
- `execution.trial_concurrency = 1`
- `reference_preflight.selection_fingerprints` and `reference_preflight.training_population` present

### F. Start exactly one new trial

```powershell
& $py -m chesslens.modeling.run_neural_hpo `
  --config configs/hpo/real_100k_reference.yaml `
  --max-new-trials 1 `
  --json `
  1> reports/local/phase32_real_hpo_start_one.json `
  2> reports/local/phase32_real_hpo_start_one.stderr.txt
```

### G. Resume and status

```powershell
& $py -m chesslens.modeling.run_neural_hpo `
  --config configs/hpo/real_100k_reference.yaml `
  --resume `
  --json `
  1> reports/local/phase32_real_hpo_resume.json `
  2> reports/local/phase32_real_hpo_resume.stderr.txt

& $py -m chesslens.modeling.run_neural_hpo `
  --config configs/hpo/real_100k_reference.yaml `
  --status `
  --json `
  1> reports/local/phase32_real_hpo_status.json `
  2> reports/local/phase32_real_hpo_status.stderr.txt
```

### H. Export and validate best config (no final training)

```powershell
& $py -m chesslens.modeling.run_neural_hpo `
  --config configs/hpo/real_100k_reference.yaml `
  --status `
  --export-best-config "C:/ChessLensWork/phase32_experiments/exports/real_100k_best.yaml" `
  --export-final-output-root "C:/ChessLensWork/phase32_experiments/final_runs/real_100k_best" `
  --json `
  1> reports/local/phase32_real_hpo_export.json `
  2> reports/local/phase32_real_hpo_export.stderr.txt
```

The export path validation performs a dry-run fingerprint parity check only. It does not start final training and does not evaluate test data.

## 5) Real-Data Guardrail

Do not launch real-data training from this walkthrough. Use dry-run, status, fixture-safe checks, and export verification only unless an explicit real-data execution gate is approved.
