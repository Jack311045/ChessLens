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
- Validation-only HPO runs evaluate on validation metrics and skip test-split evaluation.
- Historical MLflow import validates artifact integrity before logging.

## 3) Configs Added

- `configs/mlflow/local_tracking.yaml`
- `configs/hpo/fixture_smoke.yaml`
- `configs/hpo/local_cpu_smoke.yaml`
- `configs/hpo/real_100k_reference.yaml`

## 4) PowerShell Runbook (Fixture-Safe)

All commands below use the repository venv interpreter directly.

A) Dry-run neural training plan (no training):

```powershell
$py = "C:\vsdeeznutz\ChessLens\.venv\Scripts\python.exe"
$manifest = "data/modeling/datasets/fixture_2013_01_first20/_manifest.json"

& $py -m chesslens.modeling.run_neural `
  --config configs/neural/fixture_smoke.yaml `
  --modeling-manifest-path $manifest `
  --evaluation-scope validation_only `
  --dry-run --json
```

B) Validate-only neural preprocessing/encoding checks:

```powershell
& $py -m chesslens.modeling.run_neural `
  --config configs/neural/fixture_smoke.yaml `
  --modeling-manifest-path $manifest `
  --validate-only --json
```

C) Import an existing completed baseline manifest into local MLflow:

```powershell
$baselineManifest = "data/baselines/experiments/<baseline_experiment_id>/_manifest.json"

& $py -m chesslens.modeling.run_mlflow_import `
  --tracking-config configs/mlflow/local_tracking.yaml `
  --kind baseline `
  --manifest $baselineManifest `
  --mode historical --json
```

D) Import an existing completed neural manifest into local MLflow:

```powershell
$neuralManifest = "data/neural/experiments/<neural_experiment_id>/_manifest.json"

& $py -m chesslens.modeling.run_mlflow_import `
  --tracking-config configs/mlflow/local_tracking.yaml `
  --kind neural `
  --manifest $neuralManifest `
  --mode historical --json
```

E) HPO budget/status checks before any trial starts:

```powershell
& $py -m chesslens.modeling.run_neural_hpo `
  --config configs/hpo/fixture_smoke.yaml `
  --dry-run --json

& $py -m chesslens.modeling.run_neural_hpo `
  --config configs/hpo/fixture_smoke.yaml `
  --status --json
```

F) Start one bounded fixture HPO trial and resume if interrupted:

```powershell
& $py -m chesslens.modeling.run_neural_hpo `
  --config configs/hpo/fixture_smoke.yaml `
  --max-new-trials 1 `
  --resume --json
```

G) Export best tuned neural config from a completed study:

```powershell
& $py -m chesslens.modeling.run_neural_hpo `
  --config configs/hpo/fixture_smoke.yaml `
  --status `
  --export-best-config data/neural_hpo/exports/fixture_best.yaml `
  --json
```

## 5) Real-Data Guardrail

Do not launch real-data training from this walkthrough. Use dry-run/status/fixture validation only unless an explicit real-data execution gate is approved.
