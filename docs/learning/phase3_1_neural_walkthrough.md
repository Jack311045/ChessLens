# Phase 3.1 Neural Training Walkthrough

This walkthrough maps the Phase 3.1 math to concrete code paths and tensor layouts.

## 1) Data + Identity Pipeline

Entry point:
- `src/chesslens/modeling/run_neural.py`
- `src/chesslens/modeling/neural_training.py`

Selection and compatibility logic:
- Reuses deterministic row selection and train-population semantics from Phase 2.2.
- Selection fingerprints are recorded as split key hashes plus a training-population hash.
- Optional baseline manifest compatibility compares expected vs actual selection hashes.

Relevant functions:
- `run_neural(...)` in `neural_training.py`
- `_selection_inputs(...)` in `neural_training.py`
- `selection_fingerprints(...)` in `neural_data.py`
- `compare_selection_fingerprints(...)` in `neural_data.py`

## 2) Feature Encoding and Shapes

Board encoding:
- `encode_fen_18x8x8(...)` from position encoding.
- Tensor shape: `[B, 18, 8, 8]`.

Action space:
- Canonical action space size is `4672 = 64 * 73`.
- Legal mask shape: `[B, 4672]`.
- Policy target is a single integer action index per row.

Context encoding:
- Continuous base fields: mover/opponent rating, rating difference, ply.
- Each continuous field expands to `(zscore, missing_flag)`.
- Continuous context shape becomes `[B, 8]` for these four fields.
- Categorical context uses embeddings for rating-band and time-control category.

Relevant functions:
- `fit_preprocessor(...)` in `neural_data.py`
- `encode_examples(...)` in `neural_data.py`

## 3) Model Math

Model:
- `PolicyValueResidualCNN` in `neural_network.py`.
- Trunk: stem conv + residual blocks over board planes.
- Context branch: continuous MLP + two embeddings.
- Context is fused into trunk via projected bias and also fed into the value head.

Policy head mapping:
- Raw policy logits before flatten: `[B, 73, 8, 8]`.
- Flatten order is encoder-compatible via:
  - `flatten_policy_spatial_logits(...)`
  - internal mapping equivalent to `(square_index * 73 + plane_index)`.

Value head:
- Outputs `[B, 3]` logits for W/D/L in order `("win", "draw", "loss")`.

## 4) Supervised Objective

Policy loss:
- Cross-entropy over masked logits, where illegal actions receive a very negative logit.

Value loss:
- Cross-entropy over W/D/L logits.

Combined loss:
- `L_total = w_policy * L_policy + w_value * L_value`
- Implemented in `compute_supervised_losses(...)` in `neural_losses.py`.

## 5) Runtime Safety

Resume + checkpoints:
- `last.pt` and `best.pt` are written under work checkpoints.
- Resume requires matching experiment identity hash and experiment ID.
- Session-limited training is supported with `--max-epochs-this-session`.

Publication:
- Artifacts are written to staging first, then atomically promoted to experiment root.
- Manifest records artifact checksums and sizes.
- Existing completed experiments are reused only if integrity checks pass.

## 6) PowerShell Command Blocks (A-E)

A) Preflight and dry-run

```powershell
$env:CHESSLENS_MODELING_MANIFEST_PATH = "data/modeling/datasets/<modeling_dataset_id>/_manifest.json"
C:\vsdeeznutz\ChessLens\.venv\Scripts\python.exe -m chesslens.modeling.run_neural `
  --config configs/neural/fixture_smoke.yaml `
  --modeling-manifest-path $env:CHESSLENS_MODELING_MANIFEST_PATH `
  --dry-run --json
```

B) Validate-only (selection + encoding + legality)

```powershell
C:\vsdeeznutz\ChessLens\.venv\Scripts\python.exe -m chesslens.modeling.run_neural `
  --config configs/neural/fixture_smoke.yaml `
  --modeling-manifest-path $env:CHESSLENS_MODELING_MANIFEST_PATH `
  --validate-only --json
```

C) Start bounded training session

```powershell
C:\vsdeeznutz\ChessLens\.venv\Scripts\python.exe -m chesslens.modeling.run_neural `
  --config configs/neural/local_cpu_smoke.yaml `
  --modeling-manifest-path $env:CHESSLENS_MODELING_MANIFEST_PATH `
  --max-epochs-this-session 2 --json
```

D) Status and resume

```powershell
C:\vsdeeznutz\ChessLens\.venv\Scripts\python.exe -m chesslens.modeling.run_neural `
  --config configs/neural/local_cpu_smoke.yaml `
  --modeling-manifest-path $env:CHESSLENS_MODELING_MANIFEST_PATH `
  --status --json

C:\vsdeeznutz\ChessLens\.venv\Scripts\python.exe -m chesslens.modeling.run_neural `
  --config configs/neural/local_cpu_smoke.yaml `
  --modeling-manifest-path $env:CHESSLENS_MODELING_MANIFEST_PATH `
  --resume --json
```

E) Baseline-compatibility check during neural validation

```powershell
$baselineManifest = "data/baselines/experiments/<baseline_experiment_id>/_manifest.json"

C:\vsdeeznutz\ChessLens\.venv\Scripts\python.exe -m chesslens.modeling.run_neural `
  --config configs/neural/local_baseline_compare.yaml `
  --modeling-manifest-path $env:CHESSLENS_MODELING_MANIFEST_PATH `
  --baseline-reference-manifest-path $baselineManifest `
  --validate-only --json
```
