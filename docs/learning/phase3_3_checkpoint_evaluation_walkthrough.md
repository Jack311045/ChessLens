# Frozen Checkpoint WDL Evaluation

## Scope and reproducibility contract

The entry point is
[`run_checkpoint_evaluation.py`](../../src/chesslens/modeling/run_checkpoint_evaluation.py).
It reuses the saved ResNet and preprocessor; it does not train a model, refit features,
start HPO, import MLflow runs, export ONNX, or serve an API.

The default mode is deliberately **exploratory validation pilot**, not final test acceptance.
The CLI:

1. Verifies the neural identity, complete status, `_SUCCESS`, and all recorded artifact sizes/hashes.
2. Loads the saved effective config, validates supported versions, and checks modeling dataset ID
   and canonical manifest SHA256. A modeling-manifest override changes location, not identity.
3. Reproduces train/validation/test selection fingerprints and the effective training-population
   fingerprint using **key/eligibility projections, without loading train/test labels**.
   The existing modeling preflight checks split metadata, schema and game overlap; it does not
   calculate test performance.
4. Reproduces the saved validation selection. The pilot cap is applied with a stable SHA256
   ordering of `(seed, game_id, ply)`. Selected games are SHA256 ordered by `(seed, game_id)`;
   the first floor-half of games is fit, the rest check. No selected game crosses groups.
   This is a game-disjoint re-split of the existing temporal validation window, **not** a new
   temporal boundary or an independent calibration set.
5. Reconstructs `PolicyValueResidualCNN` and `NeuralPreprocessor.from_dict`, loads only the
   project's state dictionary with `torch.load(..., weights_only=True)` and strict parameter
   keys/shapes, then uses CPU `eval()` / `no_grad()` inference in bounded batches.
   Only WDL logits are retained, not the full policy-logit matrix.
6. Fits one CPU float64 temperature using **fit logits/labels only**. Reports raw and calibrated
   metrics on exactly the same check rows. Positive-T argmax changes fail with affected keys
   for numerical investigation rather than silently accepting changed predictions.
7. Publishes atomically under a single-writer lock. Repeating the same invocation recomputes
   the evaluation and validates that its report and published artifacts match. Changed inputs,
   code, runtime settings or corrupted outputs require a new output directory; nothing is overwritten.

Fewer than two games, or fewer than two positions in either group, fails explicitly. Fewer
than 30 games / 100 positions in a group, or missing WDL classes, is labelled low support.
These cutoffs are warnings for smoke/pilot evidence, not statistical acceptance thresholds.
Historical modeling manifests do not pin policy Parquet content hashes: reproducing keys
cannot prove historical label/feature byte identity. The report hashes current validation files
and explicitly retains this limitation. Original training holdout audits are not re-executed.

## Portable invocation

Run from the checkout containing this entry point, using that checkout's environment:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m chesslens.modeling.run_checkpoint_evaluation `
  --neural-manifest "<completed-neural-experiment>\_manifest.json" `
  --modeling-manifest "<same-modeling-dataset>\_manifest.json" `
  --output "reports\local\checkpoint_pilot" `
  --selection-rationale "Checkpoint locked using validation policy MRR before pilot metrics" `
  --seed 20240131 --max-positions 5000 --batch-size 128 --threads 2 --json
if ($LASTEXITCODE -ne 0) { throw "Checkpoint evaluation failed" }
```

`--modeling-manifest` is optional if the saved location still exists.
`--baseline-manifest` adds integrity-checked **full-validation context only**.
`--selection-evidence` accepts an existing `reports/trials.json` snapshot and verifies that the
checkpoint maximizes validation policy MRR among COMPLETE trials (ties: lowest trial number).
The CLI never calls the HPO runner's `--status` / `--dry-run`, which can write study state.

Outputs:

- `report.json`: model/data IDs and source hashes; versions and WDL order; fit/check positions,
  games, class supports and fingerprints; T; check NLL/log-loss, multiclass Brier, top-label ECE,
  accuracy, confusion matrix, and classwise reliability bins; lineage, limitations and protocol.
- `report.md`: concise human-readable results and source/hash references.
- `predictions.npz`: aligned `(game_id, ply)`, fit/check membership, targets, WDL logits,
  raw probabilities and calibrated probabilities. Load with `allow_pickle=False`.
- `_manifest.json`, `_SUCCESS`: publication integrity and completion markers.

The WDL order is always **win, draw, loss** from the mover's perspective. Brier is the mean
sum of squared errors across all three classes, not divided by three. ECE and reliability
bins reuse baseline metric definitions and the saved neural config's number of bins.
Full-validation model-selection aggregates must not be subtracted from pilot-check metrics.
Baseline prediction samples are capped and are not assumed to be complete paired evidence;
the report therefore claims no LightGBM/ResNet ranking delta, logistic/ResNet WDL delta, or CI.
Those require fresh full aligned predictions on a single locked evaluation population.

## Verified local real-checkpoint pilot

Implementation branch: `feat/checkpoint-evaluation`.
Base: `575fc765d2a7bbf544674fe82f07aee3c75bf2d4` (calibration PR #17).
The original checkout's uncommitted changes were left untouched. Implementation is in the
existing clean worktree `C:\ChessLensWork\code\value-temperature-scaling`, not the original
Phase 3.2 branch.

Read-only inspection found four COMPLETE HPO trials:

| Trial | Validation policy MRR |
|---|---:|
| 0 | 0.486014625132483 |
| **1** | **0.4887446561907522** |
| 2 | 0.48724304133421903 |
| 3 | 0.4875698294949637 |

Trial 1 was selected **before computing pilot metrics**, not by WDL or test results.
The study's frozen budget is 4 total trials, 1 new trial per invocation; 4 have completed,
0 remain. Recorded trial execution time sums to about 227.64 minutes. No new training,
resume, sampler write, or MLflow mutation was performed.

Frozen neural experiment:
`6e87118191e0c95c6c76844aac0ef3a40b6bafab4e0e5c187525b15f6e3a25cf`.
Modeling dataset:
`8631050546b8eda75d3fd9654b751bb91e823b74b3deb338d78bd027ee342b96`.
Canonical modeling manifest SHA256:
`cb2d01eb18ba66b8c15a3d9b0617036a11fbd47c67d53f7f8ba3e7f294d02bfa`.

The original selection has 100,000 train positions, of which 81,544 are used under
`player_disjoint`, and 5,000 validation positions. The temporal validation window is
2017-01-21 through 2017-01-25. This data was already used for early stopping and HPO.

CPU pilot: max 5,000 positions, 2 threads, batch 128, seed 20240131.
First observed wall time was about 49.5 seconds on the existing local environment.
The final implementation then completed twice in 56.23 and 50.12 seconds, producing identical
`report.json` SHA256:
`07af37caff4c036a4547c6d709822f0054cc89b33d473e7a7eccae7ef9310374`.

| Population | Games | Positions | Win | Draw | Loss |
|---|---:|---:|---:|---:|---:|
| Fit | 404 | 2,465 | 1,135 | 127 | 1,203 |
| Check | 404 | 2,535 | 1,145 | 162 | 1,228 |

T = **1.0428276983739342**. Game overlap = 0. Argmax differences = 0.

| Check metric | Raw | Calibrated |
|---|---:|---:|
| NLL | 0.8385329944497683 | 0.8372198826947286 |
| Multiclass Brier | 0.5196558714213465 | 0.5189176237047871 |
| ECE | 0.03332854280483015 | 0.02642846626582373 |
| Accuracy | 0.5783037475345167 | 0.5783037475345167 |

These are small exploratory improvements, not an independent acceptance result and not
evidence of a policy MRR improvement. The optional baseline has the same modeling-manifest
hash and training-population summary, but no verified complete pilot-check predictions;
it is explicitly **not directly comparable** in this report.

Local artifacts are under
`C:\ChessLensWork\code\value-temperature-scaling\reports\local\checkpoint_evaluation_100k`:
`trials_snapshot.json`, `pilot.attempt1.stdout.json`, `pilot.attempt1.stderr.txt`,
`pilot.attempt2.stdout.json`, `pilot.attempt2.stderr.txt`, `execution.log`, `pytest.log`,
`ruff.log`, `mypy.log`, `artifact_verification.log`, and
`trial1_validation_pilot\{report.json,report.md,predictions.npz,_manifest.json,_SUCCESS}`.
They are ignored experiment outputs, not committed source.

### Standalone PowerShell rerun with logs

This block uses the existing interpreter but explicitly imports the evaluation checkout.
It can run in a separate PowerShell after VS Code closes. It does not need a background
server, editable-install changes, dependencies, retraining or HPO budget.

```powershell
Set-Location C:\ChessLensWork\code\value-temperature-scaling
$py = "C:\vsdeeznutz\ChessLens\.venv\Scripts\python.exe"
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
& $py -c "import chesslens; print(chesslens.__file__)"
if ($LASTEXITCODE -ne 0) { throw "Import preflight failed" }

$logs = Join-Path (Get-Location) "reports\local\checkpoint_evaluation_100k"
$study = "C:\ChessLensWork\phase32_experiments\studies\real_100k_validation_hpo"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$snapshot = Join-Path $logs "trials_snapshot.json"
if (!(Test-Path $snapshot)) {
  Copy-Item (Join-Path $study "reports\trials.json") $snapshot
}
$trials = (Get-Content $snapshot -Raw | ConvertFrom-Json).trials
$winner = $trials | Where-Object state -eq "COMPLETE" |
  Sort-Object value -Descending | Select-Object -First 1
$expected = "6e87118191e0c95c6c76844aac0ef3a40b6bafab4e0e5c187525b15f6e3a25cf"
if ($winner.user_attrs.neural_experiment_id -ne $expected) {
  throw "Snapshot changed: review candidate selection before evaluation"
}
$reference = "C:\ChessLensWork\phase31_neural\data_scaling_100k"
$baselineId = (Get-Content (Join-Path $reference "baseline_experiment_id.txt")).Trim()
$baseline = Join-Path $reference "baselines\experiments\$baselineId\_manifest.json"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$rationale = "Locked before pilot metrics: maximum validation policy MRR among 4/4 COMPLETE HPO trials; trial 1 MRR=0.4887446561907522; budget exhausted, no new training."

& $py -m chesslens.modeling.run_checkpoint_evaluation `
  --neural-manifest $winner.user_attrs.neural_manifest_path `
  --baseline-manifest $baseline --selection-evidence $snapshot `
  --selection-rationale $rationale `
  --output (Join-Path $logs "trial1_validation_pilot") `
  --seed 20240131 --max-positions 5000 --batch-size 128 --threads 2 --json `
  1> (Join-Path $logs "$stamp.stdout.json") `
  2> (Join-Path $logs "$stamp.stderr.txt")
if ($LASTEXITCODE -ne 0) {
  throw "Pilot failed; inspect $logs\$stamp.stderr.txt"
}
Get-Content (Join-Path $logs "trial1_validation_pilot\report.md")
```

## Validation commands

The integration fixture trains one tiny checkpoint once, freezes it, and runs the CLI twice
against the same weights. It has twelve validation games and twenty-four validation rows.
Tests verify game-disjoint deterministic groups, fitter inputs, validation-only label loading,
dataset/selection/checkpoint/class-order mismatch failures, strict state shapes, completion
markers, output tamper rejection, HPO winner checks and unchanged model/preprocessor files.

```powershell
Set-Location C:\ChessLensWork\code\value-temperature-scaling
$py = "C:\vsdeeznutz\ChessLens\.venv\Scripts\python.exe"
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
& $py -m pytest tests\integration\test_checkpoint_evaluation.py `
  tests\unit\test_calibration.py tests\integration\test_run_neural.py `
  tests\unit\test_neural_data.py tests\integration\test_run_baselines.py -ra
& $py -m ruff check src\chesslens\modeling\run_checkpoint_evaluation.py `
  src\chesslens\modeling\run_baselines.py tests\integration\test_checkpoint_evaluation.py
& $py -m mypy src\chesslens\modeling\run_checkpoint_evaluation.py `
  src\chesslens\modeling\run_baselines.py tests\integration\test_checkpoint_evaluation.py
git --no-pager diff --check
```

The first regression run exposed an incorrect *test expectation*: two fixture manifests
had identical identity JSON but different selected keys. The test now independently exercises
both key mismatch and changed manifest identity. The subsequent 54-test run passed.
The final run, including shared baseline-selection regressions, passed all **71 tests**
in 193.82 seconds (17 existing warnings).
Existing sklearn convergence and LightGBM deprecation warnings remain unrelated to the pilot.
Ruff and mypy passed. Pylance's editor syntax endpoint could not access the other worktree;
actual interpreter imports and pytest execution validated syntax instead.
An independent read-back check also verified all source/output hashes, 5,000 unique aligned
keys and disjoint game groups. Refitting T from the saved fit-only arrays reproduced T;
recomputing check NLL (from logits), Brier and ECE matched the report within `1e-12`.

## Formal acceptance and ONNX handoff

Before any formal test evaluation:

1. Lock a checkpoint using validation-only selection evidence.
2. Obtain previously unused calibration games excluded from training, HPO and early stopping.
   Retain the temporal and player-holdout contracts; record complete split and population lineage.
3. Fit and lock T using only that calibration set.
4. Evaluate raw/scaled WDL once on an independent, pre-reserved test population. Predeclare
   acceptance criteria and paired game-level bootstrap intervals on aligned full predictions.
   If any existing test informed model choice or calibration adoption, use new unused games.
   External historical test usage is currently unknown; this pilot makes no independence claim.

Next, use the exact `model_state_best.pt` and `feature_preprocessing.json` paths and SHA256
values in `report.json`, the saved architecture config, and WDL order `(win, draw, loss)`
for PyTorch/ONNX parity. This pilot's T is only a labelled candidate, not an approved production
calibrator. Start FastAPI work only after parity; serving is outside this change.
