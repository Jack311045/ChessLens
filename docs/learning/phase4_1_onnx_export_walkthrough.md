# Trial 1 CPU FP32 ONNX export and parity

## Scope and locked source

Implemented on `feat/onnx-export-parity`, based on evaluation commit
`30c18ed2b538c6522188d4fed523c4dbea089caa`, in the existing worktree
`C:\ChessLensWork\code\value-temperature-scaling`. Other worktrees and their uncommitted changes
were not modified. No GPU, quantization, HPO, retraining, FastAPI, deployment or new model
selection is included.

The locked candidate is Trial 1, selected previously by validation policy MRR
`0.4887446561907522` among four COMPLETE trials:

- Experiment: `6e87118191e0c95c6c76844aac0ef3a40b6bafab4e0e5c187525b15f6e3a25cf`
- Dataset: `8631050546b8eda75d3fd9654b751bb91e823b74b3deb338d78bd027ee342b96`

| Source | Verified SHA256 |
|---|---|
| Neural manifest | `28a4c82b6f6e3baeeabb6bffbc972bdf20ccc2980226f198a883663c39c8d895` |
| Best state dictionary | `ff42bbc72b888d2817efa0233cc9368194f9c15ab3ce11d88af56a26e88ff1c8` |
| Preprocessor | `9ca75bdf44347a83db6f8361705456efdddb8415953d7f3c12e9ea35d6b37d3b` |
| Effective config | `6f06d8d6b7817df2777d0687533dc548fe3b0cf0e5b3a55fab6ad40cef9717f0` |

[`neural_checkpoint.py`](../../src/chesslens/modeling/neural_checkpoint.py) shares only the
existing manifest/config validation and strict CPU state-dictionary reconstruction between
the evaluation and ONNX CLIs. The checkpoint loader validates completion, artifact sizes/hashes,
identity, supported versions and WDL class order; uses `weights_only=True`; and freezes the
saved preprocessor/model without fitting or modifying them.

## Environment and exporter

Python 3.11.16, Torch 2.14.1+cpu, ONNX 1.23.2, ONNX Runtime 1.32.0, NumPy 2.4.6 were
actually used. ONNX dependencies live in the optional `onnx` extra; the base installation
does not acquire an ONNX dependency. `uv.lock` pins the extra and its dependencies.

The exporter is explicitly `torch.onnx.export(dynamo=False, opset_version=17,
external_data=False)`. This tested TorchScript exporter handles this fixed-spatial CNN without
requiring `onnxscript`. Its deprecation warning is retained in stderr, not suppressed.
The existing fixed `(73,8,8)` policy-layout shape guard also emits a TracerWarning; only
the batch axis is variable, and actual B=1/8/32 executions verify that axis. A future exporter
change must rerun the same parity gates, not silently switch engines or relax tolerances.

Setup, from a standalone PowerShell:

```powershell
Set-Location C:\ChessLensWork\code\value-temperature-scaling
python -m uv sync --frozen --extra onnx --dev --python 3.11
if ($LASTEXITCODE -ne 0) { throw "Environment sync failed" }
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
& $py -c "import sys,torch,onnx,onnxruntime,chesslens; print(sys.version); print(torch.__version__,onnx.__version__,onnxruntime.__version__); print(chesslens.__file__)"
if ($LASTEXITCODE -ne 0) { throw "Import preflight failed" }
```

## Rerun fixture, real parity and benchmark

The same CLI exports both heads, checks the disk ONNX graph with
`onnx.checker.check_model(full_check=True)`, creates a disk-backed ORT CPU session, validates
its input/output names/dtypes/shapes, and checks runtime and batch invariance.
An output directory must be fresh. Existing bundles are never overwritten; failed numerical
runs retain their differences, inputs, outputs and failure status without `_SUCCESS`.
Benchmarking is skipped when parity fails.

Run this after the setup block. It creates timestamped outputs, keeps stdout/stderr separate,
and runs both a synthetic encoding fixture and the real 256-position validation selection.
The fixture exercises white/black turns, known/unknown categories, present/missing numeric
context, castling, en passant, promotions, and fewer than five legal moves.

```powershell
$experiment = "6e87118191e0c95c6c76844aac0ef3a40b6bafab4e0e5c187525b15f6e3a25cf"
$trialRoot = "C:\ChessLensWork\phase32_experiments\studies\real_100k_validation_hpo\trials\trial-0001"
$manifest = Join-Path $trialRoot "neural_output\experiments\$experiment\_manifest.json"
$evaluation = "reports\local\checkpoint_evaluation_100k\trial1_validation_pilot\report.json"
$outputRoot = "C:\ChessLensWork\phase4_onnx"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force (Join-Path $outputRoot "logs") | Out-Null

foreach ($mode in "fixture", "trial1") {
  $out = Join-Path $outputRoot "${mode}_$stamp"
  $flags = @()
  if ($mode -eq "fixture") { $flags = @("--fixture") }
  & $py -m chesslens.modeling.run_onnx_export `
    --neural-manifest $manifest --evaluation-report $evaluation `
    --output $out --max-positions 256 --seed 20240131 `
    --batch-sizes 1 8 32 --threads 2 `
    --warmup 10 --iterations 50 --benchmark-batch-size 8 `
    --exploratory-temperature 1.0428276983739342 --json @flags `
    1> (Join-Path $outputRoot "logs\${mode}_$stamp.stdout.json") `
    2> (Join-Path $outputRoot "logs\${mode}_$stamp.stderr.txt")
  if ($LASTEXITCODE -ne 0) { throw "Export/parity failed: inspect $out and stderr" }
  Get-Content (Join-Path $out "report.md")
}
```

`--modeling-manifest` can relocate the original dataset but must preserve its ID and canonical
manifest hash. `--evaluation-report` pins the four checkpoint/preprocessor/config source hashes
against the prior report; it also checks a supplied exploratory T. Neither option is required
for generic completed checkpoint exports. The real path reproduces only the saved **validation**
selection and then deterministically caps it at 256 using SHA256(seed, game_id, ply).
No train/test split is read and no WDL outcome or generalization metric is computed.

## Actual Trial 1 result

Published bundle: `C:\ChessLensWork\phase4_onnx\trial1`.
Fixture-only bundle: `C:\ChessLensWork\phase4_onnx\fixture` (not marked ready for real handoff).
Logs: `C:\ChessLensWork\phase4_onnx\logs`, including original stdout/stderr, execution times and
an independent disk-reload verification. Wall times were about 11.92 s for fixture and 34.09 s
for the final real export/parity/benchmark invocation. The earlier real run is retained under
`trial1_initial`; it produced the same ONNX hash and parity results.

Real sample: 256 unique `(game_id, ply)` keys; selection fingerprint:
`ad0b5eea3a7667d9dca2eaa8ce410b34ad0b90447f5df9849722c45142a73ba8`.
Actual batches were 256 x B=1, 32 x B=8 and 8 x B=32 on the same encoded input tensors and masks.
Each runtime's multi-batch outputs were also compared against its own B=1 results.

Acceptance is elementwise `np.allclose(ort, pytorch, rtol=1e-4, atol=1e-5)` with PyTorch
as reference, plus shape/finite checks and legal behavior checks:

| Quantity, maximum across runtime comparisons | Max absolute error | p99 absolute error | Over tolerance |
|---|---:|---:|---:|
| Policy logits | 3.4332275390625e-5 | 2.5033950805664062e-6 | 0 |
| WDL logits | 9.5367431640625e-7 | 7.152557373046875e-7 | 0 |
| Raw WDL probabilities (T=1) | 2.25403680165215e-7 | See parity JSON per comparison | 0 |

All seven runtime/batch comparisons matched **256/256 legal top-1, ordered legal top-5 and
WDL argmax**. There were zero near-tie exceptions, unexplained differences or illegal outputs.
Sorting uses descending legal logit, then ascending action index, matching baseline tie-breaking.
Action decoding and re-encoding are checked with python-chess. Any differing cases would include
keys, UCI moves, adjacent score gaps, summed error bounds and observed errors; a near tie can
only excuse a behavior difference if numeric checks also pass and every affected gap is within
the summed tolerance bounds. Otherwise parity fails.

### Forward-only latency

Same local Windows AMD64 CPU, 2 intra-op threads and 1 inter-op thread for both runtimes;
ORT uses CPUExecutionProvider, sequential execution and all graph optimizations. Each runtime
warms up 10 times, then measures 50 iterations with alternating runtime order. Model/session
load, export, encoding, input conversion and postprocessing are excluded; output allocation
and runtime call overhead are included.

| Runtime | Batch | p50 ms | p95 ms |
|---|---:|---:|---:|
| PyTorch eager | 1 | 8.78445 | 11.002410 |
| ONNX Runtime | 1 | 4.30015 | 5.189125 |
| PyTorch eager | 8 | 41.79745 | 47.486110 |
| ONNX Runtime | 8 | 28.12035 | 30.096510 |

These are local observations, not an SLA or an end-to-end API latency claim. Hardware details
and runtime versions are in `parity.json`; timing varies with CPU state and load.
State dictionary size: **7,567,413 bytes**. Complete ONNX graph size: **7,528,264 bytes**;
there is no external-data sidecar. ONNX SHA256:
`63a6249c46cdb1d2767bebb1c3bd0e228a30b5b8821900847d3f1cbc4dfda395`.

## Portable FastAPI handoff

The real bundle has `ready_for_handoff=true`. It contains:

- `model.onnx`: raw FP32 policy/WDL logits, dynamic batch;
- `feature_preprocessing.json`: byte-identical saved preprocessor;
- `metadata.json`: schema, relative filenames, source hashes and IDs, code SHA/dirty status
  plus source hashes, actual I/O, versions, context/missing-flag order, category-map pointers
  and explicit calibration status;
- `model_state_best.pt`, `effective_config.json`, `source_neural_manifest.json`,
  `source_evaluation_report.json`: source/reconstruction evidence;
- `parity.json`, `report.md`, `parity_arrays.npz`: numerical/behavior/benchmark evidence;
- `_manifest.json`, `_SUCCESS`: artifact sizes/hashes and verified publication.

The export records the base Git SHA and dirty status because code had not yet been committed
when the bundle was generated. Per-source hashes pin the actual implementation used. Archival
config/manifest paths are not runtime dependencies; bundle loading resolves relative artifacts.

| Name | dtype | shape |
|---|---|---|
| board_tensor | float32 | B x 18 x 8 x 8 |
| continuous_context | float32 | B x 8 |
| rating_band_index | int64 | B |
| time_control_index | int64 | B |
| policy_logits | float32 | B x 4672 |
| value_logits | float32 | B x 3 |

Preprocessing is graph-external and follows
[the encoding specification](../encoding-specification.md): absolute board orientation,
`[channel, rank, file]`, and `action_index = from_square * 73 + plane`.
Continuous order is `(mover_rating zscore, missing), (opponent_rating zscore, missing),
(rating_difference zscore, missing), (ply zscore, missing)`. Missing numerics encode `(0,1)`;
observed numerics encode `((x-mean)/std,0)`. Unknown categories use saved `__UNK__=0` mappings.
No category map or feature statistic is refit.

Only rank actions allowed by the FEN-derived legal mask; decode using the same board context.
Compute WDL with stable softmax in the mover's **win, draw, loss** order. Production T is
explicitly **1**. The optional T=1.0428276983739342 is checked only as an exploratory graph-external
calculation; it is not embedded in ONNX and is not approved for production calibration.
Runtime parity does not establish model quality, blunder-risk support or independent calibration
acceptance. Those remain separate from serving handoff.

`load_bundle(...)` verifies all relative artifact hashes before creating a CPU session, and
rejects fixture/failed bundles by default. `validate_inputs(...)` checks names, dtype, shape,
finite values and embedding-index range. A future API must use these contracts or equivalent
guards; it must not infer feature order from dictionary sorting.

## Regression commands

The integration tests create one tiny frozen checkpoint locally, requiring no real datasets.
They run a CLI fixture export, move/reload the bundle, exercise B=1/B>1, missing/unknown
context and both turns, verify both heads/ranking, reject provenance/integrity errors,
and inject numerical and nonfinite errors to ensure failed outputs are retained and not promoted.

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
& $py -m pytest tests\unit\test_onnx_parity.py tests\integration\test_onnx_export.py `
  tests\integration\test_checkpoint_evaluation.py tests\unit\test_neural_network.py `
  tests\unit\test_neural_data.py tests\unit\test_calibration.py `
  tests\integration\test_run_neural.py -ra
& $py -m ruff check src\chesslens\modeling\neural_checkpoint.py `
  src\chesslens\modeling\onnx_parity.py src\chesslens\modeling\run_onnx_export.py `
  src\chesslens\modeling\run_checkpoint_evaluation.py `
  tests\unit\test_onnx_parity.py tests\integration\test_onnx_export.py
& $py -m mypy src\chesslens\modeling\neural_checkpoint.py `
  src\chesslens\modeling\onnx_parity.py src\chesslens\modeling\run_onnx_export.py `
  src\chesslens\modeling\run_checkpoint_evaluation.py `
  tests\unit\test_onnx_parity.py tests\integration\test_onnx_export.py
git --no-pager diff --check
```

The run passed 71 tests. Exporter/tracer warnings described above and existing sklearn
convergence/LightGBM deprecation warnings are retained in logs. Ruff, mypy and frozen lockfile
checks passed, including type-checking against the original base environment without the ONNX
extra installed. The documented frozen sync was rerun successfully in the isolated worktree.
An independent final disk reload also verified all bundle hashes, the graph,
dynamic B=1/8/32, zero numerical violations and exact legal/WDL agreement.
