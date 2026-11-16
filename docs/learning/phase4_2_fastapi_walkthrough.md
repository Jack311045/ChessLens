# Phase 4.2: local FastAPI single-position inference

## Scope and implementation source

This milestone serves one standard-chess position per request using the accepted Trial 1
CPU FP32 ONNX bundle. It does not retrain, export a new model, select a new candidate,
apply exploratory temperature scaling, analyze complete games, estimate blunder risk,
build containers or deploy to a cloud.

- Implementation checkout: `C:\ChessLensWork\code\value-temperature-scaling`
- Branch: `feat/fastapi-serving`
- Base commit: `b0e420ba6810350e7634204fc51b304f25f42e8e`
- Existing immutable bundle: `C:\ChessLensWork\phase4_onnx\trial1`
- Local acceptance evidence: `C:\ChessLensWork\phase4_fastapi\trial1_local`

The export metadata records its historical HEAD `30c18ed` with `worktree_dirty=true`.
The actual committed ONNX implementation is `b0e420b`, which is the base of this work.
Neither the bundle metadata nor model bytes were rewritten for serving.
The primary checkout at `C:\vsdeeznutz\ChessLens` retains its existing changes; a copy of
this English walkthrough is also placed in its `docs\learning` directory for review.
That documentation copy does not integrate the implementation branch into the primary checkout.

| Accepted artifact | Identity |
|---|---|
| Experiment / API `model_version` | `6e87118191e0c95c6c76844aac0ef3a40b6bafab4e0e5c187525b15f6e3a25cf` |
| ONNX SHA-256 | `63a6249c46cdb1d2767bebb1c3bd0e228a30b5b8821900847d3f1cbc4dfda395` |
| Preprocessor SHA-256 | `9ca75bdf44347a83db6f8361705456efdddb8415953d7f3c12e9ea35d6b37d3b` |

These are acceptance identities, not hardcoded restrictions in the serving loader.
Other bundles with the supported schema and encoding contract can be configured.

## Install and start in Windows PowerShell

Use the implementation checkout, not the primary checkout that only has review copies.
The documented sync command was executed successfully with the updated lockfile.

```powershell
Set-Location C:\ChessLensWork\code\value-temperature-scaling
python -m uv sync --frozen --extra onnx --extra api --dev --python 3.11
```

One startup command, usable from any working directory after setup:

```powershell
& C:\ChessLensWork\code\value-temperature-scaling\.venv\Scripts\python.exe -m chesslens.serving --bundle C:\ChessLensWork\phase4_onnx\trial1 --port 8765 --threads 2
```

The command binds only `127.0.0.1`, uses one Uvicorn worker, and configures ORT intra-op
threads to 2 / inter-op threads to 1. Stop it with Ctrl+C.
Port 8000 is the CLI default if `--port` is omitted.
Alternatively set `$env:CHESSLENS_BUNDLE_DIR` and omit `--bundle`.
Missing configuration fails startup explicitly; there is no fallback model or dummy prediction.

OpenAPI JSON is at `http://127.0.0.1:8765/openapi.json`; interactive API documentation
is at `http://127.0.0.1:8765/docs`. The temporary acceptance server was stopped after
verification; run the command above to start your own session.

### Call all three endpoints

In another PowerShell terminal:

```powershell
$base = 'http://127.0.0.1:8765'
Invoke-RestMethod "$base/health"
Invoke-RestMethod "$base/v1/model-info" | ConvertTo-Json -Depth 6

$body = @{
    fen = 'r3k1nr/1bqp1ppp/p1n1p3/2b5/BP2P3/2P2N2/P4PPP/RNBQ1RK1 b kq - 0 9'
    context = @{
        mover_rating = 1526
        opponent_rating = 1312
        time_control = '60+0'
    }
    top_k = 5
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri "$base/v1/rank-moves" `
    -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 6

# Context may be omitted entirely.
$withoutContext = @{
    fen = 'r3k1nr/1bqp1ppp/p1n1p3/2b5/BP2P3/2P2N2/P4PPP/RNBQ1RK1 b kq - 0 9'
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$base/v1/rank-moves" `
    -ContentType 'application/json' -Body $withoutContext | ConvertTo-Json -Depth 6
```

The first request is an actual saved validation position, not a fabricated demonstration.
Its recorded request/response are under `http_acceptance` in the evidence directory.

## API contract

| Endpoint | Behavior |
|---|---|
| `GET /health` | `{"status":"ready"}` only after a bundle has loaded successfully. Without initialized lifespan state, returns 503. |
| `GET /v1/model-info` | Experiment version, model/preprocessor hashes, dataset identity, bundle/encoding versions, CPU provider, actual thread settings, WDL class order and temperature. |
| `POST /v1/rank-moves` | Typed request/response with FEN, optional context, ranked legal UCI moves, move probabilities, WDL, model identity and processing time. |

Requests:

- `fen`: all six FEN fields, up to 256 characters. Syntax and `python-chess` standard-board
  validity are checked. Fullmove number must be positive; move counters must fit nonnegative
  signed int32. Four-field normalized position keys are not complete inference requests.
- `context`: optional object with optional/null `mover_rating`, `opponent_rating`,
  and raw `time_control`, such as `"60+0"`. Ratings refer to the current mover and opponent,
  **not** fixed white/black players. Ratings are strict integers in `[0,10000]`;
  time-control strings are at most 64 characters.
- `top_k`: strict integer, default 5, bounded to `[1,218]`. Return count is capped by the
  number of legal moves.
- Unknown request/context fields, invalid boards, bad types and out-of-range values return
  FastAPI's typed 422 validation response. Unready state returns 503; internal inference
  errors propagate as server errors, never successful empty/dummy predictions.

Each move has `rank` (one-based), `uci`, `action_index`, and `probability`.
Policy probabilities are a stable float64 softmax of raw logits over **all legal moves**,
then truncated to top-k. They are not normalized over 4672 actions or renormalized over
the returned subset. Returned probabilities may therefore sum to less than one.
Ranking is descending legal logit, then ascending action index for exact ties.
Every returned UCI is decoded with the existing board-aware action decoder, including
queen promotions and underpromotions.

`wdl` is `{win, draw, loss}` from the **side-to-move perspective**, using stable float64
softmax and **T=1**. Black-to-move probabilities are not swapped into White's perspective.
The earlier exploratory temperature `1.0428276983739342` is not used.
`model_version` is the experiment ID; `model_sha256` pins the exact graph bytes.

A valid terminal position returns HTTP 200 with `status="terminal"`, a `terminal_reason`,
`moves=[]`, and `wdl=null`. Checkmate, stalemate and insufficient material are tested.
Automatic game-over rules use `board.outcome(claim_draw=False)`; a FEN does not provide
repetition history, and claimable draws are not automatically claimed.
`legal_move_count` remains the board's legal-move count even if another terminal rule
prevents ranking.

## Loading and training-consistent encoding

[`serving/bundle.py`](../../src/chesslens/serving/bundle.py) loads inference resources once
per process in FastAPI lifespan:

1. Require supported complete manifest/metadata and `_SUCCESS`.
2. Verify metadata and required inference artifact sizes and SHA-256 hashes.
3. Validate contained relative filenames, supported versions, I/O, context order, CPU FP32,
   no external graph data, WDL order and production temperature.
4. Validate saved continuous statistics and category maps: finite means/stds, positive
   standard deviations, nonnegative counts, contiguous embedding indices and `__UNK__=0`.
5. Create one CPU-only ORT session with fallback disabled; check exact input/output names,
   dtypes, dynamic batch and fixed remaining shapes. Warm up once and validate FP32 finite
   raw-logit outputs, also probing maximum saved embedding indices.

A relocated inference-only bundle needs just `_manifest.json`, `_SUCCESS`, `metadata.json`,
the referenced ONNX model and saved preprocessor. Checkpoints, effective training config,
parity arrays/reports and Parquet files are audit resources, not startup dependencies.
Artifact hashes detect corruption/inconsistency, not authenticity through a digital signature.
The service does not import Torch, the ONNX export package, DuckDB or the training data loader.

[`neural_data.encode_position`](../../src/chesslens/modeling/neural_data.py) is the minimal
label-free shared feature encoder. Existing training/evaluation `encode_examples` delegates
its input features to this helper while retaining its target/legality checks.
The service uses the same board/action encoders and saved preprocessing; nothing is fitted.

- Board input: absolute `[channel,rank,file]`, float32 `[1,18,8,8]`.
- Continuous input: float32 `[1,8]`, ordered mover rating, opponent rating, rating difference,
  ply; each contributes standardized value followed by missing flag.
- Missing numeric value: `(0,1)`; observed zero: `((0-mean)/std,0)`.
- Rating difference is computed only if both ratings are present.
- **Ply is `board.ply() = 2*(fullmove_number-1) + (1 if Black moves else 0)`.**
  All 256 real saved parity rows confirm this relationship exactly; do not add one.
- `rating_band` and raw `time_control_category` reuse the training label helpers.
  Categories absent from the saved map use `__UNK__=0`. In Trial 1, missing time control
  maps to the *known* `"unknown"` entry at index **5**, not blindly to index 0.
- Context omission makes the first three numeric fields missing; ply remains observed.
- Category inputs: int64 `[1]`; outputs: raw FP32 policy `[1,4672]` and value `[1,3]`.

The common CPU session and legal-ordering helpers are also reused by the ONNX parity CLI.
Existing export/evaluation/training regression tests still pass after this extraction.

## Real Trial 1 acceptance

[`scripts/verify_local_api.py`](../../scripts/verify_local_api.py) reproduces the saved
validation selection from the original dataset, rather than estimating ratings by inverting
standardized float32 inputs. It verifies the accepted parity array/report hashes, confirms
the neural source identity, restores original ratings/time-control context, checks all four
encoded inputs against saved arrays, and calls the running API separately for every row.
The expected ranking/probabilities come from the already accepted saved ORT logits.
Only original validation partitions are read by this offline verifier; serving reads no
training data. The verifier's fixed sampling seed is `20240131`, matching this Trial 1 bundle.

Observed:

- Exact restored inputs: **256/256** across all four input arrays.
- HTTP legal ordered Top-5: **256/256** matched.
- Maximum policy-probability absolute difference: **0.0**.
- Maximum WDL-probability absolute difference: **0.0**.
- Real missing-context request: HTTP 200 with legal moves and mover-relative WDL.
- Actual PowerShell POST command and a real terminal-position response: passed.

For the Black-to-move example above, all 49 legal moves participate in normalization:

| Rank | UCI | Probability |
|---|---|---|
| 1 | `c5b6` | 0.281009438 |
| 2 | `c5d6` | 0.116563235 |
| 3 | `g8f6` | 0.082839087 |
| 4 | `c5e7` | 0.078752656 |
| 5 | `c5a7` | 0.069750604 |

WDL: win **0.895777743**, draw **0.003974695**, loss **0.100247563**.
These are model predictions, not a claim of independent calibration or chess-engine accuracy.

### Reproduce real HTTP acceptance

Keep the server running in the first terminal. In a second terminal:

```powershell
Set-Location C:\ChessLensWork\code\value-temperature-scaling
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$experiment = '6e87118191e0c95c6c76844aac0ef3a40b6bafab4e0e5c187525b15f6e3a25cf'
$neuralRoot = 'C:\ChessLensWork\phase32_experiments\studies\real_100k_validation_hpo\trials\trial-0001\neural_output\experiments'
$manifest = Join-Path (Join-Path $neuralRoot $experiment) '_manifest.json'
$out = Join-Path 'C:\ChessLensWork\phase4_fastapi' ("http_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $py scripts\verify_local_api.py `
    --bundle C:\ChessLensWork\phase4_onnx\trial1 `
    --neural-manifest $manifest --url http://127.0.0.1:8765 --output $out
if ($LASTEXITCODE -ne 0) { throw 'Real HTTP acceptance failed' }
Get-Content (Join-Path $out 'report.json')
```

Use a fresh output directory for each run. Acceptance verification needs the local original
validation dataset and audit artifacts; merely running the API does not.

## Timing and checks

`processing_time_ms` starts on entry to `rank_position`, after request validation.
It includes board/terminal checks, feature encoding, ORT execution and Python postprocessing.
It ends just before constructing the final response model; request parsing/validation,
threadpool wait, response serialization, network transfer and client parsing are excluded.
It is **not** the prior forward-only ORT benchmark.

The HTTP measurement uses a persistent `httpx` localhost connection: request construction/
serialization through response-body receipt and successful status check, excluding JSON
decoding. After parity verification, discard 5 further warmups and measure 30 sequential
requests for the same known-context position:

| Measurement | Observed |
|---|---|
| Localhost HTTP p50 | **13.684550 ms** |
| Localhost HTTP p95 | **15.158015 ms** |
| Samples / additional warmup | 30 / 5 |
| Server | one Uvicorn process; ORT intra-op 2 / inter-op 1 |

Environment: Windows build 26200, Intel64 Family 6 Model 197 Stepping 2; OS-reported
16 physical / 16 logical cores; Python 3.11.16, FastAPI 0.143.0, Uvicorn 0.54.0,
ORT 1.32.0, httpx 0.28.1. This small, warmed sequential sample is local evidence,
not a concurrency/load test or production latency guarantee.

Validation completed:

- **112 tests passed** across serving, neural encoding/network, calibration, checkpoint
  evaluation, ONNX parity/export and neural training integration.
- **41 serving tests passed again** after final response-type changes.
- Ruff passed; mypy passed on all 9 changed/new Python files; frozen lock check passed.
- Frozen sync with `--extra onnx --extra api --dev --python 3.11` passed.
- Tests cover lifecycle, metadata, malformed requests, terminal states, both movers,
  promotion UCI, castling/en-passant, tie-breaking, full-legal probability normalization,
  missing-vs-zero features, graph/statistics/version/hash corruption, and a relocated
  minimal bundle without audit artifacts.
- Warnings are retained: the existing legacy exporter/tracer, sklearn/LightGBM warnings,
  and Starlette's deprecation warning about its working httpx TestClient transport.

Run the same targeted regression suite:

```powershell
Set-Location C:\ChessLensWork\code\value-temperature-scaling
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& $py -m pytest tests\integration\test_serving.py tests\unit\test_onnx_parity.py `
    tests\integration\test_onnx_export.py tests\integration\test_checkpoint_evaluation.py `
    tests\unit\test_neural_network.py tests\unit\test_neural_data.py `
    tests\unit\test_calibration.py tests\integration\test_run_neural.py
$targets = @(
    'src\chesslens\serving',
    'src\chesslens\modeling\neural_data.py',
    'src\chesslens\modeling\onnx_parity.py',
    'src\chesslens\modeling\run_onnx_export.py',
    'tests\integration\test_serving.py',
    'scripts\verify_local_api.py'
)
& $py -m ruff check @targets
& $py -m mypy @targets
python -m uv lock --check
git diff --check
```

Machine-readable evidence is in `http_acceptance\report.json`, with the known request,
known response, health and model-info snapshots beside it. Check logs are in `logs`.
Large models, original data and generated runtime evidence remain outside Git.
