# Phase 1.2d Walkthrough: Safe Parallel Shard Ingestion

This walkthrough explains how to run shard ingestion faster with process-based
parallelism while preserving all Phase 1.2c identity and resume guarantees.

## 1. What changed in Phase 1.2d

Phase 1.2d keeps the exact same data contracts and identities from Phase 1.2c, but
adds a parallel coordinator in `run_sharded_ingestion`.

New capabilities:

- `--workers N` to run up to `N` shard ingestion workers concurrently.
- single-writer collection lock for mutating runs.
- in-order manifest commits even when workers finish out of order.
- run-level wall-clock + process-tree memory metrics.

What did **not** change:

- `game_id`, `dataset_id`, `collection_id`, `SourceLineage` semantics.
- schema/encoding/pipeline version constants.
- HMAC secret handling rules.

## 2. Thread vs process (and why processes)

### Threads

Threads share one Python process and one Global Interpreter Lock (GIL). For CPU-heavy
Python work, threads often cannot run Python bytecode in true parallel.

### Processes

Processes have separate interpreters and separate memory spaces. With a process pool,
multiple shard ingestions can run at the same time on different CPU cores.

Phase 1.2d uses a process pool (`ProcessPoolExecutor`) for shard-level parallelism.

## 3. Coordinator vs worker roles

### Coordinator (parent process)

The coordinator is the only process that:

- selects pending shard indices;
- acquires/releases the collection lock;
- updates `_collection_manifest.json`;
- enforces contiguous shard-index commit order;
- reports run-level progress and metrics.

### Workers (child processes)

Each worker:

- receives one shard task;
- calls existing `run_ingestion` with `SourceLineage`;
- writes only to that shard's deterministic output root;
- returns success/failure and metrics to the coordinator.

Workers never mutate the collection manifest.

## 4. Why out-of-order completion is safe

Workers may finish in any order. Example with workers 3:

- shard 6 may finish before shard 5.
- coordinator buffers successful results.
- coordinator commits only the contiguous prefix in order (`..., 4, 5, 6`).

If shard 5 fails but shard 6 succeeds:

- shard 6 dataset may exist on disk;
- manifest does not advance past missing shard 5;
- resume can process shard 5 and reuse shard 6 deterministically.

## 5. Collection lock and stale-lock recovery

Mutating runs create a collection lock file atomically.

- if another active mutator holds the lock: run fails loudly.
- if lock owner process is gone: lock is treated as stale and recovered.
- read-only operations (`--status`) stay lock-free.

This prevents two orchestrators from processing the same pending shard set.

## 6. Workers vs max-new-shards

`--workers` and `--max-new-shards` are different:

- `--workers`: concurrency limit.
- `--max-new-shards`: total new shards selected for this invocation.

Examples:

- `--workers 3 --max-new-shards 3` -> up to 3 at once, exactly 3 selected.
- `--workers 3 --max-new-shards 6` -> up to 3 at once, 6 total selected.
- `--workers 8 --max-new-shards 1` -> one shard selected, one worker effectively used.

## 7. Metrics added in Phase 1.2d

Run output includes:

- worker count and selected shard indices/count;
- wall-clock duration;
- sum of worker active durations;
- games/moves per wall-clock second;
- per-worker peak RSS;
- max individual worker peak RSS;
- process-tree peak RSS (coordinator + live workers sampled over time).

Important memory note:

- process-tree peak RSS is not the sum of per-worker peaks after the run;
- it is sampled during execution to capture true concurrent memory pressure.

## 8. Failure and interruption behavior

- One worker failure does not corrupt completed shards.
- Coordinator commits any valid contiguous successful prefix.
- CLI exits nonzero when selected shards fail.
- Failed/incomplete shards remain pending for resume.
- Manifest writes stay atomic (temp + replace).

## 9. PowerShell commands

Run from repository root.

### 9.1 Status check (safe, read-only)

```powershell
python -m uv run python -m chesslens.ingestion.run_sharded_ingestion `
  --config configs/ingestion/2017_01_sharded.yaml --status
```

### 9.2 Parallel resume run

```powershell
python -m uv run python -m chesslens.ingestion.run_sharded_ingestion `
  --config configs/ingestion/2017_01_sharded.yaml `
  --workers 3 `
  --max-new-shards 3 `
  --resume
```

### 9.3 Verify-only after completion

```powershell
python -m uv run python -m chesslens.ingestion.run_sharded_ingestion `
  --config configs/ingestion/2017_01_sharded.yaml --verify-only
```

### 9.4 Reproducible fixture benchmark (1/2/4 workers)

```powershell
python -m uv run python scripts/benchmark_parallel_shard_ingestion.py `
  --workers 1,2,4 `
  --output reports/benchmarks/phase12d_parallel_fixture.json
```

## 10. Choosing workers on a 32 GB Windows laptop

Practical rule of thumb:

1. Start with `workers=1` to establish baseline.
2. Try `workers=2` and check process-tree peak RSS.
3. Try `workers=4` only if memory headroom remains comfortable.
4. Keep enough free RAM for OS and background tools.
5. Prefer stable runs over maximum parallelism.

If process-tree peak is too high or system becomes unstable, reduce workers.

## 11. Quick glossary for this phase

- shard: independently compressed subset of complete PGN games.
- collection: set of manifest-listed shard datasets for one shard plan + ingestion identity.
- contiguous prefix: committed shard indices with no gaps from 0.
- stale lock: leftover lock whose owning process no longer exists.
- process-tree RSS: sampled sum of coordinator RSS plus all live worker RSS.
