# Phase 1 Closeout

## Purpose

This document closes out Phase 1 of ChessLens and separates what has been measured from what is still planned.

## What Phase 1 Accomplished

Phase 1 delivered a complete, reproducible data pipeline from compressed chess archives to validated warehouse outputs.

Completed outcomes:

- streaming PGN ingestion from compressed monthly archives;
- deterministic game identity and dataset identity;
- resumable, game-boundary-aware sharding;
- safe parallel shard ingestion with a single-writer collection manifest flow;
- Parquet publication for bronze datasets;
- DuckDB preflight validation and dbt transformations/tests;
- deterministic Tier-2 sample validation;
- full Tier-2 dbt acceptance for the sampled 2017-01 collection.

## How the Data Stack Fits Together (Beginner View)

Think of the stack as layers:

1. PGN is the chess-game text format.
2. zstd keeps the source files compressed to reduce storage and transfer size.
3. The ingestion code streams and parses one game at a time so memory use stays bounded.
4. Parsed records are written as Parquet, a columnar analytics file format.
5. DuckDB runs SQL checks and analytical queries locally over Parquet.
6. dbt organizes SQL models/tests into a dependency graph and verifies warehouse quality.

## Why Sharding and Resumability Were Needed

A full monthly archive is too large for a single fragile one-shot run.

Sharding and resumability provide:

- bounded work units;
- restart safety after interruptions;
- deterministic global identity across sessions;
- practical multi-session execution;
- controlled parallelism without breaking correctness.

## Measured Results (Completed)

### 2017-01 collection acceptance

- collection_id: 56c008fed930b6f9883e688af135a334931b69abcb43db78a84558a3a07512eb
- shard count: 43
- accepted games: 10,644,186
- emitted moves: 718,564,684
- portfolio_10m_satisfied: true

### Deterministic 10% Tier-2 sample

- sampling predicate: mod(hash(game_id), 10000) < 1000
- sampled games: 1,064,712
- sampled moves: 71,831,105
- full games: 10,644,186
- full moves: 718,564,684
- shard coverage: all 43 shards

### Tier-2 dbt acceptance build

- models: 8
- tests: 39
- total dbt nodes: 47
- PASS=47, WARN=0, ERROR=0, SKIP=0
- exit code: 0
- elapsed seconds: 37,120.95

Machine-readable reports:

- reports/acceptance/phase1_2_2017_collection_summary.json
- reports/acceptance/phase1_2_2017_tier2_sample_summary.json
- reports/acceptance/phase1_2b_2017_dbt_summary.json

## What the 10% Deterministic Sample Proves

The deterministic sample proves:

- the sample rule is stable and reproducible;
- sample and full counts reconcile as expected;
- all shards participate in the sample;
- warehouse/dbt logic runs successfully on a large, representative subset.

The deterministic sample does not prove:

- model quality;
- production latency or availability;
- cloud deployment readiness;
- long-term drift behavior.

## What 47/47 dbt Means

A 47/47 dbt result means every planned model/test node in that acceptance graph finished without warnings, errors, or skips.

It is strong warehouse quality evidence for this phase, but it is still data-pipeline evidence, not model-performance evidence.

## Why the Full Tier-2 Build Took a Long Time

The run includes heavy transformations and global checks over tens of millions of move rows in the sample.

Large table materializations and uniqueness/consistency tests require substantial disk and compute work, so long wall-clock time is expected even when correctness is good.

## Resume-Safe Evidence You Can Claim Now

Safe claims now:

- reproducible 10M+ chess ETL with deterministic identities;
- resumable and parallel shard ingestion with correctness guards;
- DuckDB/dbt warehouse modeling and test-driven data quality;
- deterministic sampled acceptance evidence and full 47/47 dbt success.

## Claims That Must Wait

Do not claim yet:

- baseline model superiority;
- neural-network improvements;
- calibrated production predictions;
- ONNX serving latency wins;
- cloud deployment SLOs or drift monitoring outcomes.

## Planned Work Boundary

### Measured and complete in this phase

- ingestion, sharding, collection assembly, Parquet publication, and warehouse/dbt acceptance.

### Planned next phases

- modeling datasets and leakage-aware train/validation/test splits;
- frequency, logistic, and LightGBM baselines;
- PyTorch multi-task model training;
- calibration, ablation, and error analysis;
- MLflow/Optuna workflows;
- ONNX/FastAPI serving;
- Docker and cloud deployment/monitoring.

## Recommended Next Phase and Acceptance Criteria

Recommended next phase: Phase 2 modeling foundation.

Acceptance criteria for entering Phase 3:

- frozen leakage-aware split definitions committed;
- reproducible training dataset build from dbt outputs;
- baseline models trained and evaluated with documented metrics;
- calibration and subgroup/error reporting wired into repeatable evaluation;
- metrics and artifacts tracked reproducibly.
