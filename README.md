# ChessLens

ChessLens is a production-style machine learning project for ranking chess moves and estimating blunder risk.

This repository currently implements:

- Phase 0 foundation: reproducible environment, package layout, tests, CI, and architecture decisions.
- Phase 1.1 contracts: versioned game/move/position/eval/error/manifest schemas.
- Phase 1.2a ingestion: bounded-memory batch ETL to schema-controlled partitioned Parquet with safe publication, reproducibility manifests, and DuckDB validation.
- Phase 1.2b warehouse: DuckDB/dbt source registration, staging/intermediate/marts, and data quality tests for training-safe marts.
- Phase 1.2c sharding: streaming, resumable, game-boundary-aware sharding and multi-session collection ingestion with preserved global game identity.
- A bounded-memory streaming PGN reader and sample profiler to validate assumptions against real Lichess data.

This repository does not yet implement the full 10M-game ETL, model training, deployment, or performance claims.

## Project Objective

Build a verifiable end-to-end ML system with strong engineering fundamentals:

- correct data contracts and leakage controls;
- deterministic chess position/action encodings;
- testable ingestion logic that can scale later;
- reproducibility and CI discipline.

## Current Status

Implemented now:

- versioned schemas for `GameRecord`, `MoveRecord`, `PositionRecord`, `EngineEvalRecord`, `IngestionErrorRecord`, and `RunManifest`;
- `18 x 8 x 8` board encoding and fixed `8 x 8 x 73` action encoding;
- streaming `.pgn.zst` parsing with strict/tolerant behavior;
- sanitized tiny fixture generation (`20` games default);
- bounded sample profiling output at `reports/sample_profile.json`;
- batch-based Parquet output for bronze `games`, `moves`, and `ingestion_errors` datasets;
- deterministic `dataset_id` derived from source checksum and effective config;
- explicit ingestion pipeline version (`parquet_etl_v1`) included in dataset identity;
- optional source checksum verification (`expected_source_sha256`);
- staged write + validation + publish workflow for idempotent dataset publication;
- dbt models for `stg_games`, `stg_moves`, `stg_ingestion_errors`, `stg_manifest`, `int_positions`, `int_move_context`, `fct_move_events`, and `mart_policy_examples`;
- preflight validation + DuckDB bronze view registration before dbt builds;
- singular and generic dbt tests for keys, contiguity, manifest reconciliation, and leakage guards.
- optional HMAC-based player anonymization mode for real archives.
- streaming PGN-boundary sharding (`run_sharding`) with atomic `.partial` publication and a resumable shard manifest;
- multi-session sharded ingestion (`run_sharded_ingestion`) that preserves global `game_id` identity and records a collection manifest;
- collection-aware warehouse preflight that unions only manifest-listed shard datasets for dbt.

Deferred on purpose:

- Airflow DAG execution;
- LightGBM/PyTorch/Optuna/MLflow training workflows;
- FastAPI/ONNX/Docker/AWS serving.

## Architecture Overview

Current flow in this phase:

1. Stream compressed PGN from `data/raw/*.pgn.zst`.
2. Parse one game at a time with `python-chess`.
3. Validate legal pre-move states and derive versioned IDs.
4. Buffer only bounded batches of records and write partitioned Parquet parts.
5. Validate relational and schema invariants with DuckDB SQL.
6. Publish completed datasets by renaming validated staging output into deterministic dataset paths.
7. Validate published dataset roots and register DuckDB bronze views for dbt.
8. Build dbt staging/intermediate/marts and enforce warehouse quality tests.
9. Emit run manifest metrics for reproducibility and benchmarking.

Reproducibility semantics:

- `dataset_id` identifies a logical transformation identity (source checksum + effective config + supported versions + pipeline version + key ID), not guaranteed byte-for-byte file identity across environments.
- `run_id` identifies one execution attempt and is always unique.
- `ingestion_errors` include run-level provenance (`run_id`) to trace which attempt observed each rejection.

## Repository Layout

```text
chesslens/
├── .github/workflows/ci.yml
├── configs/ingestion/
├── data/
│   ├── fixtures/
│   ├── manifests/
│   └── raw/                  # ignored
├── dbt/                      # Phase 1.2b warehouse models and tests
├── docs/
├── scripts/create_fixture.py
├── src/chesslens/
│   ├── domain/
│   ├── features/
│   ├── ingestion/
│   ├── warehouse/
│   └── validation/
└── tests/
```

## Environment Setup

Python target: `3.11`.

`uv` workflow:

```bash
uv sync --dev
```

If `uv` is not on your PATH in Windows PowerShell:

```powershell
python -m uv sync --dev
```

## Commands

Make targets:

- `make setup`
- `make fixture`
- `make inspect-sample`
- `make ingest-fixture`
- `make ingest-2013-sample`
- `make warehouse-preflight`
- `make dbt-debug`
- `make dbt-compile`
- `make dbt-build`
- `make dbt-docs`
- `make benchmark-report`
- `make shard-fixture`
- `make shard-fixture-status`
- `make ingest-fixture-shards`
- `make collection-verify`
- `make shard-2017-dry-run`
- `make shard-2017`
- `make ingest-2017-shard`
- `make ingest-2017-status`
- `make ingest-2017-verify`
- `make test`
- `make lint`
- `make typecheck`

Windows direct equivalents:

- `python -m uv sync --dev`
- `python -m uv run python scripts/create_fixture.py --config configs/ingestion/fixture.yaml`
- `python -m uv run python -m chesslens.ingestion.sample_profiler --config configs/ingestion/smoke.yaml`
- `python -m uv run python -m chesslens.ingestion.run_ingestion --config configs/ingestion/fixture_etl.yaml`
- `python -m uv run python -m chesslens.ingestion.run_ingestion --config configs/ingestion/2013_01_sample.yaml`
- `python -m uv run python -m chesslens.warehouse.preflight`
- `python -m uv run dbt debug --project-dir dbt --profiles-dir dbt`
- `python -m uv run dbt compile --project-dir dbt --profiles-dir dbt`
- `python -m uv run dbt build --project-dir dbt --profiles-dir dbt`
- `python -m uv run dbt docs generate --project-dir dbt --profiles-dir dbt`
- `python -m uv run python -m chesslens.warehouse.benchmark --dataset-root <dataset-root> --output reports/benchmarks/ingestion_benchmark.json`
- `python -m uv run python -m chesslens.warehouse.benchmark --dataset-root data/processed/datasets/c7703b6c4404e13814dedd4431146c09fd6a389843a400a7ea4c4e2ec40ab4a9 --output reports/benchmarks/ingestion_2013_01.json`
- `python -m uv run pytest -q`
- `python -m uv run ruff check .`
- `python -m uv run mypy src tests scripts`

All commands are expected to run from repository root.

Real-data configs require environment variables for HMAC mode:

- `CHESSLENS_PLAYER_HMAC_KEY`
- `CHESSLENS_PLAYER_HMAC_KEY_ID`

Never commit HMAC secrets to Git.

HMAC key-ID policy:

- one key ID must map to one stable secret;
- rotating or changing the secret requires a new key ID;
- never reuse one key ID for different secrets;
- key IDs may be written to manifests, but secrets must never appear in YAML, Git, manifests, logs, or tests.

## Phase 1.2b dbt Warehouse Workflow

Required environment variable:

- `CHESSLENS_DATASET_ROOT` -> published dataset path, e.g. `data/processed/datasets/<dataset_id>`.

Optional environment variables (defaults are safe for local development):

- `CHESSLENS_DUCKDB_PATH` (default `data/tmp/chesslens_warehouse.duckdb`)
- `CHESSLENS_DUCKDB_MEMORY_LIMIT` (default `4GB`)
- `CHESSLENS_DUCKDB_TEMP_DIR` (default `data/tmp/duckdb_temp`)

Recommended sequence:

1. Run ingestion to create or reuse a published dataset.
2. Export `CHESSLENS_DATASET_ROOT`.
3. Run preflight (`python -m chesslens.warehouse.preflight`) to verify manifest/data consistency and register DuckDB bronze views.
4. Run dbt `debug`, `compile`, `build`, and optionally `docs generate`.

Preflight fails early with explicit errors when the dataset root is missing, manifest status is not complete, required parquet partitions are absent, or manifest counts disagree with physical parquet counts.

## Phase 1.2c Resumable Sharding Workflow

Large monthly archives are processed over multiple sessions by splitting them into
independent, game-boundary-aware shards and ingesting a few shards per session.

Core commands (PowerShell shown; Linux/macOS use `uv run ...` directly):

```powershell
# Inspect without processing (disk + checksum + plan preview)
python -m uv run python -m chesslens.ingestion.run_sharding `
  --config configs/sharding/2017_01.yaml --dry-run

# Create or resume raw shards
python -m uv run python -m chesslens.ingestion.run_sharding `
  --config configs/sharding/2017_01.yaml --resume

# Process one new shard this session (then you may shut down)
python -m uv run python -m chesslens.ingestion.run_sharded_ingestion `
  --config configs/ingestion/2017_01_sharded.yaml --max-new-shards 1 --resume

# Progress without processing
python -m uv run python -m chesslens.ingestion.run_sharded_ingestion `
  --config configs/ingestion/2017_01_sharded.yaml --status

# Validate the completed collection
python -m uv run python -m chesslens.ingestion.run_sharded_ingestion `
  --config configs/ingestion/2017_01_sharded.yaml --verify-only
```

Collection dbt build (unions only manifest-listed shard datasets):

```powershell
$env:CHESSLENS_COLLECTION_ROOT = "data/processed/collections/<collection_id>"
python -m uv run python -m chesslens.warehouse.preflight
python -m uv run dbt build --project-dir dbt --profiles-dir dbt
```

Identity guarantee: a game's `game_id` is derived from the parent archive SHA-256
and its global game index, so it is identical whether ingested directly from the
parent or from a shard. Raw shards live under `data/raw_shards/` (git-ignored);
collections under `data/processed/collections/` (git-ignored). Do not start a full
2017-01 run casually — it is a multi-session, multi-hour job.

## Phase 1.2a Output Layout

Completed datasets are published under `data/processed/datasets/<dataset_id>/`.

Example layout:

```text
data/processed/
	datasets/
		<dataset_id>/
			games/
				source_month=2013-01/
					part-000000.parquet
			moves/
				source_month=2013-01/
					part-000000.parquet
			ingestion_errors/
				source_month=2013-01/
					part-000000.parquet
			_manifest.json
```

During execution, data is written first to `data/processed/staging/<dataset_id>__<run_id>/`.
Only validated datasets with completed manifests are published into `datasets/`.

Preflight config guards:

- if `source_month` is configured and the archive filename includes `YYYY-MM`, they must match;
- source months must be real calendar months;
- schema/encoding version fields must match currently supported code constants.

Design note:

- This phase intentionally publishes only bronze `games`, `moves`, and `ingestion_errors`.
- A globally deduplicated positions table is deferred to Phase 1.2b dbt SQL so it is truly global, not only game-local or batch-local.

Timing metrics note:

- manifests store `checksum_duration_seconds`, `processing_and_validation_duration_seconds`, and `total_duration_seconds` separately;
- throughput metrics are reported with explicit `processing_*` and `total_*` names to avoid ambiguity.

## Fixture Generation

Default fixture config selects the first 20 complete valid games from the source archive, replaces player names and site URLs with deterministic placeholders, then writes:

- `data/fixtures/lichess_2013_01_first20.pgn`
- `data/fixtures/lichess_2013_01_first20.pgn.zst`
- `data/manifests/fixture_2013_01_first20.json`

The full raw archive is excluded from Git.

## Sample Inspection

Run bounded profiling (default 100 games):

```bash
make inspect-sample
```

or

```powershell
python -m uv run python -m chesslens.ingestion.sample_profiler --config configs/ingestion/smoke.yaml
```

Output path: `reports/sample_profile.json`.

This report is a functional validation sample, not a statistical population study.

## Testing, Linting, and Type Checking

- Unit tests: FEN normalization, IDs, schemas, config, encodings.
- Property tests: randomized legal positions for encode/decode and determinism invariants.
- Integration tests: streaming fixture replay, schema conformance, Arrow/Parquet round-trips.

CI runs Ruff, mypy, and all Python tests first, then builds a fixture ingestion dataset and executes warehouse preflight + dbt `debug`/`compile`/`build`, then builds a fixture shard collection and runs the collection dbt build. CI never depends on local raw archives.

## Data Source and Licensing

Data source: Lichess Open Database.

Lichess database exports are published under CC0. Verify current terms at the official source before redistribution.

## Data Privacy Policy

- Raw source archives are never committed.
- Derived fixture output replaces player names and identifying site URLs.
- Schema includes anonymized player identifiers only.
- Production-grade hashing should use externally provided keyed HMAC secrets that are never committed.

## Limitations

- Current ingestion focuses on schema/identity correctness and bounded-memory streaming.
- No full monthly ETL partitions yet.
- No training metrics, serving latency, or cloud deployment metrics yet.

## Roadmap (Next Phases)

1. Phase 1.2: scalable ETL writing Parquet partitions and run manifests.
2. Phase 2: baseline models and leakage-aware evaluation framework.
3. Phase 3: PyTorch multi-task network, experiment tracking, and model selection.
4. Phase 4: API serving, ONNX optimization, Docker, CI/CD deployment.

## Core References

- Blueprint copy: `docs/ChessLens_Project_Blueprint.md`
- Data contracts: `docs/data-contract.md`
- Encoding specification: `docs/encoding-specification.md`
- Architecture decisions: `docs/architecture-decisions.md`
- Learning walkthrough: `docs/learning/phase0_and_1_1_walkthrough.md`
- Phase 1.2a walkthrough: `docs/learning/phase1_2a_parquet_etl_walkthrough.md`
- Phase 1.2b walkthrough: `docs/learning/phase1_2b_dbt_sql_walkthrough.md`
- Phase 1.2c walkthrough: `docs/learning/phase1_2c_resumable_sharding_walkthrough.md`