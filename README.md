# ChessLens

ChessLens is a production-style machine learning project for ranking chess moves and estimating blunder risk.

This repository currently implements:

- Phase 0 foundation: reproducible environment, package layout, tests, CI, and architecture decisions.
- Phase 1.1 contracts: versioned game/move/position/eval/error/manifest schemas.
- Phase 1.2a ingestion: bounded-memory batch ETL to schema-controlled partitioned Parquet with safe publication, reproducibility manifests, and DuckDB validation.
- Phase 1.2b warehouse: DuckDB/dbt source registration, staging/intermediate/marts, and data quality tests for training-safe marts.
- Phase 1.2c sharding: streaming, resumable, game-boundary-aware sharding and multi-session collection ingestion with preserved global game identity.
- Phase 1.2d parallel ingestion: safe shard-level multiprocessing with a single-writer collection lock, in-order manifest commits, and process-tree memory telemetry.
- Phase 2.1 modeling datasets: deterministic game sampling, leakage-aware temporal splits, player-holdout policy, policy/value labels, and staged idempotent publication.
- Phase 2.1b provenance hardening: typed warehouse provenance identity, DuckDB snapshot validation, and two-stage sampling manifest semantics.
- A bounded-memory streaming PGN reader and sample profiler to validate assumptions against real Lichess data.

Phase 1 ETL and warehouse acceptance are complete, including 10M+ real-game processing and deterministic Tier-2 dbt validation.

## Project Objective

Build a verifiable end-to-end ML system with strong engineering fundamentals:

- correct data contracts and leakage controls;
- deterministic chess position/action encodings;
- testable ingestion logic that can scale later;
- reproducibility and CI discipline.

## Current Status

Completed:

- repository and dependency foundation;
- schemas and chess encodings;
- streaming PGN ingestion;
- resumable and parallel shard processing;
- 10M+ game Parquet ETL (2017-01 collection acceptance complete);
- DuckDB/dbt transformations and warehouse-style tests;
- deterministic 10% sample validation using `mod(hash(game_id), 10000) < 1000`;
- Phase 2.1 leakage-aware modeling dataset foundation with deterministic reuse checks;
- Phase 2.1b upstream warehouse provenance hardening for safe modeling identity/reuse;
- Phase 1 acceptance evidence in `reports/acceptance/`.

Not yet completed:

- frequency and logistic baselines;
- LightGBM ranking baseline;
- custom PyTorch multi-task ResNet;
- calibration, ablations, and error analysis;
- MLflow and Optuna experiment workflows;
- ONNX export and FastAPI serving;
- Docker production image;
- AWS deployment, monitoring, and drift reporting.

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
9. Build leakage-aware modeling datasets from `stg_games` + `int_move_context` using deterministic policy contracts.
10. Emit versioned manifests and leakage audit metrics for reproducibility and benchmarking.

Reproducibility semantics:

- `dataset_id` identifies a logical transformation identity (source checksum + effective config + supported versions + pipeline version + key ID), not guaranteed byte-for-byte file identity across environments.
- `run_id` identifies one execution attempt and is always unique.
- `ingestion_errors` include run-level provenance (`run_id`) to trace which attempt observed each rejection.

## Repository Layout

```text
chesslens/
├── .github/workflows/ci.yml
├── configs/ingestion/
├── configs/modeling/
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
│   ├── modeling/
│   ├── warehouse/
│   └── validation/
└── tests/
```

## Environment Setup

Python target: `3.11`.

`uv` workflow:

```bash
python -m uv sync --frozen --dev
```

Dependency reproducibility notes:

- `pyproject.toml` defines project metadata and allowed dependency requirements.
- `uv.lock` pins the exact resolved dependency graph.
- `--frozen` prevents unnoticed lockfile updates during setup.

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
- `make benchmark-shard-parallel-fixture`
- `make modeling-fixture`
- `make modeling-2017-sample`
- `make test`
- `make lint`
- `make typecheck`

Windows direct equivalents:

- `python -m uv sync --frozen --dev`
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
- `python -m uv run python -m chesslens.modeling.build_dataset --config configs/modeling/fixture.yaml --collection-root <collection-root> --duckdb-path <duckdb-path> --warehouse-provenance-path <warehouse-provenance-path> --output-root data/modeling`
- `python -m uv run python -m chesslens.modeling.generate_warehouse_provenance --collection-root <collection-root> --duckdb-path <duckdb-path> --games-relation main.stg_games --move-context-relation main.int_move_context --warehouse-kind <full|deterministic_sample|fixture> --output <output-json-path>`
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

Do not set an explicit DuckDB temp-directory override in normal usage. The explicit profile temp-directory path was removed after reconnect/runtime issues.

Recommended sequence:

1. Run ingestion to create or reuse a published dataset.
2. Export `CHESSLENS_DATASET_ROOT`.
3. Run preflight (`python -m chesslens.warehouse.preflight`) to verify manifest/data consistency and register DuckDB bronze views.
4. Run dbt `debug`, `compile`, `build`, and optionally `docs generate`.

Preflight fails early with explicit errors when the dataset root is missing, manifest status is not complete, required parquet partitions are absent, or manifest counts disagree with physical parquet counts.

## Phase 1.2c/1.2d Resumable + Parallel Sharding Workflow

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

# Process one new shard this session (sequential default: workers=1)
python -m uv run python -m chesslens.ingestion.run_sharded_ingestion `
	--config configs/ingestion/2017_01_sharded.yaml --max-new-shards 1 --resume

# Process three pending shards with up to three concurrent workers
python -m uv run python -m chesslens.ingestion.run_sharded_ingestion `
	--config configs/ingestion/2017_01_sharded.yaml --workers 3 --max-new-shards 3 --resume

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

Fixture benchmark command (workers 1/2/4, no real-data ingestion):

```powershell
python -m uv run python scripts/benchmark_parallel_shard_ingestion.py `
	--workers 1,2,4 --output reports/benchmarks/phase12d_parallel_fixture.json
```

## Phase 2.1 Modeling Dataset Workflow

Build a leakage-aware modeling dataset from warehouse relations:

```powershell
python -m uv run python -m chesslens.modeling.build_dataset `
	--config configs/modeling/fixture.yaml `
	--collection-root "$env:CHESSLENS_COLLECTION_ROOT" `
	--duckdb-path "$env:CHESSLENS_DUCKDB_PATH" `
	--warehouse-provenance-path "$env:CHESSLENS_WAREHOUSE_PROVENANCE_PATH" `
	--output-root data/modeling
```

Warehouse provenance is now mandatory. It is a typed artifact describing the exact
DuckDB snapshot identity (kind, declared row counts, transformation identity,
canonical SHA-256).

Beginner explanation:

- `collection_id` answers "which source collection?"
- warehouse provenance answers "which DuckDB snapshot rows?"

Those are different questions. Two DuckDB files can point at the same
`collection_id` while containing different rows (for example full warehouse versus
deterministic Tier-2 sample).

Real 2017 Tier-2 sample config uses:

- `configs/modeling/2017_01_sample.yaml`
- generated provenance output path: `reports/local/phase2_1b_2017_tier2_warehouse_provenance.json`

Do not treat any template as acceptance evidence. Generate provenance from your
actual DuckDB snapshot first.

Use existing Tier-2 acceptance evidence as sampling input:

- `reports/acceptance/phase1_2_2017_tier2_sample_summary.json`
- `reports/acceptance/phase1_2b_2017_dbt_summary.json`

Find your real DuckDB path:

```powershell
Get-ChildItem data\tmp -Filter *.duckdb |
	Select-Object FullName, Length, LastWriteTime
```

Read-only relation inspection:

```powershell
python -m uv run python - <<'PY'
import duckdb
from pathlib import Path

duckdb_path = Path("REPLACE_WITH_FULL_DUCKDB_PATH")
connection = duckdb.connect(str(duckdb_path), read_only=True)
try:
	rows = connection.execute(
		"""
		SELECT table_schema, table_name, table_type
		FROM information_schema.tables
		WHERE table_schema = 'main'
		ORDER BY table_name
		"""
	).fetchall()
	print("main schema relations:")
	for row in rows:
		print(row)

	for relation in ("main.stg_games", "main.int_move_context"):
		count = connection.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0]
		print(f"{relation}: {count}")
finally:
	connection.close()
PY
```

Generate real warehouse provenance artifact:

```powershell
python -m uv run python -m chesslens.modeling.generate_warehouse_provenance `
	--collection-root data/processed/collections/56c008fed930b6f9883e688af135a334931b69abcb43db78a84558a3a07512eb `
	--duckdb-path "REPLACE_WITH_FULL_DUCKDB_PATH" `
	--games-relation main.stg_games `
	--move-context-relation main.int_move_context `
	--warehouse-kind deterministic_sample `
	--sampling-evidence-path reports/acceptance/phase1_2_2017_tier2_sample_summary.json `
	--output reports/local/phase2_1b_2017_tier2_warehouse_provenance.json
```

Use the generated provenance in modeling commands:

```powershell
$env:CHESSLENS_COLLECTION_ROOT = "data/processed/collections/56c008fed930b6f9883e688af135a334931b69abcb43db78a84558a3a07512eb"
$env:CHESSLENS_DUCKDB_PATH = "REPLACE_WITH_FULL_DUCKDB_PATH"
$env:CHESSLENS_WAREHOUSE_PROVENANCE_PATH = "reports/local/phase2_1b_2017_tier2_warehouse_provenance.json"

python -m uv run python -m chesslens.modeling.build_dataset `
	--config configs/modeling/2017_01_sample.yaml `
	--collection-root "$env:CHESSLENS_COLLECTION_ROOT" `
	--duckdb-path "$env:CHESSLENS_DUCKDB_PATH" `
	--warehouse-provenance-path "$env:CHESSLENS_WAREHOUSE_PROVENANCE_PATH" `
	--output-root data/modeling `
	--dry-run

python -m uv run python -m chesslens.modeling.build_dataset `
	--config configs/modeling/2017_01_sample.yaml `
	--collection-root "$env:CHESSLENS_COLLECTION_ROOT" `
	--duckdb-path "$env:CHESSLENS_DUCKDB_PATH" `
	--warehouse-provenance-path "$env:CHESSLENS_WAREHOUSE_PROVENANCE_PATH" `
	--output-root data/modeling `
	--validate-only
```

Published modeling outputs are written under:

- `data/modeling/datasets/<modeling_dataset_id>/game_assignments/temporal_split=.../part-*.parquet`
- `data/modeling/datasets/<modeling_dataset_id>/policy_examples/temporal_split=.../part-*.parquet`
- `data/modeling/datasets/<modeling_dataset_id>/leakage_audits/temporal_position_overlap.json`
- `data/modeling/datasets/<modeling_dataset_id>/_manifest.json`

Re-running the exact same effective configuration validates and reuses the existing
published dataset rather than writing duplicates.

Phase 2.1b also records sampling in two explicit stages:

- warehouse stage (for example deterministic 10% Tier-2 warehouse),
- modeling stage (for example deterministic 10% modeling selection).

So a 10% warehouse plus 10% modeling selection is approximately a 1% cumulative
sample of the full collection, not a direct 10% sample.

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

CI runs Ruff, mypy, and all Python tests first, then builds a fixture ingestion
dataset and executes warehouse preflight + dbt `debug`/`compile`/`build`, then
builds a fixture shard collection and runs the collection dbt build, then executes
the Phase 2.1 fixture modeling builder twice to verify deterministic idempotent
reuse. CI never depends on local raw archives.

## Data Source and Licensing

Data source: Lichess Open Database.

Lichess database exports are published under CC0. Verify current terms at the official source before redistribution.

## Data Privacy Policy

- Raw source archives are never committed.
- Derived fixture output replaces player names and identifying site URLs.
- Schema includes anonymized player identifiers only.
- Production-grade hashing should use externally provided keyed HMAC secrets that are never committed.

## Limitations

- Phase 1 ETL/warehouse and Phase 2.1 dataset foundations are complete, but model
	training and deployment work are still pending.
- No training metrics, serving latency, or cloud deployment metrics are claimed yet.

## Roadmap (Next Phases)

1. Phase 2: baseline models on top of the completed leakage-aware modeling datasets.
2. Phase 3: multi-task PyTorch training, ablations, calibration, and error analysis.
3. Phase 4: model export, serving API, Docker packaging, and CI/CD hardening.
4. Phase 5: cloud deployment, monitoring, and drift reporting.

## Core References

- Blueprint copy: `docs/ChessLens_Project_Blueprint.md`
- Phase 1 closeout guide: `docs/phase1_closeout.md`
- Phase 1 collection acceptance report: `reports/acceptance/phase1_2_2017_collection_summary.json`
- Phase 1 deterministic sample report: `reports/acceptance/phase1_2_2017_tier2_sample_summary.json`
- Phase 1 dbt acceptance report: `reports/acceptance/phase1_2b_2017_dbt_summary.json`
- Data contracts: `docs/data-contract.md`
- Encoding specification: `docs/encoding-specification.md`
- Architecture decisions: `docs/architecture-decisions.md`
- Learning walkthrough: `docs/learning/phase0_and_1_1_walkthrough.md`
- Phase 1.2a walkthrough: `docs/learning/phase1_2a_parquet_etl_walkthrough.md`
- Phase 1.2b walkthrough: `docs/learning/phase1_2b_dbt_sql_walkthrough.md`
- Phase 1.2c walkthrough: `docs/learning/phase1_2c_resumable_sharding_walkthrough.md`
- Phase 1.2d walkthrough: `docs/learning/phase1_2d_parallel_shard_ingestion_walkthrough.md`
- Phase 2.1 walkthrough: `docs/learning/phase2_1_modeling_dataset_and_splits_walkthrough.md`