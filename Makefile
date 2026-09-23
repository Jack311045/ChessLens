UV ?= uv

.PHONY: setup fixture inspect-sample ingest-fixture ingest-2013-sample ingest-2013-full ingest-2017-full warehouse-preflight dbt-debug dbt-compile dbt-build dbt-docs benchmark-report test lint typecheck shard-fixture shard-fixture-status ingest-fixture-shards collection-verify shard-2013-benchmark shard-2017-dry-run shard-2017 ingest-2017-shard ingest-2017-status ingest-2017-verify

setup:
	$(UV) sync --dev

fixture:
	$(UV) run python scripts/create_fixture.py --config configs/ingestion/fixture.yaml

inspect-sample:
	$(UV) run python -m chesslens.ingestion.sample_profiler --config configs/ingestion/smoke.yaml

ingest-fixture:
	$(UV) run python -m chesslens.ingestion.run_ingestion --config configs/ingestion/fixture_etl.yaml

ingest-2013-sample:
	$(UV) run python -m chesslens.ingestion.run_ingestion --config configs/ingestion/2013_01_sample.yaml

ingest-2013-full:
	$(UV) run python -m chesslens.ingestion.run_ingestion --config configs/ingestion/2013_01_full.yaml

ingest-2017-full:
	$(UV) run python -m chesslens.ingestion.run_ingestion --config configs/ingestion/2017_01_full.yaml

warehouse-preflight:
	$(UV) run python -m chesslens.warehouse.preflight

dbt-debug: warehouse-preflight
	$(UV) run dbt debug --project-dir dbt --profiles-dir dbt

dbt-compile: warehouse-preflight
	$(UV) run dbt compile --project-dir dbt --profiles-dir dbt

dbt-build: warehouse-preflight
	$(UV) run dbt build --project-dir dbt --profiles-dir dbt

dbt-docs: warehouse-preflight
	$(UV) run dbt docs generate --project-dir dbt --profiles-dir dbt

benchmark-report:
	$(UV) run python -m chesslens.warehouse.benchmark --dataset-root "$$CHESSLENS_DATASET_ROOT" --output reports/benchmarks/ingestion_benchmark.json

# --- Phase 1.2c: resumable, game-boundary-aware sharding ---

shard-fixture:
	$(UV) run python -m chesslens.ingestion.run_sharding --config configs/sharding/fixture.yaml --resume

shard-fixture-status:
	$(UV) run python -m chesslens.ingestion.run_sharding --config configs/sharding/fixture.yaml --status

ingest-fixture-shards:
	$(UV) run python -m chesslens.ingestion.run_sharded_ingestion --config configs/ingestion/fixture_sharded.yaml --resume

collection-verify:
	$(UV) run python -m chesslens.ingestion.run_sharded_ingestion --config configs/ingestion/fixture_sharded.yaml --verify-only

shard-2013-benchmark:
	$(UV) run python -m chesslens.ingestion.run_sharding --config configs/sharding/2013_01.yaml --resume

shard-2017-dry-run:
	$(UV) run python -m chesslens.ingestion.run_sharding --config configs/sharding/2017_01.yaml --dry-run

shard-2017:
	$(UV) run python -m chesslens.ingestion.run_sharding --config configs/sharding/2017_01.yaml --resume

ingest-2017-shard:
	$(UV) run python -m chesslens.ingestion.run_sharded_ingestion --config configs/ingestion/2017_01_sharded.yaml --max-new-shards 1 --resume

ingest-2017-status:
	$(UV) run python -m chesslens.ingestion.run_sharded_ingestion --config configs/ingestion/2017_01_sharded.yaml --status

ingest-2017-verify:
	$(UV) run python -m chesslens.ingestion.run_sharded_ingestion --config configs/ingestion/2017_01_sharded.yaml --verify-only

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy src tests scripts