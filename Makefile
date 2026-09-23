UV ?= uv

.PHONY: setup fixture inspect-sample ingest-fixture ingest-2013-sample ingest-2013-full ingest-2017-full warehouse-preflight dbt-debug dbt-compile dbt-build dbt-docs benchmark-report test lint typecheck

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

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy src tests scripts