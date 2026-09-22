UV ?= uv

.PHONY: setup fixture inspect-sample ingest-fixture ingest-2013-sample ingest-2013-full ingest-2017-full test lint typecheck

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

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy src tests scripts