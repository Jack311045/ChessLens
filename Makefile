UV ?= uv

.PHONY: setup fixture inspect-sample test lint typecheck

setup:
	$(UV) sync --dev

fixture:
	$(UV) run python scripts/create_fixture.py --config configs/ingestion/fixture.yaml

inspect-sample:
	$(UV) run python -m chesslens.ingestion.sample_profiler --config configs/ingestion/smoke.yaml

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy src tests scripts