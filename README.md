# ChessLens

ChessLens is a production-style machine learning project for ranking chess moves and estimating blunder risk.

This repository currently implements:

- Phase 0 foundation: reproducible environment, package layout, tests, CI, and architecture decisions.
- Phase 1.1 contracts: versioned game/move/position/eval/error/manifest schemas.
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
- bounded sample profiling output at `reports/sample_profile.json`.

Deferred on purpose:

- full-scale Parquet partition writer for monthly archives;
- Airflow DAG execution;
- dbt SQL transformations beyond skeleton;
- LightGBM/PyTorch/Optuna/MLflow training workflows;
- FastAPI/ONNX/Docker/AWS serving.

## Architecture Overview

Current flow in this phase:

1. Stream compressed PGN from `data/raw/*.pgn.zst`.
2. Parse one game at a time with `python-chess`.
3. Validate legal pre-move states and derive versioned IDs.
4. Emit typed records and profile statistics.
5. Generate a tiny sanitized `.pgn.zst` fixture for deterministic tests and CI.

## Repository Layout

```text
chesslens/
├── .github/workflows/ci.yml
├── configs/ingestion/
├── data/
│   ├── fixtures/
│   ├── manifests/
│   └── raw/                  # ignored
├── dbt/                      # skeleton only in this phase
├── docs/
├── scripts/create_fixture.py
├── src/chesslens/
│   ├── domain/
│   ├── features/
│   ├── ingestion/
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
- `make test`
- `make lint`
- `make typecheck`

Windows direct equivalents:

- `python -m uv sync --dev`
- `python -m uv run python scripts/create_fixture.py --config configs/ingestion/fixture.yaml`
- `python -m uv run python -m chesslens.ingestion.sample_profiler --config configs/ingestion/smoke.yaml`
- `python -m uv run pytest -q`
- `python -m uv run ruff check .`
- `python -m uv run mypy src tests scripts`

All commands are expected to run from repository root.

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

CI runs Ruff, mypy, and all tests on checked-in fixture data only.

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
