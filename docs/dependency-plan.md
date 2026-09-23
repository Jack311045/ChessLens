# Dependency Plan (Phase 0 and 1.1)

## Python Version

Target version: `Python 3.11`.

Why:

- broad compatibility for data and ML ecosystem packages;
- stable baseline for CI and cross-platform reproduction;
- aligns with near-term model and ETL tooling plans.

## Runtime Dependencies

- `python-chess`: PGN parsing, legal move validation, board reconstruction.
- `zstandard`: streaming decompression of `.pgn.zst` without full extraction, and streaming compression of independent Phase 1.2c shards.
- `numpy`: deterministic tensor representation for position/action encodings.
- `polars`: lightweight, future-proof analytics and schema checks.
- `pyarrow`: explicit analytical schemas and Parquet interoperability.
- `duckdb`: local analytical backend compatibility for next ETL stage.
- `psutil`: cross-platform process memory sampling for measured peak RSS ingestion metrics.
- `PyYAML`: externalized ingestion config loading.
- `dbt-core` (pinned): dbt CLI and SQL DAG execution for Phase 1.2b warehouse transformations.
- `dbt-duckdb` (pinned): dbt adapter that materializes models in DuckDB and reads local views.

Phase 1.2c (resumable sharding + collection ingestion) added **no new dependencies**;
it reuses the existing `zstandard`, `pyarrow`, `duckdb`, `psutil`, and `PyYAML` stack.

## Development Dependencies

- `pytest`: unit + integration test runner.
- `hypothesis`: property-based tests over randomized legal positions.
- `ruff`: linting and formatting checks.
- `mypy`: static typing checks for correctness and maintainability.
- `pre-commit`: local quality gates before commits.
- `types-PyYAML`: type stubs for config parsing paths.

## Dependency Groups

Defined in `pyproject.toml`:

- default `project.dependencies`: packages needed to run library and scripts.
- `dev` dependency group: quality and testing tooling.

## pyproject.toml vs uv.lock vs .venv vs requirements.txt

`pyproject.toml`:

- project metadata;
- high-level dependency constraints;
- tooling configuration (ruff, mypy, pytest).

`uv.lock`:

- exact resolved dependency graph (fully pinned);
- platform-aware lock details used for deterministic installs.

`.venv`:

- local virtual environment directory;
- machine-specific and intentionally ignored in Git.

`requirements.txt`:

- optional export format for platforms that cannot read `pyproject.toml`/`uv.lock` directly.
- not the source of truth in this repository.

## Why Commit uv.lock

- Guarantees that teammates and CI resolve the same versions.
- Prevents accidental drift from unconstrained transitive upgrades.
- Makes debugging reproducible across machines.

## Why Ignore .venv

- Contains machine-specific binaries and paths.
- Not portable across OS and Python patch versions.
- Increases repository size without adding source value.

## Command Semantics

`uv sync --dev`:

- creates/updates local environment;
- installs exactly the locked dependencies including development group.

`uv run pytest`:

- executes tests using the managed environment.

`uv run ruff check .`:

- runs lint checks against repository code.

`uv run dbt build --project-dir dbt --profiles-dir dbt`:

- executes Phase 1.2b staging/intermediate/mart models and tests in DuckDB.

`uv run python -m chesslens.warehouse.preflight`:

- validates published dataset completeness before dbt runs;
- registers `bronze_*` DuckDB views over parquet and manifest inputs.

If `uv` is not on PATH in Windows:

- use `python -m uv ...` equivalents.

## Exporting requirements.txt Later

If an external platform requires a requirements file:

```bash
uv export --format requirements-txt --no-hashes -o requirements.txt
```

This keeps `pyproject.toml` and `uv.lock` as source of truth while producing compatibility output.