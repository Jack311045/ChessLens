# ChessLens dbt Warehouse (Phase 1.2b)

This folder contains the analytical SQL layer for published Phase 1.2a parquet datasets.

## What This Layer Builds

Sources (registered as DuckDB views by preflight):

- `bronze_games`
- `bronze_moves`
- `bronze_ingestion_errors`
- `bronze_manifest`

Models:

- staging: `stg_games`, `stg_moves`, `stg_ingestion_errors`, `stg_manifest`
- intermediate: `int_move_context`, `int_positions`
- marts: `fct_move_events`, `mart_policy_examples`

## Environment Variables

Required:

- `CHESSLENS_DATASET_ROOT`: path to a published dataset directory containing `_manifest.json`

Optional (with defaults):

- `CHESSLENS_DUCKDB_PATH` (default `data/tmp/chesslens_warehouse.duckdb`)
- `CHESSLENS_DUCKDB_MEMORY_LIMIT` (default `4GB`)
- `CHESSLENS_DUCKDB_TEMP_DIR` (default `data/tmp/duckdb_temp`)

## Why Preflight Runs First

`python -m chesslens.warehouse.preflight` validates and then registers bronze views before dbt models run.

Validation checks:

- required `CHESSLENS_DATASET_ROOT` exists and is a directory;
- `_manifest.json` exists and `status == complete`;
- required parquet datasets and partition parts exist;
- manifest counts match physical parquet row counts.

If any check fails, preflight exits with a clear error and dbt is not run.

## Local Commands

Linux/macOS shell:

```bash
uv run python -m chesslens.warehouse.preflight
uv run dbt debug --project-dir dbt --profiles-dir dbt
uv run dbt compile --project-dir dbt --profiles-dir dbt
uv run dbt build --project-dir dbt --profiles-dir dbt
uv run dbt docs generate --project-dir dbt --profiles-dir dbt
```

PowerShell:

```powershell
python -m uv run python -m chesslens.warehouse.preflight
python -m uv run dbt debug --project-dir dbt --profiles-dir dbt
python -m uv run dbt compile --project-dir dbt --profiles-dir dbt
python -m uv run dbt build --project-dir dbt --profiles-dir dbt
python -m uv run dbt docs generate --project-dir dbt --profiles-dir dbt
```

## CI Notes

CI builds fixture ingestion output first, exports dataset and DuckDB env vars, runs preflight, then runs dbt `debug`, `compile`, and `build`.

No external database service is required.