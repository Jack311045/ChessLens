# dbt Skeleton for ChessLens

This directory is intentionally lightweight in Phase 0/1.1.

Planned ownership in the next ETL PR:

- `models/`: staging, intermediate, and training marts built on real ingestion outputs.
- `tests/`: SQL data tests for schema constraints and leakage guards.
- `seeds/`: small static lookup data when needed.

Substantive dbt implementation is deferred until parquet outputs from ingestion are stable.