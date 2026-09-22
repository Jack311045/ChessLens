# Phase 1.2a Parquet ETL Walkthrough

Audience: a developer who understands basic Python and wants to understand why this ingestion design is production-style.

## 1. What Phase 1.2a Adds

Phase 1.2a takes the existing streaming parser and adds a full ingestion command that:

- streams `.pgn.zst` archives game-by-game;
- keeps memory bounded with configurable batch buffers;
- writes partitioned Parquet datasets (`games`, `moves`, `ingestion_errors`);
- validates outputs with DuckDB SQL;
- writes reproducibility manifests;
- publishes only complete validated datasets.

## 2. Why Arrow and Parquet Are Both Used

Arrow:

- in-memory, columnar table format;
- used while Python is actively processing and writing batches;
- gives explicit schemas (`field names + types + nullability`).

Parquet:

- persistent on-disk columnar format;
- compressed and efficient for analytics tools (DuckDB, dbt, Spark, etc.);
- stores columns in a way that supports selective reads and partition pruning.

Short version:

- Arrow is for in-memory structure.
- Parquet is for durable storage and analytics.

## 3. Why Batches Bound Memory

Without batching, buffers can grow forever on large archives.

Phase 1.2a uses two limits:

- `batch_games`: flush after this many accepted games.
- `max_buffered_records`: flush when total buffered rows gets large (`games + moves + errors`).

Important nuance:

- flushing happens only after finishing a game;
- this preserves consistency so one game row and all of its move rows stay together in the same flush;
- one very long game may exceed the record threshold by that one game, which is expected and documented.

## 4. Batch Size vs Parquet Row-Group Size

These are different knobs:

- batch size (`batch_games` / `max_buffered_records`): memory and flush cadence in Python;
- row-group size (`parquet_row_group_size`): internal layout inside each Parquet file.

A simple mental model:

- batch size controls when a new file part is produced;
- row-group size controls how each file is internally chunked for scan performance.

## 5. Why Partitioning Helps DuckDB and dbt

Outputs are written under `source_month=YYYY-MM` partitions.

That helps because:

- DuckDB can filter partitions quickly when month is constrained;
- dbt source models can treat partitioned Parquet as stable, queryable inputs;
- month-level lineage is explicit and auditable.

## 6. Why We Do Not Publish `positions` Yet

`positions` must represent globally deduplicated board identities.

Batch-local or game-local deduplication is not global deduplication.

So in Phase 1.2a:

- we publish only bronze `games`, `moves`, `ingestion_errors`;
- Phase 1.2b will build globally deduplicated `int_positions` from `moves` via SQL.

## 7. `dataset_id` vs `run_id`

`dataset_id`:

- deterministic identity of the published dataset contents/layout;
- based on source checksum + canonical effective config + schema/version parameters + non-secret key ID.

`run_id`:

- unique execution attempt identifier (time/UUID based);
- different on every run, even if dataset contents are identical.

Meaning:

- many runs can point to one dataset_id;
- this is how idempotent reuse works.

## 8. Staging and Publication Safety

The runner does not write directly into final dataset paths.

It does:

1. write to `data/processed/staging/<dataset_id>__<run_id>/`;
2. close and validate Parquet outputs;
3. write complete manifest;
4. rename staging dataset to `data/processed/datasets/<dataset_id>/`.

Why this matters:

- partial failures remain in staging;
- downstream readers never mistake partial output for complete data;
- existing complete datasets are never silently overwritten.

## 9. What Idempotency Means Here

Idempotency means: running the same ingestion config + same source bytes repeatedly should not duplicate rows.

In this implementation:

- deterministic dataset_id identifies the target dataset;
- if an existing dataset_id is already complete and validates, the runner reuses it;
- no append/duplicate write is performed.

## 10. Why SHA-256 Source Verification Is Required

`expected_source_sha256` protects source identity.

For official monthly configs:

- compute SHA-256 once before parsing;
- compare to expected value;
- abort early on mismatch.

This catches accidental file mixups or corruption before expensive ETL work.

## 11. Why HMAC Protects Player Identities

Unkeyed hashes are deterministic but weaker from a privacy perspective.

For real archives, use `player_hash_mode: hmac_sha256`:

- secret comes from env var (never committed);
- manifest includes only non-secret key ID;
- same key gives stable pseudonyms, different key changes them.

This balances privacy and reproducibility.

## 12. DuckDB Validation Queries: What They Check

Before publication, DuckDB checks at least:

- `games` row count equals accepted games;
- `moves` row count equals emitted moves;
- `game_id` uniqueness in games;
- `(game_id, ply)` uniqueness in moves;
- move foreign-key integrity to games;
- contiguous zero-based ply sequence per game;
- `games.ply_count` matches counted move rows;
- Parquet schemas match Arrow contracts;
- partition month in paths matches row values.

Any non-zero bad-row check fails publication.

## 13. Strict vs Tolerant Mode

Tolerant mode (`strict: false`):

- recoverable per-game errors are recorded in `ingestion_errors`;
- ingestion continues.

Strict mode (`strict: true`):

- first recoverable per-game rejection raises;
- final dataset is not published.

Fatal errors (checksum mismatch, truncated Zstandard stream) are always fatal in either mode.

## 14. How to Run Ingestion Configs

Fixture run:

```powershell
python -m uv run python -m chesslens.ingestion.run_ingestion --config configs/ingestion/fixture_etl.yaml
```

2013 sample run (100 raw games):

```powershell
$env:CHESSLENS_PLAYER_HMAC_KEY = "<set-secret>"
$env:CHESSLENS_PLAYER_HMAC_KEY_ID = "dev-key-1"
python -m uv run python -m chesslens.ingestion.run_ingestion --config configs/ingestion/2013_01_sample.yaml
```

2013 full run:

```powershell
$env:CHESSLENS_PLAYER_HMAC_KEY = "<set-secret>"
$env:CHESSLENS_PLAYER_HMAC_KEY_ID = "prod-key-2026-09"
python -m uv run python -m chesslens.ingestion.run_ingestion --config configs/ingestion/2013_01_full.yaml
```

Note:

- CLI `--max-games` accepts integers only.
- Use dedicated config files with `max_games: null` for EOF/full runs.

2017 sample (100 raw games):

```powershell
$env:CHESSLENS_PLAYER_HMAC_KEY = "<set-secret>"
$env:CHESSLENS_PLAYER_HMAC_KEY_ID = "dev-key-1"
python -m uv run python -m chesslens.ingestion.run_ingestion --config configs/ingestion/2017_01_full.yaml --max-games 100
```

## 15. Which Generated Files Are Ignored by Git

Generated ETL outputs are intentionally ignored:

- `data/processed/` (includes `datasets/` and `staging/`)
- large Parquet artifacts (`*.parquet`)
- raw archives in `data/raw/`

This keeps repository history small, reviewable, and CI-safe.
