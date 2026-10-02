# Phase 2.1 Walkthrough: Modeling Dataset and Leakage-Aware Splits

This walkthrough explains how ChessLens Phase 2.1 turns warehouse relations into
reproducible supervised training data.

## 1. What Phase 2.1 Produces

The modeling builder publishes a versioned dataset root:

- `game_assignments`: one row per selected game with split + holdout decisions.
- `policy_examples`: one row per selected move with policy/value targets.
- `novel_position_test`: subset of test rows with unseen positions vs train.
- `leakage_audits`: overlap metrics for position leakage visibility.
- `_manifest.json` + `_SUCCESS`: reproducibility and publication proof.

Output path pattern:

- `data/modeling/datasets/<modeling_dataset_id>/...`

## 2. Inputs and Prerequisites

Phase 2.1 expects warehouse relations already built from a validated collection:

- `main.stg_games`
- `main.int_move_context`

Phase 2.1b additionally requires a warehouse provenance artifact describing the
exact DuckDB snapshot used for modeling.

- config field: `input.warehouse_provenance_path`
- CLI override: `--warehouse-provenance-path`

Why this matters (beginner version):

- `collection_id` tells you which source collection was used,
- but it does not prove which DuckDB rows are currently in `stg_games` and
  `int_move_context`.

Two DuckDB files can share the same `collection_id` and relation names but still
contain different rows (for example full warehouse vs deterministic 10% sample).
The provenance artifact closes that gap.

Run order for fixture workflow:

1. Build/verify fixture collection ingestion.
2. Run warehouse preflight.
3. Run dbt build.
4. Run modeling dataset builder.

## 3. Deterministic Sampling and Split Assignment

Sampling is deterministic at game grain using hash-mod rules:

- `selected = (hash(namespace|seed|game_id) % hash_modulus) < hash_threshold`

Temporal split is assigned once per game from `played_date`:

- train range
- validation range
- test range
- explicit missing/invalid date policy

All moves from a game inherit that game split.

## 4. Player Holdout Policy

Player-disjoint training eligibility is derived from deterministic hash rules over
`white_player_hash` and `black_player_hash`.

This supports experiments where some players are excluded from the core train
population while preserving reproducibility.

## 5. Label Construction

For each selected move row:

- `policy_target_action_index` is computed from `pre_move_fen` + `played_move_uci`.
- `value_target_wdl` is derived from side-to-move and final game result.

Validation guards ensure:

- move legality at label time,
- action index range `[0, 4671]`,
- split consistency and uniqueness constraints.

## 6. Leakage Audit and Novel Position Slice

The builder computes overlap metrics between train/validation/test on `position_id`.

It also emits `novel_position_test`, containing only test rows whose `position_id`
does not appear in train.

This provides a stricter generalization view than raw test alone.

## 7. Stage -> Validate -> Publish

Publication is safe and idempotent:

1. Write into a unique staging directory.
2. Validate warehouse provenance + relational + split invariants.
3. Write manifest and `_SUCCESS`.
4. Atomically move staging to final dataset path.

If the same effective config is rerun, the builder validates identity fields and
reuses the existing published dataset instead of duplicating work.

Identity now includes:

- collection identity,
- warehouse provenance version/hash,
- modeling config/version/sampling controls.

So changing the upstream warehouse provenance changes `modeling_dataset_id`.

## 8. Basic Commands

Fixture config:

```powershell
python -m uv run python -m chesslens.modeling.build_dataset `
  --config configs/modeling/fixture.yaml `
  --collection-root "$env:CHESSLENS_COLLECTION_ROOT" `
  --duckdb-path "$env:CHESSLENS_DUCKDB_PATH" `
  --warehouse-provenance-path "$env:CHESSLENS_WAREHOUSE_PROVENANCE_PATH" `
  --output-root data/modeling
```

2017 sample config:

```powershell
python -m uv run python -m chesslens.modeling.build_dataset `
  --config configs/modeling/2017_01_sample.yaml `
  --collection-root "$env:CHESSLENS_COLLECTION_ROOT" `
  --duckdb-path "$env:CHESSLENS_DUCKDB_PATH" `
  --warehouse-provenance-path "$env:CHESSLENS_WAREHOUSE_PROVENANCE_PATH" `
  --output-root data/modeling
```

Optional controls:

- `--max-games`
- `--max-examples`
- `--dry-run`
- `--validate-only`

Real-data safety checks before 2017 modeling commands:

1. Locate your actual DuckDB path (do not guess file names):

```powershell
Get-ChildItem data\tmp -Filter *.duckdb |
  Select-Object FullName, Length, LastWriteTime
```

2. Read-only relation inspection for required warehouse relations:

```powershell
python -m uv run python - <<'PY'
import duckdb
from pathlib import Path

duckdb_path = Path("REPLACE_WITH_FULL_DUCKDB_PATH")
connection = duckdb.connect(str(duckdb_path), read_only=True)
try:
  print(connection.execute(
    """
    SELECT table_schema, table_name, table_type
    FROM information_schema.tables
    WHERE table_schema = 'main'
    ORDER BY table_name
    """
  ).fetchall())
  print("main.stg_games:", connection.execute("SELECT COUNT(*) FROM main.stg_games").fetchone()[0])
  print(
    "main.int_move_context:",
    connection.execute("SELECT COUNT(*) FROM main.int_move_context").fetchone()[0],
  )
finally:
  connection.close()
PY
```

3. Generate warehouse provenance from the actual DuckDB snapshot:

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

## 9. Reading the Manifest Quickly

Key fields to inspect in `_manifest.json`:

- `modeling_dataset_id`
- `identity_payload_sha256`
- `upstream.collection_id`
- `upstream.warehouse_provenance_version`
- `upstream.warehouse_provenance_sha256`
- `sampling` and `player_holdout` rule params
- `sampling_stages.warehouse`
- `sampling_stages.modeling`
- `sampling_stages.cumulative_effective_rate_percent_of_full_collection`
- `counts` by split
- `leakage_audit.metrics`
- `datasets.<name>.<partition>.relative_files`

For reproducibility claims, confirm both:

- identity fields match intent,
- status is `complete` and `_SUCCESS` exists.

## 10. Direct vs Two-Stage Sampling

Do not treat the final dataset as a direct single-stage sample unless that is
actually how it was built.

Example:

- Warehouse is deterministic 10% sample of full collection.
- Modeling stage selects deterministic 10% of warehouse games.

Effective cumulative rate relative to full collection is approximately 1%, not
10%. The manifest records these stages separately so this is explicit and auditable.

## 11. Why Absolute Paths Are Not Identity

Absolute paths like `C:/.../warehouse.duckdb` are machine-local references. They
change between users and CI runners even when the data is logically identical.

Phase 2.1b identity uses canonical provenance payload hashing instead of absolute
paths, preventing accidental reuse mismatches across environments.
