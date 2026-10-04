# Data Contract (Phase 1.2a and 1.2b)

Schema version: `1.1.0`.

This document defines logical contracts and semantics for ingestion and the Phase 1.2b warehouse layer.

Phase 1.2a publishes bronze Parquet datasets for:

- `games`
- `moves`
- `ingestion_errors`

with Hive-style partitioning by `source_month`.

Phase 1.2b consumes those published parquet outputs and materializes:

- staging: `stg_games`, `stg_moves`, `stg_ingestion_errors`, `stg_manifest`;
- intermediate: `int_positions`, `int_move_context`;
- marts: `fct_move_events`, `mart_policy_examples`.

`PositionRecord` remains a logical contract, but this phase intentionally does not publish a physical bronze `positions` table. Global deduplication of positions is deferred to Phase 1.2b dbt SQL (`int_positions`) so deduplication is global, not only game-local or batch-local.

Reproducibility semantics:

- `dataset_id` identifies a logical transformation identity, not guaranteed byte-for-byte file identity.
- `run_id` identifies a single execution attempt.
- Repeated runs may reuse one completed `dataset_id` while producing distinct `run_id` values.

## Contract Principles

- Every model has explicit grain.
- Primary and foreign keys are named and stable.
- Nullability reflects source reality.
- Leakage-prone fields are labeled and constrained.
- Versioned normalization and encoding identifiers are persisted.

## GameRecord

Grain:

- one row per parsed source game.

Primary key:

- `game_id`.

Foreign keys:

- none at this stage.

Schema version field:

- `schema_version`.

| Field | Type | Null | Source | Meaning | Validation | Role |
| --- | --- | --- | --- | --- | --- | --- |
| game_id | string | no | derived | stable game identifier from source checksum + ordinal | SHA-256 hex, deterministic | metadata |
| source_archive | string | no | runtime | archive filename | non-empty | metadata |
| source_archive_sha256 | string | no | runtime | archive integrity identity | 64-char hex | metadata |
| source_month | string | yes | filename parse | month extracted from archive name | `YYYY-MM` if present | metadata |
| source_game_index | int64 | no | stream ordinal | zero-based game position in archive | >= 0 | metadata |
| event | string | yes | PGN header | event descriptor | none | metadata |
| played_date | string | yes | PGN header | date header text | none | metadata |
| round | string | yes | PGN header | round header text | none | metadata |
| result | string | yes | PGN header | game result (`1-0`,`0-1`,`1/2-1/2`,`*`) | if present must be valid token | label/metadata |
| white_rating | int32 | yes | PGN header | white Elo | if present plausible integer | feature/metadata |
| black_rating | int32 | yes | PGN header | black Elo | if present plausible integer | feature/metadata |
| time_control_raw | string | yes | PGN header | original time control string | no parse required in v1 | feature/metadata |
| eco | string | yes | PGN header | ECO code | optional | metadata (potential leakage in early plies) |
| opening | string | yes | PGN header | opening text label | optional | metadata (potential leakage in early plies) |
| termination | string | yes | PGN header | game termination reason | optional | prohibited feature (post-outcome) |
| white_player_hash | string | yes | derived | anonymized white player ID | deterministic hash; production should be keyed HMAC | metadata/group holdout |
| black_player_hash | string | yes | derived | anonymized black player ID | deterministic hash; production should be keyed HMAC | metadata/group holdout |
| ply_count | int32 | no | derived | total mainline plies parsed | > 0 for accepted games | metadata/label support |
| schema_version | string | no | constant | contract version | must equal active schema | metadata |

## MoveRecord

Grain:

- one row per played ply (zero-based, pre-move state).

Primary key:

- composite (`game_id`, `ply`).

Foreign keys:

- `game_id` -> `GameRecord.game_id`.
- `position_id` -> `PositionRecord.position_id`.

| Field | Type | Null | Source | Meaning | Validation | Role |
| --- | --- | --- | --- | --- | --- | --- |
| game_id | string | no | derived | parent game ID | exists in GameRecord | metadata/fk |
| ply | int32 | no | enumeration | zero-based ply index | >= 0, contiguous per game | metadata/key |
| position_id | string | no | derived | normalized pre-move position identity | SHA-256 over normalized FEN + version | feature-key |
| pre_move_fen | string | no | board state | full pre-move FEN | valid FEN | feature |
| normalized_pre_move_fen | string | no | derived | normalized FEN first four fields | equals normalization function output | feature-key |
| side_to_move | string | no | board state | side to move before played move (`w`/`b`) | in `{w,b}` | feature |
| played_move_uci | string | no | move state | played move UCI | legal in pre-move board | label |
| played_move_san | string | yes | derived | SAN representation | if present, must parse on board | metadata/label aid |
| halfmove_clock | int32 | no | board state | pre-move halfmove clock | >= 0 | metadata |
| fullmove_number | int32 | no | board state | pre-move fullmove number | >= 1 | metadata |
| clock_annotation_raw | string | yes | PGN comments | optional clock annotation from comments | optional raw capture | metadata (potential leakage if post-move) |
| schema_version | string | no | constant | contract version | equals active schema | metadata |

Leakage note:

- `played_move_uci` is the policy label, not an input feature.
- final result/termination must not be joined as features for pre-move prediction.

## PositionRecord

Grain:

- one row per normalized position identity.

Primary key:

- `position_id`.

Foreign keys:

- none required at this layer; referenced by MoveRecord and EngineEvalRecord.

| Field | Type | Null | Source | Meaning | Validation | Role |
| --- | --- | --- | --- | --- | --- | --- |
| position_id | string | no | derived | stable position identity | deterministic SHA-256 | feature-key |
| normalized_fen | string | no | derived | `fen4_legal_ep_v1` representation | valid normalized format | feature |
| side_to_move | string | no | normalized FEN | mover side | in `{w,b}` | feature |
| castling_rights | string | no | normalized FEN | castling token | legal FEN castling format | feature |
| en_passant_square | string | yes | normalized FEN | legal en-passant target | `null` or valid square | feature |
| encoding_version | string | no | constant | board encoding identifier | equals supported version | metadata |
| schema_version | string | no | constant | schema version | equals active schema | metadata |

Deduplication note:

- same normalized position can appear in many games and many plies.

## EngineEvalRecord (Contract Only, Parser Deferred)

Grain:

- one row per evaluated position and PV rank from external engine data.

Primary key:

- composite (`position_id`, `engine_source_version`, `pv_rank`, `depth`, `nodes`).

Foreign keys:

- `position_id` -> `PositionRecord.position_id`.

| Field | Type | Null | Source | Meaning | Validation | Role |
| --- | --- | --- | --- | --- | --- | --- |
| position_id | string | no | join key | normalized position ID | must exist in position space | feature-key |
| normalized_fen | string | no | source/join | normalized FEN for audit | should match position mapping | metadata |
| engine_source_version | string | no | source metadata | engine config/version identifier | non-empty | metadata |
| depth | int32 | yes | source | search depth | >= 0 if present | metadata |
| nodes | int64 | yes | source | searched nodes/kilonodes | >= 0 if present | metadata |
| pv_rank | int32 | no | source | principal variation rank | >= 1 | label-source metadata |
| centipawn_score | int32 | yes | source | cp evaluation from side to move | int range sanity checks | label-source |
| mate_score | int32 | yes | source | mate-in-N signal | int sanity checks | label-source |
| principal_variation_uci | string | yes | source | PV move sequence in UCI | parseable sequence if present | label-source metadata |
| schema_version | string | no | constant | schema version | equals active schema | metadata |

## IngestionErrorRecord

Grain:

- one row per rejected/warning ingestion event.

Primary key:

- no natural global key; operational composite (`run_id`, `source_archive`, `source_game_index`, `error_code`).

| Field | Type | Null | Source | Meaning | Validation | Role |
| --- | --- | --- | --- | --- | --- | --- |
| source_archive | string | no | runtime | archive containing error | non-empty | metadata |
| source_game_index | int64 | no | stream ordinal | game index where issue occurred | >= 0 | metadata |
| error_code | string | no | parser | normalized issue code | controlled vocabulary | metadata |
| error_message | string | no | parser | human-readable details | non-empty | metadata |
| handling_decision | string | no | parser policy | rejected vs accepted-with-warning | enum | metadata |
| recoverability | string | no | parser policy | recoverable/fatal | enum | metadata |
| run_id | string | no | run context | ingestion run identifier | non-empty | metadata |
| schema_version | string | no | constant | schema version | equals active schema | metadata |

Run provenance note:

- `run_id` ties each ingestion error row to one concrete execution attempt.

## RunManifest

Grain:

- one row per ingestion run.

Primary key:

- `run_id`.

| Field | Type | Null | Source | Meaning | Validation | Role |
| --- | --- | --- | --- | --- | --- | --- |
| run_id | string | no | runtime | unique run identity | non-empty | metadata |
| created_at_utc | string | no | runtime | UTC timestamp | ISO-8601 | metadata |
| git_commit | string | yes | VCS | commit hash for reproducibility | hash-like if present | metadata |
| source_archive | string | no | runtime | archive name | non-empty | metadata |
| source_archive_sha256 | string | no | runtime | integrity hash | 64-char hex | metadata |
| configuration_hash | string | no | runtime | hash of config payload | deterministic hash | metadata |
| schema_version | string | no | constant | schema version used | equals active schema | metadata |
| position_normalization_version | string | no | constant | normalization version | supported value | metadata |
| board_encoding_version | string | no | constant | board encoding version | supported value | metadata |
| action_encoding_version | string | no | constant | action encoding version | supported value | metadata |
| requested_games | int32 | yes | config | requested max games | >= 0 or null | metadata |
| accepted_games | int32 | no | runtime | accepted valid games | >= 0 | metadata |
| rejected_games | int32 | no | runtime | rejected games | >= 0 | metadata |
| emitted_moves | int32 | no | runtime | total emitted move rows | >= 0 | metadata |
| output_artifacts | list<string> | no | runtime | written artifact paths | path list | metadata |
| python_version | string | no | runtime | runtime version | non-empty | metadata |
| package_versions | map<string,string> | no | runtime | package snapshot | key/value map | metadata |

Phase 1.2a manifest extensions:

- `versions.ingestion_pipeline_version` (e.g. `parquet_etl_v1`), incremented when output-affecting ingestion logic changes without an Arrow schema change.
- Explicit timing fields in `performance`:
	- `checksum_duration_seconds`
	- `processing_and_validation_duration_seconds`
	- `total_duration_seconds`
	- `parse_and_reconstruct_seconds`
	- `record_transformation_seconds`
	- `arrow_parquet_write_seconds`
	- `duckdb_schema_validation_seconds`
	- `dataset_publication_seconds`
- Throughput fields with explicit names (`processing_*` and `total_*`) to avoid ambiguous timing claims.

## Explicit Prohibited Features in This Phase

- final game result as input for move prediction rows;
- termination reason as input feature;
- post-move clock annotations as pre-move feature;
- future opening labels at early plies when they embed future moves.

## Physical Bronze Layout (Phase 1.2a)

Published datasets use deterministic paths:

```text
data/processed/
	datasets/
		<dataset_id>/
			games/source_month=YYYY-MM/part-000000.parquet
			moves/source_month=YYYY-MM/part-000000.parquet
			ingestion_errors/source_month=YYYY-MM/part-000000.parquet
			_manifest.json
```

Operational guarantees:

- `dataset_id` is deterministic for a fixed source checksum + effective config.
- `dataset_id` reflects logical transformation identity; Parquet files are not promised to be byte-identical across all machines/tool versions.
- Part filenames are deterministic within a dataset (`part-000000`, `part-000001`, ...).
- Publication occurs only after staged DuckDB + schema validation passes.
- Re-running the same effective config reuses an existing completed dataset rather than appending duplicates.

## Warehouse Contracts (Phase 1.2b)

### Source Registration Contract

Before dbt runs, `python -m chesslens.warehouse.preflight` must succeed and register DuckDB bronze views:

- `main.bronze_games`
- `main.bronze_moves`
- `main.bronze_ingestion_errors`
- `main.bronze_manifest`

Preflight must fail for:

- missing `CHESSLENS_DATASET_ROOT`;
- missing `_manifest.json`;
- non-`complete` manifest status;
- missing required parquet parts;
- manifest counts that disagree with physical parquet row counts.

### `stg_games`

Grain:

- one row per `game_id`.

Required checks:

- `game_id` unique and non-null;
- `result` in `{1-0, 0-1, 1/2-1/2}`.

### `stg_moves`

Grain:

- one row per (`game_id`, `ply`).

Required checks:

- (`game_id`, `ply`) uniqueness;
- `game_id` foreign key to `stg_games`;
- `position_id` non-null;
- `side_to_move` in `{w, b}`;
- ply is contiguous and starts at zero within each game.

### `int_positions`

Grain:

- one row per `position_id` across all games.

Required checks:

- one `position_id` must map to exactly one `normalized_fen`.

Core fields:

- `position_id`, `normalized_fen`, `side_to_move`, `castling_rights`, `en_passant_square`, `occurrence_count`, `first_seen_source_month`, `last_seen_source_month`.

### `int_move_context`

Grain:

- one row per (`game_id`, `ply`).

Contract:

- enriches move rows with game-level metadata without changing move grain.

### `fct_move_events`

Grain:

- one row per (`game_id`, `ply`).

Contract:

- includes explicit outcome labels (`final_result_label`, `termination_label`) for analytics;
- these outcome labels are prohibited as pre-move model features.

### `mart_policy_examples`

Grain:

- one row per (`game_id`, `ply`).

Contract:

- contains leakage-aware pre-move fields and supervised target `played_move_uci`;
- computes mover/opponent ratings from `side_to_move`;
- must not contain post-outcome columns (`result`, `termination`, `final_result_label`, `termination_label`, `ply_count`, `game_ply_count`).

### Manifest Reconciliation Contract

Warehouse tests must assert that:

- bronze and staging row counts match for all three bronze datasets;
- manifest counts (`accepted_games`, `emitted_moves`, `error_records`) match staging row counts;
- manifest dataset row-count entries match staging row counts.

## Sharding and Collection Contracts (Phase 1.2c)

### Global game identity invariant

- `game_id = sha256(parent_archive_sha256 + "|" + global_source_game_index)`.
- `source_game_index` remains the **global** index over the parent archive (unchanged
  schema). It counts every raw game boundary, including tolerantly-rejected games.
- A game has an identical `game_id` and `GameRecord` whether ingested directly from
  the parent archive or from a shard with a non-zero global offset.

### Dataset manifest addition: `source_lineage`

Per-shard dataset manifests add an optional `source_lineage` object (null for direct
ingestion):

- `parent_archive_filename`, `parent_archive_sha256`: identity anchor for `game_id`.
- `global_game_index_offset`: added to local index to form the global index.
- `shard_index`, `shard_filename`, `shard_sha256`: physical shard lineage.

The dataset manifest `source.archive_sha256` remains the **physical** shard SHA (used
for dataset identity and reuse checks); the parent SHA lives in `source_lineage`.

### Shard manifest (`_shard_manifest.json`)

Grain: one manifest per sharding plan (per parent archive + month + shard size).

Required fields: `manifest_version`, `splitter_pipeline_version`, `status`
(`incomplete`/`complete`), parent archive filename/SHA-256/size, `source_month`,
`expected_total_games`, `games_per_shard`, compression settings, timestamps,
`total_emitted_games`, `total_shard_count`, performance (peak RSS, active seconds,
games/second), and one entry per shard with: `shard_index`, `filename`,
`start_global_index` (inclusive), `end_global_index` (exclusive), `game_count`,
`compressed_bytes`, `sha256`, `status`.

Range invariants (validated): indices `0..n-1` in order; first start `0`; each next
start equals the previous end; no gaps or overlaps; final end equals
`total_emitted_games`. Writes are atomic (temp file + `os.replace`). The logical
identity hash excludes timestamps, byte sizes, and paths.

### Collection manifest (`_collection_manifest.json`)

Grain: one manifest per collection (per shard plan + ingestion identity).

Required fields: `manifest_version`, `collection_id`, `status`, parent
filename/SHA-256, `source_month`, `shard_manifest_identity_hash`, `games_per_shard`,
versions (sharding/ingestion pipeline, schema/encoding, non-secret `player_hmac_key_id`),
`expected_raw_games`, `selected_shard_count`, `completed_shard_count`, aggregate
`counts` (`scanned`/`accepted`/`rejected`/`emitted_moves`/`error_records`),
`aggregate_output_bytes`, `timing` (active seconds, aggregate rates, peak RSS,
started/updated timestamps), `portfolio_10m_satisfied`, and one entry per processed
shard with its `dataset_id`, `dataset_relpath`, global range, counts, and per-shard
performance.

Phase 1.2d adds optional run-level observability metadata under `last_run`, including
worker count, selected shard indices/count, new shards processed, wall-clock and
worker-active durations, games/moves per wall-clock second, per-worker peak RSS,
maximum worker peak RSS, process-tree peak RSS, failed shard indices, and optional
speedup/parallel-efficiency values when a valid workers=1 baseline is provided.

Reconciliation invariants (validated): shard-index contiguity; half-open global range
contiguity; unique `dataset_id`s; aggregate counts equal the sum of per-shard counts;
`portfolio_10m_satisfied` is `true` only when `status == complete` and
`accepted_games >= 10,000,000`. Deep row-level uniqueness (global `game_id` and
`(game_id, ply)`) is enforced by the collection dbt build over the manifest-listed
union.

### `collection_id` determinism

`collection_id` is derived only from logical inputs — parent SHA-256, `source_month`,
`shard_manifest_identity_hash`, `games_per_shard`, pipeline/schema/encoding versions,
and the non-secret `player_hmac_key_id`. It never includes timestamps, secrets,
absolute paths, or run ids.

### Collection mutation lock

Mutating sharded-ingestion runs are single-writer: a collection-scoped lock file is
created atomically before selecting pending shards and removed on completion. If a
lock owner process is no longer alive, stale-lock recovery removes the stale lock and
proceeds safely. Read-only status checks remain lock-free.

## Date Enrichment Sidecar Contract (Phase 2.1)

Phase 2.1 temporal splitting now uses a separate, header-only date enrichment sidecar.

Published path:

- `data/processed/date_enrichment/<date_enrichment_id>/...`

Canonical date policy (`utc_date_first_v1`):

- if `UTCDate` is a valid calendar date, use it;
- else if `Date` is a valid calendar date, use it;
- else canonical date is null with source `missing_or_invalid`.

Per-game sidecar grain and required fields:

- `game_id` (stable identity, same derivation as ingestion);
- `source_game_index`;
- `canonical_played_date_iso` (`YYYY-MM-DD` or null);
- `canonical_date_source` in `{utc_date, date, missing_or_invalid}`;
- raw header audit fields and header-validity booleans;
- resolver version.

Manifest/reuse invariants:

- shard entries are contiguous by `shard_index` and half-open global ranges;
- one sidecar row per game in each shard range;
- artifact checksums are validated before reuse;
- manifest writes are atomic and resumable per shard.

## Modeling Dataset Contracts (Phase 2.1)

Phase 2.1 publishes leakage-aware supervised modeling datasets under:

- `data/modeling/datasets/<modeling_dataset_id>/game_assignments/...`
- `data/modeling/datasets/<modeling_dataset_id>/policy_examples/...`
- `data/modeling/datasets/<modeling_dataset_id>/leakage_audits/...`

with staged-write, validation, and atomic publish semantics mirroring ingestion.

### `game_assignments`

Grain:

- one row per selected game (`game_id`).

Key fields:

- `temporal_split` in `{train, validation, test}`;
- `temporal_split_reason` describing direct date interval or policy assignment;
- `played_date_raw` from upstream `stg_games` (audit-preserved);
- `played_date_iso` from canonical resolver input;
- `played_date_source` in `{utc_date, date, missing_or_invalid, legacy_played_date}`;
- deterministic `sample_score_u64` from hash-rule sampling;
- `is_player_holdout_game` and `player_disjoint_training_eligible` derived from
	deterministic player-hash holdout policy.

Invariants:

- each selected `game_id` appears once;
- temporal split assignment is game-level (all moves inherit game split);
- held-out player hashes must not appear inside the player-disjoint training
	population.

### `policy_examples`

Grain:

- one row per selected move (`game_id`, `ply`).

Target fields:

- `played_move_uci_target`;
- `policy_target_action_index` in `[0, 4671]`;
- `value_target_wdl` in `{win, draw, loss}` from side-to-move + final result.

Feature constraints:

- pre-move board/context fields only;
- post-outcome fields are present only as explicit targets/audit metadata;
- schema + encoding + label version fields are persisted per row.

### `novel_position_test` subset

`policy_examples` additionally includes partition `temporal_split=novel_position_test`
containing test rows whose `position_id` does not appear in train. This partition is
diagnostic and complements full test-split evaluation.

### Modeling manifest identity and reuse

`modeling_dataset_id` is derived from:

- upstream collection identity (`collection_id`, collection-manifest hash),
- upstream warehouse provenance identity (`provenance_version`, canonical provenance SHA-256,
	warehouse kind),
- upstream date enrichment identity (`date_enrichment_id`, enrichment manifest SHA-256,
	resolver version) when enabled,
- modeling/split/schema/encoding version identifiers,
- sampling and player-holdout rule parameters,
- split date ranges + missing-date policy,
- output-affecting query/runtime options.

### Warehouse provenance contract (`warehouse_provenance_v1`)

Phase 2.1b requires an explicit warehouse provenance artifact passed through
`input.warehouse_provenance_path` (or `--warehouse-provenance-path`).

Generate this artifact from the actual DuckDB snapshot using:

- `python -m chesslens.modeling.generate_warehouse_provenance ...`

Template/example files are documentation aids only and are not acceptance evidence
until regenerated and validated against the real target DuckDB relations.

The artifact must include:

- `provenance_version` and `warehouse_kind` (`full`, `deterministic_sample`, `fixture`),
- `collection_id` from the parent collection manifest,
- relation names used by modeling (`games_relation`, `move_context_relation`),
- declared warehouse snapshot row counts (`snapshot_counts.games`, `snapshot_counts.moves`),
- stable transformation identity (for example dbt manifest hash),
- canonical payload hash `warehouse_provenance_sha256`.

For `deterministic_sample`, sampling metadata is mandatory:

- `parent_collection_id`, sampling rule + predicate, requested percent,
- parent full counts,
- sampled counts,
- optional effective rate percent.

Before writing modeling output, the builder validates provenance against DuckDB:

- required relations must exist,
- provenance collection ID must match the collection manifest,
- declared snapshot counts must match DuckDB relation counts,
- sampled/full declarations must be internally consistent,
- `warehouse_kind=full` must match full collection counts (sampled warehouses cannot
	silently claim full input).

Publication contract:

- write to `data/modeling/staging/<modeling_dataset_id>__<run_id>/`;
- validate uniqueness/split/holdout/range invariants and artifact existence;
- write completed `_manifest.json` and `_SUCCESS`;
- atomically move to `data/modeling/datasets/<modeling_dataset_id>/`.

Reuse contract:

- existing published dataset can be reused only when manifest identity fields match
	expected values;
- identity mismatch is a hard failure (tamper/staleness guard).
- when date enrichment is configured, modeling fails fast if enrichment does not
	fully cover `games_relation.game_id` or if enrichment has duplicate `game_id`
	rows.

### Two-stage sampling semantics

Modeling manifests now record sampling as two separate stages:

- warehouse stage (`sampling_stages.warehouse.*`): how DuckDB was produced from the
	full collection (full/fixture/deterministic sample),
- modeling stage (`sampling_stages.modeling.*`): game-level modeling selection from
	that warehouse.

The manifest also records
`sampling_stages.cumulative_effective_rate_percent_of_full_collection` when derivable.
This prevents mislabeling a two-stage flow (for example 10% warehouse then 10%
modeling) as a direct single-stage 10% sample.