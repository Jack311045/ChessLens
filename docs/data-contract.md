# Data Contract (Phase 1.2a)

Schema version: `1.1.0`.

This document defines logical contracts and semantics for ingestion.

Phase 1.2a publishes bronze Parquet datasets for:

- `games`
- `moves`
- `ingestion_errors`

with Hive-style partitioning by `source_month`.

`PositionRecord` remains a logical contract, but this phase intentionally does not publish a physical bronze `positions` table. Global deduplication of positions is deferred to Phase 1.2b dbt SQL (`int_positions`) so deduplication is global, not only game-local or batch-local.

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
- Part filenames are deterministic within a dataset (`part-000000`, `part-000001`, ...).
- Publication occurs only after staged DuckDB + schema validation passes.
- Re-running the same effective config reuses an existing completed dataset rather than appending duplicates.