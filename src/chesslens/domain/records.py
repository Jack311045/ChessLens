"""Versioned logical records used by ChessLens ingestion contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

SCHEMA_VERSION = "1.1.0"
POSITION_NORMALIZATION_VERSION = "fen4_legal_ep_v1"
BOARD_ENCODING_VERSION = "board18_abs_v1"
ACTION_ENCODING_VERSION = "action8x8x73_v1"
INGESTION_PIPELINE_VERSION = "parquet_etl_v1"
SHARDING_PIPELINE_VERSION = "pgn_shard_v1"
SHARD_MANIFEST_VERSION = "1.0.0"
COLLECTION_MANIFEST_VERSION = "1.0.0"

SideToMove = Literal["w", "b"]
HandlingDecision = Literal["rejected", "accepted_with_warning", "accepted"]
Recoverability = Literal["recoverable", "fatal"]


@dataclass(frozen=True)
class GameRecord:
    """One row per source game."""

    game_id: str
    source_archive: str
    source_archive_sha256: str
    source_month: str | None
    source_game_index: int
    event: str | None
    played_date: str | None
    round: str | None
    result: str | None
    white_rating: int | None
    black_rating: int | None
    time_control_raw: str | None
    eco: str | None
    opening: str | None
    termination: str | None
    white_player_hash: str | None
    black_player_hash: str | None
    ply_count: int
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class MoveRecord:
    """One row per played ply. Ply is zero-based and pre-move."""

    game_id: str
    ply: int
    position_id: str
    pre_move_fen: str
    normalized_pre_move_fen: str
    side_to_move: SideToMove
    played_move_uci: str
    played_move_san: str | None
    halfmove_clock: int
    fullmove_number: int
    clock_annotation_raw: str | None
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class PositionRecord:
    """One row per normalized position identity."""

    position_id: str
    normalized_fen: str
    side_to_move: SideToMove
    castling_rights: str
    en_passant_square: str | None
    encoding_version: str
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class EngineEvalRecord:
    """Contract for engine evaluation rows (parser deferred to a later phase)."""

    position_id: str
    normalized_fen: str
    engine_source_version: str
    depth: int | None
    nodes: int | None
    pv_rank: int
    centipawn_score: int | None
    mate_score: int | None
    principal_variation_uci: str | None
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class IngestionErrorRecord:
    """Record describing a rejected or warning-level ingestion event."""

    source_archive: str
    source_game_index: int
    error_code: str
    error_message: str
    handling_decision: HandlingDecision
    recoverability: Recoverability
    run_id: str
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class RunManifest:
    """Run-level metadata and reproducibility audit fields."""

    run_id: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(tz=UTC).isoformat(timespec="seconds")
    )
    git_commit: str | None = None
    source_archive: str = ""
    source_archive_sha256: str = ""
    configuration_hash: str = ""
    schema_version: str = SCHEMA_VERSION
    position_normalization_version: str = POSITION_NORMALIZATION_VERSION
    board_encoding_version: str = BOARD_ENCODING_VERSION
    action_encoding_version: str = ACTION_ENCODING_VERSION
    requested_games: int | None = None
    accepted_games: int = 0
    rejected_games: int = 0
    emitted_moves: int = 0
    output_artifacts: list[str] = field(default_factory=list)
    python_version: str = ""
    package_versions: dict[str, str] = field(default_factory=dict)