"""Explicit Arrow/Polars schemas and lightweight invariants for logical records."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import fields, is_dataclass
from typing import Any, cast

import polars as pl
import pyarrow as pa

from chesslens.domain.records import (
    EngineEvalRecord,
    GameRecord,
    IngestionErrorRecord,
    MoveRecord,
    PositionRecord,
    RunManifest,
)

GAME_ARROW_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("source_archive", pa.string(), nullable=False),
        pa.field("source_archive_sha256", pa.string(), nullable=False),
        pa.field("source_month", pa.string(), nullable=True),
        pa.field("source_game_index", pa.int64(), nullable=False),
        pa.field("event", pa.string(), nullable=True),
        pa.field("played_date", pa.string(), nullable=True),
        pa.field("round", pa.string(), nullable=True),
        pa.field("result", pa.string(), nullable=True),
        pa.field("white_rating", pa.int32(), nullable=True),
        pa.field("black_rating", pa.int32(), nullable=True),
        pa.field("time_control_raw", pa.string(), nullable=True),
        pa.field("eco", pa.string(), nullable=True),
        pa.field("opening", pa.string(), nullable=True),
        pa.field("termination", pa.string(), nullable=True),
        pa.field("white_player_hash", pa.string(), nullable=True),
        pa.field("black_player_hash", pa.string(), nullable=True),
        pa.field("ply_count", pa.int32(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
)

MOVE_ARROW_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("ply", pa.int32(), nullable=False),
        pa.field("position_id", pa.string(), nullable=False),
        pa.field("pre_move_fen", pa.string(), nullable=False),
        pa.field("normalized_pre_move_fen", pa.string(), nullable=False),
        pa.field("side_to_move", pa.string(), nullable=False),
        pa.field("played_move_uci", pa.string(), nullable=False),
        pa.field("played_move_san", pa.string(), nullable=True),
        pa.field("halfmove_clock", pa.int32(), nullable=False),
        pa.field("fullmove_number", pa.int32(), nullable=False),
        pa.field("clock_annotation_raw", pa.string(), nullable=True),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
)

POSITION_ARROW_SCHEMA = pa.schema(
    [
        pa.field("position_id", pa.string(), nullable=False),
        pa.field("normalized_fen", pa.string(), nullable=False),
        pa.field("side_to_move", pa.string(), nullable=False),
        pa.field("castling_rights", pa.string(), nullable=False),
        pa.field("en_passant_square", pa.string(), nullable=True),
        pa.field("encoding_version", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
)

ENGINE_EVAL_ARROW_SCHEMA = pa.schema(
    [
        pa.field("position_id", pa.string(), nullable=False),
        pa.field("normalized_fen", pa.string(), nullable=False),
        pa.field("engine_source_version", pa.string(), nullable=False),
        pa.field("depth", pa.int32(), nullable=True),
        pa.field("nodes", pa.int64(), nullable=True),
        pa.field("pv_rank", pa.int32(), nullable=False),
        pa.field("centipawn_score", pa.int32(), nullable=True),
        pa.field("mate_score", pa.int32(), nullable=True),
        pa.field("principal_variation_uci", pa.string(), nullable=True),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
)

INGESTION_ERROR_ARROW_SCHEMA = pa.schema(
    [
        pa.field("source_archive", pa.string(), nullable=False),
        pa.field("source_game_index", pa.int64(), nullable=False),
        pa.field("error_code", pa.string(), nullable=False),
        pa.field("error_message", pa.string(), nullable=False),
        pa.field("handling_decision", pa.string(), nullable=False),
        pa.field("recoverability", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
    ]
)

RUN_MANIFEST_ARROW_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("created_at_utc", pa.string(), nullable=False),
        pa.field("git_commit", pa.string(), nullable=True),
        pa.field("source_archive", pa.string(), nullable=False),
        pa.field("source_archive_sha256", pa.string(), nullable=False),
        pa.field("configuration_hash", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("position_normalization_version", pa.string(), nullable=False),
        pa.field("board_encoding_version", pa.string(), nullable=False),
        pa.field("action_encoding_version", pa.string(), nullable=False),
        pa.field("requested_games", pa.int32(), nullable=True),
        pa.field("accepted_games", pa.int32(), nullable=False),
        pa.field("rejected_games", pa.int32(), nullable=False),
        pa.field("emitted_moves", pa.int32(), nullable=False),
        pa.field("output_artifacts", pa.list_(pa.string()), nullable=False),
        pa.field("python_version", pa.string(), nullable=False),
        pa.field("package_versions", pa.map_(pa.string(), pa.string()), nullable=False),
    ]
)

SCHEMA_REGISTRY: dict[type[Any], pa.Schema] = {
    GameRecord: GAME_ARROW_SCHEMA,
    MoveRecord: MOVE_ARROW_SCHEMA,
    PositionRecord: POSITION_ARROW_SCHEMA,
    EngineEvalRecord: ENGINE_EVAL_ARROW_SCHEMA,
    IngestionErrorRecord: INGESTION_ERROR_ARROW_SCHEMA,
    RunManifest: RUN_MANIFEST_ARROW_SCHEMA,
}


def _record_to_dict(record: Any) -> dict[str, Any]:
    if not is_dataclass(record) or isinstance(record, type):
        raise TypeError(f"Expected dataclass record, received: {type(record)}")
    return {field.name: getattr(record, field.name) for field in fields(record)}


def records_to_arrow_table(records: Sequence[Any], schema: pa.Schema) -> pa.Table:
    if not records:
        arrays = [pa.array([], type=field.type) for field in schema]
        return pa.Table.from_arrays(arrays=arrays, schema=schema)

    rows = [_record_to_dict(record) for record in records]
    return pa.Table.from_pylist(rows, schema=schema)


def records_to_polars_frame(records: Iterable[Any], schema: pa.Schema) -> pl.DataFrame:
    rows = [_record_to_dict(record) for record in records]
    if not rows:
        return cast(pl.DataFrame, pl.from_arrow(records_to_arrow_table([], schema)))
    return cast(pl.DataFrame, pl.from_arrow(pa.Table.from_pylist(rows, schema=schema)))