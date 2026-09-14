"""Typed domain records for ChessLens."""

from chesslens.domain.records import (
    ACTION_ENCODING_VERSION,
    BOARD_ENCODING_VERSION,
    POSITION_NORMALIZATION_VERSION,
    SCHEMA_VERSION,
    EngineEvalRecord,
    GameRecord,
    IngestionErrorRecord,
    MoveRecord,
    PositionRecord,
    RunManifest,
)

__all__ = [
    "SCHEMA_VERSION",
    "POSITION_NORMALIZATION_VERSION",
    "BOARD_ENCODING_VERSION",
    "ACTION_ENCODING_VERSION",
    "GameRecord",
    "MoveRecord",
    "PositionRecord",
    "EngineEvalRecord",
    "IngestionErrorRecord",
    "RunManifest",
]