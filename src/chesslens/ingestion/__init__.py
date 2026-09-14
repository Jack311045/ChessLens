"""Streaming PGN ingestion and profiling utilities."""

from chesslens.ingestion.config import IngestionConfig, load_ingestion_config
from chesslens.ingestion.pgn_reader import (
    ParsedGame,
    compute_sha256,
    hash_player_identifier,
    is_complete_game,
    iter_parsed_games,
    iter_raw_games,
    stable_game_id,
)

__all__ = [
    "IngestionConfig",
    "load_ingestion_config",
    "ParsedGame",
    "iter_raw_games",
    "iter_parsed_games",
    "compute_sha256",
    "stable_game_id",
    "hash_player_identifier",
    "is_complete_game",
]