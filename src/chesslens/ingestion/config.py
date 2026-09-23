"""Configuration loader for ingestion, fixture generation, and profiling."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from chesslens.domain.records import (
    ACTION_ENCODING_VERSION,
    BOARD_ENCODING_VERSION,
    POSITION_NORMALIZATION_VERSION,
    SCHEMA_VERSION,
)

PlayerHashMode = Literal["fixture_placeholder", "hmac_sha256"]

_ALLOWED_HASH_MODES: set[str] = {"fixture_placeholder", "hmac_sha256"}
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


@dataclass(frozen=True)
class IngestionConfig:
    input_path: Path
    output_root: Path
    expected_source_sha256: str | None
    max_games: int | None
    strict: bool
    require_complete_games: bool
    batch_games: int
    max_buffered_records: int
    parquet_compression: str
    parquet_row_group_size: int
    player_hash_mode: PlayerHashMode
    player_hmac_key_env: str | None
    player_hmac_key_id_env: str | None
    source_month: str | None
    fixture_output_pgn: Path | None = None
    fixture_output_zst: Path | None = None
    fixture_manifest_path: Path | None = None
    profile_output_path: Path | None = None
    schema_version: str = SCHEMA_VERSION
    position_normalization_version: str = POSITION_NORMALIZATION_VERSION
    board_encoding_version: str = BOARD_ENCODING_VERSION
    action_encoding_version: str = ACTION_ENCODING_VERSION


def _resolve_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Cannot coerce to bool: {value!r}")


def _as_optional_non_negative_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer or null")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _as_positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected object at root of config file: {path}")
    return dict(data)


def load_ingestion_config(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> IngestionConfig:
    config_path = Path(path)
    raw = _load_yaml_dict(config_path)

    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    if "input_path" not in raw:
        raise ValueError("Config must include input_path")

    max_games = _as_optional_non_negative_int(raw.get("max_games", 100), field_name="max_games")
    strict = _as_bool(raw.get("strict", False))
    require_complete_games = _as_bool(raw.get("require_complete_games", True))

    batch_games = _as_positive_int(raw.get("batch_games", 1000), field_name="batch_games")
    max_buffered_records = _as_positive_int(
        raw.get("max_buffered_records", 50000),
        field_name="max_buffered_records",
    )

    parquet_compression = str(raw.get("parquet_compression", "zstd")).strip().lower()
    if not parquet_compression:
        raise ValueError("parquet_compression must be non-empty")

    parquet_row_group_size = _as_positive_int(
        raw.get("parquet_row_group_size", 10000),
        field_name="parquet_row_group_size",
    )

    player_hash_mode_raw = str(raw.get("player_hash_mode", "fixture_placeholder")).strip()
    if player_hash_mode_raw not in _ALLOWED_HASH_MODES:
        raise ValueError(
            "player_hash_mode must be one of: fixture_placeholder, hmac_sha256"
        )
    player_hash_mode = player_hash_mode_raw

    player_hmac_key_env = _as_optional_string(raw.get("player_hmac_key_env"))
    player_hmac_key_id_env = _as_optional_string(raw.get("player_hmac_key_id_env"))
    if player_hash_mode == "hmac_sha256":
        if player_hmac_key_env is None:
            raise ValueError("player_hmac_key_env is required for hmac_sha256 mode")
        if player_hmac_key_id_env is None:
            raise ValueError("player_hmac_key_id_env is required for hmac_sha256 mode")

    expected_source_sha256 = _as_optional_string(raw.get("expected_source_sha256"))
    if expected_source_sha256 is not None:
        expected_source_sha256 = expected_source_sha256.lower()
        if not _SHA256_HEX_RE.fullmatch(expected_source_sha256):
            raise ValueError("expected_source_sha256 must be a 64-character lowercase hex string")

    source_month = _as_optional_string(raw.get("source_month"))
    if source_month is not None and not _SOURCE_MONTH_RE.fullmatch(source_month):
        raise ValueError("source_month must be in YYYY-MM format")

    config = IngestionConfig(
        input_path=_resolve_path(str(raw["input_path"])) or Path(""),
        output_root=_resolve_path(str(raw.get("output_root", "data/processed"))) or Path(""),
        expected_source_sha256=expected_source_sha256,
        max_games=max_games,
        strict=strict,
        require_complete_games=require_complete_games,
        batch_games=batch_games,
        max_buffered_records=max_buffered_records,
        parquet_compression=parquet_compression,
        parquet_row_group_size=parquet_row_group_size,
        player_hash_mode=cast(PlayerHashMode, player_hash_mode),
        player_hmac_key_env=player_hmac_key_env,
        player_hmac_key_id_env=player_hmac_key_id_env,
        source_month=source_month,
        fixture_output_pgn=_resolve_path(raw.get("fixture_output_pgn")),
        fixture_output_zst=_resolve_path(raw.get("fixture_output_zst")),
        fixture_manifest_path=_resolve_path(raw.get("fixture_manifest_path")),
        profile_output_path=_resolve_path(raw.get("profile_output_path")),
        schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
        position_normalization_version=str(
            raw.get("position_normalization_version", POSITION_NORMALIZATION_VERSION)
        ),
        board_encoding_version=str(raw.get("board_encoding_version", BOARD_ENCODING_VERSION)),
        action_encoding_version=str(raw.get("action_encoding_version", ACTION_ENCODING_VERSION)),
    )

    if config.max_games == 0:
        return config

    if not config.input_path.exists():
        raise FileNotFoundError(f"Input archive not found: {config.input_path}")

    return config


def with_overrides(config: IngestionConfig, **updates: Any) -> IngestionConfig:
    return replace(config, **updates)