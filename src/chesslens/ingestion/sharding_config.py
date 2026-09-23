"""Typed configuration for streaming PGN sharding (Phase 1.2c)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_SOURCE_MONTH_IN_NAME_RE = re.compile(r"(\d{4}-\d{2})")
_ALLOWED_CODECS = {"zstd"}
_SECRET_KEY_HINTS = ("secret", "password", "token", "hmac_key")


class ShardingConfigError(ValueError):
    """Raised when a sharding configuration is invalid."""


@dataclass(frozen=True)
class ShardingConfig:
    input_path: Path
    shard_output_root: Path
    source_month: str
    expected_source_sha256: str | None
    expected_total_games: int | None
    games_per_shard: int
    compression_codec: str
    compression_level: int
    max_game_bytes: int
    chunk_size: int


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ShardingConfigError(f"{field_name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ShardingConfigError(f"{field_name} must be positive")
    return parsed


def _validate_calendar_month(value: str) -> str:
    if not _SOURCE_MONTH_RE.fullmatch(value):
        raise ShardingConfigError("source_month must be in YYYY-MM format")
    try:
        datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise ShardingConfigError("source_month must be a real calendar month") from exc
    return value


def _reject_secret_values(raw: Mapping[str, Any]) -> None:
    for key in raw:
        lowered = str(key).lower()
        if any(hint in lowered for hint in _SECRET_KEY_HINTS):
            raise ShardingConfigError(
                f"Sharding config must not contain secret-like key: {key!r}"
            )


def load_sharding_config(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> ShardingConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ShardingConfigError(f"Sharding config not found: {config_path}")
    raw_loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw: dict[str, Any] = dict(raw_loaded) if isinstance(raw_loaded, dict) else {}
    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    _reject_secret_values(raw)

    if "input_path" not in raw:
        raise ShardingConfigError("Sharding config must include input_path")
    input_path = _resolve_path(str(raw["input_path"]))

    shard_output_root = _resolve_path(str(raw.get("shard_output_root", "data/raw_shards")))

    source_month_raw = raw.get("source_month")
    if source_month_raw is None:
        raise ShardingConfigError("Sharding config must include source_month")
    source_month = _validate_calendar_month(str(source_month_raw).strip())

    inferred = _SOURCE_MONTH_IN_NAME_RE.search(input_path.name)
    if inferred is not None and inferred.group(1) != source_month:
        raise ShardingConfigError(
            "Configured source_month does not match archive filename month: "
            f"{source_month!r} != {inferred.group(1)!r}"
        )

    expected_source_sha256 = raw.get("expected_source_sha256")
    if expected_source_sha256 is not None:
        expected_source_sha256 = str(expected_source_sha256).strip().lower()
        if not _SHA256_HEX_RE.fullmatch(expected_source_sha256):
            raise ShardingConfigError(
                "expected_source_sha256 must be a 64-character lowercase hex string"
            )

    expected_total_games = raw.get("expected_total_games")
    if expected_total_games is not None:
        expected_total_games = _positive_int(
            expected_total_games, field_name="expected_total_games"
        )

    games_per_shard = _positive_int(
        raw.get("games_per_shard", 250000), field_name="games_per_shard"
    )

    compression_codec = str(raw.get("compression_codec", "zstd")).strip().lower()
    if compression_codec not in _ALLOWED_CODECS:
        raise ShardingConfigError(
            f"Unsupported compression_codec: {compression_codec!r} (allowed: zstd)"
        )
    compression_level = int(raw.get("compression_level", 10))
    if not (1 <= compression_level <= 22):
        raise ShardingConfigError("compression_level must be between 1 and 22")

    max_game_bytes = _positive_int(
        raw.get("max_game_bytes", 8 * 1024 * 1024), field_name="max_game_bytes"
    )
    chunk_size = _positive_int(raw.get("chunk_size", 1 << 20), field_name="chunk_size")

    return ShardingConfig(
        input_path=input_path,
        shard_output_root=shard_output_root,
        source_month=source_month,
        expected_source_sha256=expected_source_sha256,
        expected_total_games=expected_total_games,
        games_per_shard=games_per_shard,
        compression_codec=compression_codec,
        compression_level=compression_level,
        max_game_bytes=max_game_bytes,
        chunk_size=chunk_size,
    )
