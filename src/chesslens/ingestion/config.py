"""Configuration loader for ingestion, fixture generation, and profiling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from chesslens.domain.records import (
    ACTION_ENCODING_VERSION,
    BOARD_ENCODING_VERSION,
    POSITION_NORMALIZATION_VERSION,
    SCHEMA_VERSION,
)


@dataclass(frozen=True)
class IngestionConfig:
    input_path: Path
    max_games: int
    strict: bool
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

    max_games = int(raw.get("max_games", 100))
    if max_games < 0:
        raise ValueError("max_games must be non-negative")

    config = IngestionConfig(
        input_path=_resolve_path(str(raw["input_path"])) or Path(""),
        max_games=max_games,
        strict=_as_bool(raw.get("strict", False)),
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