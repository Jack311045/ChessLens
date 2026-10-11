"""Typed configuration loader for local MLflow tracking integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MlflowTrackingConfig:
    experiment_name: str
    sqlite_path: Path
    artifact_root: Path
    strict: bool
    default_include_model_weights: bool

    @property
    def tracking_uri(self) -> str:
        # MLflow expects sqlite URI form: sqlite:///absolute/path/to/file.db
        return f"sqlite:///{self.sqlite_path.as_posix()}"


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"MLflow config file does not exist: {path.as_posix()}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("MLflow config root must be an object")
    return dict(loaded)


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def _as_non_empty_string(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _as_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def load_mlflow_tracking_config(path: Path) -> MlflowTrackingConfig:
    raw = _load_yaml_dict(path)
    tracking_raw = _as_mapping(raw.get("tracking"), field_name="tracking")

    config = MlflowTrackingConfig(
        experiment_name=_as_non_empty_string(
            tracking_raw.get("experiment_name", "ChessLens"),
            field_name="tracking.experiment_name",
        ),
        sqlite_path=_resolve_path(
            _as_non_empty_string(
                tracking_raw.get("sqlite_path", "data/mlflow/chesslens_mlflow.db"),
                field_name="tracking.sqlite_path",
            )
        ),
        artifact_root=_resolve_path(
            _as_non_empty_string(
                tracking_raw.get("artifact_root", "data/mlflow/artifacts"),
                field_name="tracking.artifact_root",
            )
        ),
        strict=_as_bool(
            tracking_raw.get("strict", True),
            field_name="tracking.strict",
        ),
        default_include_model_weights=_as_bool(
            tracking_raw.get("default_include_model_weights", False),
            field_name="tracking.default_include_model_weights",
        ),
    )

    if config.sqlite_path.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
        raise ValueError(
            "tracking.sqlite_path must point to a sqlite database file "
            "(.db/.sqlite/.sqlite3)"
        )

    return config
