"""Typed configuration loader for Phase 2.1 modeling datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from chesslens.domain.records import (
    ACTION_ENCODING_VERSION,
    BOARD_ENCODING_VERSION,
    POSITION_NORMALIZATION_VERSION,
    SCHEMA_VERSION,
)
from chesslens.modeling.contracts import (
    FEATURE_SCHEMA_VERSION,
    LABEL_DEFINITION_VERSION,
    MODELING_PIPELINE_VERSION,
    PLAYER_HOLDOUT_RULE_VERSION,
    SAMPLING_RULE_VERSION,
    SPLIT_DEFINITION_VERSION,
)
from chesslens.modeling.splits import DatePolicy, TemporalRange, validate_temporal_ranges

MissingPlayerHashPolicy = Literal[
    "exclude_from_player_disjoint_training",
    "include_in_player_disjoint_training",
]


@dataclass(frozen=True)
class ModelingInputConfig:
    collection_root: Path
    duckdb_path: Path
    warehouse_provenance_path: Path
    date_enrichment_manifest_path: Path | None
    require_date_enrichment: bool
    expected_collection_id: str | None
    move_context_relation: str
    games_relation: str


@dataclass(frozen=True)
class ModelingVersionConfig:
    modeling_pipeline_version: str
    split_definition_version: str
    feature_schema_version: str
    label_definition_version: str
    schema_version: str
    position_normalization_version: str
    board_encoding_version: str
    action_encoding_version: str


@dataclass(frozen=True)
class ModelingSamplingConfig:
    rule_version: str
    seed: str
    hash_modulus: int
    hash_threshold: int
    requested_rate_percent: float
    max_games: int | None


@dataclass(frozen=True)
class TemporalSplitConfig:
    train: TemporalRange
    validation: TemporalRange
    test: TemporalRange
    missing_or_invalid_date_policy: DatePolicy


@dataclass(frozen=True)
class PlayerHoldoutConfig:
    rule_version: str
    seed: str
    hash_modulus: int
    hash_threshold: int
    requested_rate_percent: float
    missing_player_hash_policy: MissingPlayerHashPolicy


@dataclass(frozen=True)
class ModelingOutputConfig:
    output_root: Path
    batch_rows: int
    max_examples: int | None
    parquet_compression: str
    parquet_row_group_size: int


@dataclass(frozen=True)
class ModelingBehaviorConfig:
    strict: bool


@dataclass(frozen=True)
class ModelingConfig:
    input: ModelingInputConfig
    versions: ModelingVersionConfig
    sampling: ModelingSamplingConfig
    splits: TemporalSplitConfig
    player_holdout: PlayerHoldoutConfig
    output: ModelingOutputConfig
    behavior: ModelingBehaviorConfig


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Modeling config root must be a mapping")
    return dict(loaded)


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


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


def _as_non_empty_string(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _as_positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _as_non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _as_optional_non_negative_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_non_negative_int(value, field_name=field_name)


def _as_rate_percent(value: Any, *, field_name: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 100.0:
        raise ValueError(f"{field_name} must be in [0, 100]")
    return parsed


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _resolve_optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return _resolve_path(value)


def _parse_date(value: Any, *, field_name: str) -> datetime:
    text = _as_non_empty_string(value, field_name=field_name)
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be in YYYY-MM-DD format") from exc


def _validate_supported_version(
    *,
    field_name: str,
    configured: str,
    supported: str,
) -> str:
    if configured != supported:
        raise ValueError(
            f"Unsupported {field_name}: {configured!r}. Supported value is {supported!r}."
        )
    return configured


def _build_temporal_range(payload: Mapping[str, Any], *, prefix: str) -> TemporalRange:
    start_dt = _parse_date(payload.get("start_date"), field_name=f"{prefix}.start_date")
    end_dt = _parse_date(payload.get("end_date"), field_name=f"{prefix}.end_date")
    return TemporalRange(start_date=start_dt.date(), end_date=end_dt.date())


def load_modeling_config(path: str | Path) -> ModelingConfig:
    raw = _load_yaml_dict(Path(path))

    raw_input = _as_mapping(raw.get("input"), field_name="input")
    raw_versions = _as_mapping(raw.get("versions"), field_name="versions")
    raw_sampling = _as_mapping(raw.get("sampling"), field_name="sampling")
    raw_splits = _as_mapping(raw.get("splits"), field_name="splits")
    raw_holdout = _as_mapping(raw.get("player_holdout"), field_name="player_holdout")
    raw_output = _as_mapping(raw.get("output"), field_name="output")
    raw_behavior = _as_mapping(raw.get("behavior"), field_name="behavior")

    input_config = ModelingInputConfig(
        collection_root=_resolve_path(
            _as_non_empty_string(
                raw_input.get("collection_root", "data/processed/collections"),
                field_name="input.collection_root",
            )
        ),
        duckdb_path=_resolve_path(
            _as_non_empty_string(
                raw_input.get("duckdb_path", "data/tmp/chesslens_warehouse.duckdb"),
                field_name="input.duckdb_path",
            )
        ),
        warehouse_provenance_path=_resolve_path(
            _as_non_empty_string(
                raw_input.get("warehouse_provenance_path", ""),
                field_name="input.warehouse_provenance_path",
            )
        ),
        date_enrichment_manifest_path=_resolve_optional_path(
            _as_optional_string(raw_input.get("date_enrichment_manifest_path"))
        ),
        require_date_enrichment=_as_bool(
            raw_input.get("require_date_enrichment", False),
            field_name="input.require_date_enrichment",
        ),
        expected_collection_id=_as_optional_string(raw_input.get("expected_collection_id")),
        move_context_relation=_as_non_empty_string(
            raw_input.get("move_context_relation", "main.int_move_context"),
            field_name="input.move_context_relation",
        ),
        games_relation=_as_non_empty_string(
            raw_input.get("games_relation", "main.stg_games"),
            field_name="input.games_relation",
        ),
    )

    versions = ModelingVersionConfig(
        modeling_pipeline_version=_validate_supported_version(
            field_name="versions.modeling_pipeline_version",
            configured=_as_non_empty_string(
                raw_versions.get("modeling_pipeline_version", MODELING_PIPELINE_VERSION),
                field_name="versions.modeling_pipeline_version",
            ),
            supported=MODELING_PIPELINE_VERSION,
        ),
        split_definition_version=_validate_supported_version(
            field_name="versions.split_definition_version",
            configured=_as_non_empty_string(
                raw_versions.get("split_definition_version", SPLIT_DEFINITION_VERSION),
                field_name="versions.split_definition_version",
            ),
            supported=SPLIT_DEFINITION_VERSION,
        ),
        feature_schema_version=_validate_supported_version(
            field_name="versions.feature_schema_version",
            configured=_as_non_empty_string(
                raw_versions.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
                field_name="versions.feature_schema_version",
            ),
            supported=FEATURE_SCHEMA_VERSION,
        ),
        label_definition_version=_validate_supported_version(
            field_name="versions.label_definition_version",
            configured=_as_non_empty_string(
                raw_versions.get("label_definition_version", LABEL_DEFINITION_VERSION),
                field_name="versions.label_definition_version",
            ),
            supported=LABEL_DEFINITION_VERSION,
        ),
        schema_version=_validate_supported_version(
            field_name="versions.schema_version",
            configured=_as_non_empty_string(
                raw_versions.get("schema_version", SCHEMA_VERSION),
                field_name="versions.schema_version",
            ),
            supported=SCHEMA_VERSION,
        ),
        position_normalization_version=_validate_supported_version(
            field_name="versions.position_normalization_version",
            configured=_as_non_empty_string(
                raw_versions.get(
                    "position_normalization_version",
                    POSITION_NORMALIZATION_VERSION,
                ),
                field_name="versions.position_normalization_version",
            ),
            supported=POSITION_NORMALIZATION_VERSION,
        ),
        board_encoding_version=_validate_supported_version(
            field_name="versions.board_encoding_version",
            configured=_as_non_empty_string(
                raw_versions.get("board_encoding_version", BOARD_ENCODING_VERSION),
                field_name="versions.board_encoding_version",
            ),
            supported=BOARD_ENCODING_VERSION,
        ),
        action_encoding_version=_validate_supported_version(
            field_name="versions.action_encoding_version",
            configured=_as_non_empty_string(
                raw_versions.get("action_encoding_version", ACTION_ENCODING_VERSION),
                field_name="versions.action_encoding_version",
            ),
            supported=ACTION_ENCODING_VERSION,
        ),
    )

    sampling = ModelingSamplingConfig(
        rule_version=_validate_supported_version(
            field_name="sampling.rule_version",
            configured=_as_non_empty_string(
                raw_sampling.get("rule_version", SAMPLING_RULE_VERSION),
                field_name="sampling.rule_version",
            ),
            supported=SAMPLING_RULE_VERSION,
        ),
        seed=_as_non_empty_string(
            raw_sampling.get("seed", "phase2-default-seed"), field_name="sampling.seed"
        ),
        hash_modulus=_as_positive_int(
            raw_sampling.get("hash_modulus", 10000),
            field_name="sampling.hash_modulus",
        ),
        hash_threshold=_as_non_negative_int(
            raw_sampling.get("hash_threshold", 10000),
            field_name="sampling.hash_threshold",
        ),
        requested_rate_percent=_as_rate_percent(
            raw_sampling.get("requested_rate_percent", 100.0),
            field_name="sampling.requested_rate_percent",
        ),
        max_games=_as_optional_non_negative_int(
            raw_sampling.get("max_games"),
            field_name="sampling.max_games",
        ),
    )
    if sampling.hash_threshold > sampling.hash_modulus:
        raise ValueError("sampling.hash_threshold must be <= sampling.hash_modulus")

    train_range = _build_temporal_range(
        _as_mapping(raw_splits.get("train"), field_name="splits.train"),
        prefix="splits.train",
    )
    validation_range = _build_temporal_range(
        _as_mapping(raw_splits.get("validation"), field_name="splits.validation"),
        prefix="splits.validation",
    )
    test_range = _build_temporal_range(
        _as_mapping(raw_splits.get("test"), field_name="splits.test"),
        prefix="splits.test",
    )

    policy = _as_non_empty_string(
        raw_splits.get("missing_or_invalid_date_policy", "reject"),
        field_name="splits.missing_or_invalid_date_policy",
    )
    if policy not in {"reject", "assign_train", "assign_validation", "assign_test"}:
        raise ValueError(
            "splits.missing_or_invalid_date_policy must be one of: "
            "reject, assign_train, assign_validation, assign_test"
        )

    splits = TemporalSplitConfig(
        train=train_range,
        validation=validation_range,
        test=test_range,
        missing_or_invalid_date_policy=policy,  # type: ignore[arg-type]
    )
    validate_temporal_ranges(train=splits.train, validation=splits.validation, test=splits.test)

    holdout_policy = _as_non_empty_string(
        raw_holdout.get(
            "missing_player_hash_policy",
            "exclude_from_player_disjoint_training",
        ),
        field_name="player_holdout.missing_player_hash_policy",
    )
    if holdout_policy not in {
        "exclude_from_player_disjoint_training",
        "include_in_player_disjoint_training",
    }:
        raise ValueError(
            "player_holdout.missing_player_hash_policy must be one of: "
            "exclude_from_player_disjoint_training, include_in_player_disjoint_training"
        )

    player_holdout = PlayerHoldoutConfig(
        rule_version=_validate_supported_version(
            field_name="player_holdout.rule_version",
            configured=_as_non_empty_string(
                raw_holdout.get("rule_version", PLAYER_HOLDOUT_RULE_VERSION),
                field_name="player_holdout.rule_version",
            ),
            supported=PLAYER_HOLDOUT_RULE_VERSION,
        ),
        seed=_as_non_empty_string(
            raw_holdout.get("seed", "phase2-player-holdout-seed"),
            field_name="player_holdout.seed",
        ),
        hash_modulus=_as_positive_int(
            raw_holdout.get("hash_modulus", 10000),
            field_name="player_holdout.hash_modulus",
        ),
        hash_threshold=_as_non_negative_int(
            raw_holdout.get("hash_threshold", 1000),
            field_name="player_holdout.hash_threshold",
        ),
        requested_rate_percent=_as_rate_percent(
            raw_holdout.get("requested_rate_percent", 10.0),
            field_name="player_holdout.requested_rate_percent",
        ),
        missing_player_hash_policy=holdout_policy,  # type: ignore[arg-type]
    )
    if player_holdout.hash_threshold > player_holdout.hash_modulus:
        raise ValueError("player_holdout.hash_threshold must be <= player_holdout.hash_modulus")

    output = ModelingOutputConfig(
        output_root=_resolve_path(
            _as_non_empty_string(
                raw_output.get("output_root", "data/modeling"),
                field_name="output.output_root",
            )
        ),
        batch_rows=_as_positive_int(
            raw_output.get("batch_rows", 50000),
            field_name="output.batch_rows",
        ),
        max_examples=_as_optional_non_negative_int(
            raw_output.get("max_examples"),
            field_name="output.max_examples",
        ),
        parquet_compression=_as_non_empty_string(
            raw_output.get("parquet_compression", "zstd"),
            field_name="output.parquet_compression",
        ).lower(),
        parquet_row_group_size=_as_positive_int(
            raw_output.get("parquet_row_group_size", 10000),
            field_name="output.parquet_row_group_size",
        ),
    )

    behavior = ModelingBehaviorConfig(
        strict=_as_bool(raw_behavior.get("strict", True), field_name="behavior.strict")
    )

    return ModelingConfig(
        input=input_config,
        versions=versions,
        sampling=sampling,
        splits=splits,
        player_holdout=player_holdout,
        output=output,
        behavior=behavior,
    )


def apply_cli_overrides(
    config: ModelingConfig,
    *,
    collection_root: str | None,
    output_root: str | None,
    duckdb_path: str | None,
    warehouse_provenance_path: str | None,
    date_enrichment_manifest_path: str | None = None,
    max_games: int | None,
    max_examples: int | None,
) -> ModelingConfig:
    new_input = config.input
    if collection_root is not None:
        new_input = replace(new_input, collection_root=_resolve_path(collection_root))
    if duckdb_path is not None:
        new_input = replace(new_input, duckdb_path=_resolve_path(duckdb_path))
    if warehouse_provenance_path is not None:
        new_input = replace(
            new_input,
            warehouse_provenance_path=_resolve_path(warehouse_provenance_path),
        )
    if date_enrichment_manifest_path is not None:
        new_input = replace(
            new_input,
            date_enrichment_manifest_path=_resolve_path(date_enrichment_manifest_path),
        )

    new_sampling = config.sampling
    if max_games is not None:
        if max_games < 0:
            raise ValueError("--max-games must be non-negative")
        new_sampling = replace(new_sampling, max_games=max_games)

    new_output = config.output
    if output_root is not None:
        new_output = replace(new_output, output_root=_resolve_path(output_root))
    if max_examples is not None:
        if max_examples < 0:
            raise ValueError("--max-examples must be non-negative")
        new_output = replace(new_output, max_examples=max_examples)

    return replace(config, input=new_input, sampling=new_sampling, output=new_output)
