"""Typed configuration loader for Phase 3.1 supervised neural training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from chesslens.domain.records import ACTION_ENCODING_VERSION, BOARD_ENCODING_VERSION
from chesslens.features.position_encoding import POSITION_NORMALIZATION_VERSION
from chesslens.modeling.contracts import FEATURE_SCHEMA_VERSION, SPLIT_DEFINITION_VERSION

NEURAL_PIPELINE_VERSION = "neural_supervised_v1"
RATING_BAND_DEFINITION_VERSION = "rating_band_v1"
TIME_CONTROL_CATEGORY_DEFINITION_VERSION = "time_control_category_v1"

SUPPORTED_TRAINING_POPULATION_MODES: tuple[str, ...] = (
    "temporal_all",
    "player_disjoint",
)
SUPPORTED_DEVICE_VALUES: tuple[str, ...] = ("cpu", "cuda", "auto")


@dataclass(frozen=True)
class NeuralInputConfig:
    modeling_manifest_path: Path
    baseline_reference_manifest_path: Path | None


@dataclass(frozen=True)
class NeuralOutputConfig:
    output_root: Path
    prediction_artifact_row_limit: int
    progress_update_seconds: int


@dataclass(frozen=True)
class NeuralSelectionConfig:
    training_population_mode: str
    max_train_positions: int | None
    max_validation_positions: int | None
    max_test_positions: int | None


@dataclass(frozen=True)
class NeuralRuntimeConfig:
    seed: int
    threads: int
    num_workers: int
    device: str


@dataclass(frozen=True)
class NeuralModelConfig:
    trunk_channels: int
    residual_blocks: int
    context_embedding_dim: int
    context_hidden_dim: int
    value_hidden_dim: int


@dataclass(frozen=True)
class NeuralOptimizationConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float


@dataclass(frozen=True)
class NeuralTrainingConfig:
    max_epochs: int
    early_stopping_patience: int
    selection_metric: str


@dataclass(frozen=True)
class NeuralLossConfig:
    policy_weight: float
    value_weight: float


@dataclass(frozen=True)
class NeuralEvaluationConfig:
    ranking_cutoffs: tuple[int, ...]
    ece_bins: int


@dataclass(frozen=True)
class NeuralVersionConfig:
    neural_pipeline_version: str
    feature_schema_version: str
    split_definition_version: str
    board_encoding_version: str
    action_encoding_version: str
    position_normalization_version: str
    rating_band_definition_version: str
    time_control_category_definition_version: str


@dataclass(frozen=True)
class NeuralBehaviorConfig:
    strict: bool


@dataclass(frozen=True)
class NeuralConfig:
    input: NeuralInputConfig
    output: NeuralOutputConfig
    selection: NeuralSelectionConfig
    runtime: NeuralRuntimeConfig
    model: NeuralModelConfig
    optimization: NeuralOptimizationConfig
    training: NeuralTrainingConfig
    loss: NeuralLossConfig
    evaluation: NeuralEvaluationConfig
    versions: NeuralVersionConfig
    behavior: NeuralBehaviorConfig


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path.as_posix()}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Neural config root must be an object")
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


def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
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


def _as_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _as_non_negative_int(value: Any, *, field_name: str) -> int:
    parsed = _as_int(value, field_name=field_name)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _as_positive_int(value: Any, *, field_name: str) -> int:
    parsed = _as_int(value, field_name=field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _as_optional_non_negative_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_non_negative_int(value, field_name=field_name)


def _as_non_negative_float(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if parsed < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _as_positive_float(value: Any, *, field_name: str) -> float:
    parsed = _as_non_negative_float(value, field_name=field_name)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _as_enum_value(value: Any, *, field_name: str, allowed: tuple[str, ...]) -> str:
    parsed = _as_non_empty_string(value, field_name=field_name)
    if parsed not in allowed:
        raise ValueError(
            f"{field_name} must be one of {', '.join(allowed)}; got {parsed!r}"
        )
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


def _validate_loss_weights(config: NeuralLossConfig) -> None:
    if config.policy_weight < 0.0 or config.value_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if (config.policy_weight + config.value_weight) <= 0.0:
        raise ValueError("loss weights must have positive total")


def _validate_cutoffs(cutoffs: list[Any]) -> tuple[int, ...]:
    parsed = tuple(
        _as_positive_int(item, field_name="evaluation.ranking_cutoffs[]") for item in cutoffs
    )
    if not parsed:
        raise ValueError("evaluation.ranking_cutoffs must be non-empty")
    if tuple(sorted(set(parsed))) != parsed:
        raise ValueError("evaluation.ranking_cutoffs must be strictly increasing and unique")
    return parsed


def load_neural_config(path: Path) -> NeuralConfig:
    raw = _load_yaml_dict(path)

    raw_input = _as_mapping(raw.get("input"), field_name="input")
    raw_output = _as_mapping(raw.get("output"), field_name="output")
    raw_selection = _as_mapping(raw.get("selection"), field_name="selection")
    raw_runtime = _as_mapping(raw.get("runtime"), field_name="runtime")
    raw_model = _as_mapping(raw.get("model"), field_name="model")
    raw_optim = _as_mapping(raw.get("optimization"), field_name="optimization")
    raw_training = _as_mapping(raw.get("training"), field_name="training")
    raw_loss = _as_mapping(raw.get("loss"), field_name="loss")
    raw_eval = _as_mapping(raw.get("evaluation"), field_name="evaluation")
    raw_versions = _as_mapping(raw.get("versions"), field_name="versions")
    raw_behavior = _as_mapping(raw.get("behavior"), field_name="behavior")

    input_config = NeuralInputConfig(
        modeling_manifest_path=_resolve_path(
            _as_non_empty_string(
                raw_input.get(
                    "modeling_manifest_path",
                    "data/modeling/datasets/fixture/_manifest.json",
                ),
                field_name="input.modeling_manifest_path",
            )
        ),
        baseline_reference_manifest_path=_resolve_optional_path(
            _as_optional_string(raw_input.get("baseline_reference_manifest_path"))
        ),
    )

    output_config = NeuralOutputConfig(
        output_root=_resolve_path(
            _as_non_empty_string(
                raw_output.get("output_root", "data/modeling/neural"),
                field_name="output.output_root",
            )
        ),
        prediction_artifact_row_limit=_as_positive_int(
            raw_output.get("prediction_artifact_row_limit", 1000),
            field_name="output.prediction_artifact_row_limit",
        ),
        progress_update_seconds=_as_positive_int(
            raw_output.get("progress_update_seconds", 10),
            field_name="output.progress_update_seconds",
        ),
    )

    selection_config = NeuralSelectionConfig(
        training_population_mode=_as_enum_value(
            raw_selection.get("training_population_mode", "player_disjoint"),
            field_name="selection.training_population_mode",
            allowed=SUPPORTED_TRAINING_POPULATION_MODES,
        ),
        max_train_positions=_as_optional_non_negative_int(
            raw_selection.get("max_train_positions"),
            field_name="selection.max_train_positions",
        ),
        max_validation_positions=_as_optional_non_negative_int(
            raw_selection.get("max_validation_positions"),
            field_name="selection.max_validation_positions",
        ),
        max_test_positions=_as_optional_non_negative_int(
            raw_selection.get("max_test_positions"),
            field_name="selection.max_test_positions",
        ),
    )

    runtime_config = NeuralRuntimeConfig(
        seed=_as_non_negative_int(raw_runtime.get("seed", 20240131), field_name="runtime.seed"),
        threads=_as_positive_int(raw_runtime.get("threads", 1), field_name="runtime.threads"),
        num_workers=_as_non_negative_int(
            raw_runtime.get("num_workers", 0),
            field_name="runtime.num_workers",
        ),
        device=_as_enum_value(
            raw_runtime.get("device", "cpu"),
            field_name="runtime.device",
            allowed=SUPPORTED_DEVICE_VALUES,
        ),
    )

    model_config = NeuralModelConfig(
        trunk_channels=_as_positive_int(
            raw_model.get("trunk_channels", 32),
            field_name="model.trunk_channels",
        ),
        residual_blocks=_as_positive_int(
            raw_model.get("residual_blocks", 2),
            field_name="model.residual_blocks",
        ),
        context_embedding_dim=_as_positive_int(
            raw_model.get("context_embedding_dim", 8),
            field_name="model.context_embedding_dim",
        ),
        context_hidden_dim=_as_positive_int(
            raw_model.get("context_hidden_dim", 32),
            field_name="model.context_hidden_dim",
        ),
        value_hidden_dim=_as_positive_int(
            raw_model.get("value_hidden_dim", 64),
            field_name="model.value_hidden_dim",
        ),
    )

    optimization_config = NeuralOptimizationConfig(
        batch_size=_as_positive_int(
            raw_optim.get("batch_size", 32),
            field_name="optimization.batch_size",
        ),
        learning_rate=_as_positive_float(
            raw_optim.get("learning_rate", 0.001),
            field_name="optimization.learning_rate",
        ),
        weight_decay=_as_non_negative_float(
            raw_optim.get("weight_decay", 0.0001),
            field_name="optimization.weight_decay",
        ),
        grad_clip_norm=_as_non_negative_float(
            raw_optim.get("grad_clip_norm", 1.0),
            field_name="optimization.grad_clip_norm",
        ),
    )

    training_config = NeuralTrainingConfig(
        max_epochs=_as_positive_int(
            raw_training.get("max_epochs", 10),
            field_name="training.max_epochs",
        ),
        early_stopping_patience=_as_non_negative_int(
            raw_training.get("early_stopping_patience", 3),
            field_name="training.early_stopping_patience",
        ),
        selection_metric=_as_non_empty_string(
            raw_training.get("selection_metric", "validation_policy_mrr"),
            field_name="training.selection_metric",
        ),
    )

    loss_config = NeuralLossConfig(
        policy_weight=_as_non_negative_float(
            raw_loss.get("policy_weight", 1.0),
            field_name="loss.policy_weight",
        ),
        value_weight=_as_non_negative_float(
            raw_loss.get("value_weight", 1.0),
            field_name="loss.value_weight",
        ),
    )

    cutoffs_raw = raw_eval.get("ranking_cutoffs", [1, 3, 5])
    if not isinstance(cutoffs_raw, list):
        raise ValueError("evaluation.ranking_cutoffs must be a list")

    eval_config = NeuralEvaluationConfig(
        ranking_cutoffs=_validate_cutoffs(cutoffs_raw),
        ece_bins=_as_positive_int(raw_eval.get("ece_bins", 10), field_name="evaluation.ece_bins"),
    )
    if eval_config.ece_bins < 2:
        raise ValueError("evaluation.ece_bins must be >= 2")

    versions_config = NeuralVersionConfig(
        neural_pipeline_version=_as_non_empty_string(
            raw_versions.get("neural_pipeline_version", NEURAL_PIPELINE_VERSION),
            field_name="versions.neural_pipeline_version",
        ),
        feature_schema_version=_as_non_empty_string(
            raw_versions.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
            field_name="versions.feature_schema_version",
        ),
        split_definition_version=_as_non_empty_string(
            raw_versions.get("split_definition_version", SPLIT_DEFINITION_VERSION),
            field_name="versions.split_definition_version",
        ),
        board_encoding_version=_as_non_empty_string(
            raw_versions.get("board_encoding_version", BOARD_ENCODING_VERSION),
            field_name="versions.board_encoding_version",
        ),
        action_encoding_version=_as_non_empty_string(
            raw_versions.get("action_encoding_version", ACTION_ENCODING_VERSION),
            field_name="versions.action_encoding_version",
        ),
        position_normalization_version=_as_non_empty_string(
            raw_versions.get("position_normalization_version", POSITION_NORMALIZATION_VERSION),
            field_name="versions.position_normalization_version",
        ),
        rating_band_definition_version=_as_non_empty_string(
            raw_versions.get(
                "rating_band_definition_version",
                RATING_BAND_DEFINITION_VERSION,
            ),
            field_name="versions.rating_band_definition_version",
        ),
        time_control_category_definition_version=_as_non_empty_string(
            raw_versions.get(
                "time_control_category_definition_version",
                TIME_CONTROL_CATEGORY_DEFINITION_VERSION,
            ),
            field_name="versions.time_control_category_definition_version",
        ),
    )

    behavior_config = NeuralBehaviorConfig(
        strict=_as_bool(raw_behavior.get("strict", True), field_name="behavior.strict")
    )

    config = NeuralConfig(
        input=input_config,
        output=output_config,
        selection=selection_config,
        runtime=runtime_config,
        model=model_config,
        optimization=optimization_config,
        training=training_config,
        loss=loss_config,
        evaluation=eval_config,
        versions=versions_config,
        behavior=behavior_config,
    )

    _validate_loss_weights(config.loss)

    return config


def apply_neural_cli_overrides(
    config: NeuralConfig,
    *,
    modeling_manifest_path: str | None,
    baseline_reference_manifest_path: str | None,
    output_root: str | None,
    max_train_positions: int | None,
    max_validation_positions: int | None,
    max_test_positions: int | None,
    device: str | None,
) -> NeuralConfig:
    updated = config

    if modeling_manifest_path is not None:
        updated = replace(
            updated,
            input=replace(
                updated.input,
                modeling_manifest_path=_resolve_path(modeling_manifest_path),
            ),
        )

    if baseline_reference_manifest_path is not None:
        updated = replace(
            updated,
            input=replace(
                updated.input,
                baseline_reference_manifest_path=_resolve_path(
                    baseline_reference_manifest_path
                ),
            ),
        )

    if output_root is not None:
        updated = replace(
            updated,
            output=replace(updated.output, output_root=_resolve_path(output_root)),
        )

    if any(
        value is not None
        for value in (max_train_positions, max_validation_positions, max_test_positions)
    ):
        updated = replace(
            updated,
            selection=replace(
                updated.selection,
                max_train_positions=(
                    updated.selection.max_train_positions
                    if max_train_positions is None
                    else _as_non_negative_int(
                        max_train_positions,
                        field_name="selection.max_train_positions",
                    )
                ),
                max_validation_positions=(
                    updated.selection.max_validation_positions
                    if max_validation_positions is None
                    else _as_non_negative_int(
                        max_validation_positions,
                        field_name="selection.max_validation_positions",
                    )
                ),
                max_test_positions=(
                    updated.selection.max_test_positions
                    if max_test_positions is None
                    else _as_non_negative_int(
                        max_test_positions,
                        field_name="selection.max_test_positions",
                    )
                ),
            ),
        )

    if device is not None:
        updated = replace(
            updated,
            runtime=replace(
                updated.runtime,
                device=_as_enum_value(
                    device,
                    field_name="runtime.device",
                    allowed=SUPPORTED_DEVICE_VALUES,
                ),
            ),
        )

    return updated


def neural_config_identity_payload(config: NeuralConfig) -> dict[str, Any]:
    return {
        "input": {
            "modeling_manifest_path": config.input.modeling_manifest_path.as_posix(),
            "baseline_reference_manifest_path": (
                None
                if config.input.baseline_reference_manifest_path is None
                else config.input.baseline_reference_manifest_path.as_posix()
            ),
        },
        "output": {
            "prediction_artifact_row_limit": config.output.prediction_artifact_row_limit,
        },
        "selection": {
            "training_population_mode": config.selection.training_population_mode,
            "max_train_positions": config.selection.max_train_positions,
            "max_validation_positions": config.selection.max_validation_positions,
            "max_test_positions": config.selection.max_test_positions,
        },
        "runtime": {
            "seed": config.runtime.seed,
            "threads": config.runtime.threads,
            "num_workers": config.runtime.num_workers,
            "device": config.runtime.device,
        },
        "model": {
            "trunk_channels": config.model.trunk_channels,
            "residual_blocks": config.model.residual_blocks,
            "context_embedding_dim": config.model.context_embedding_dim,
            "context_hidden_dim": config.model.context_hidden_dim,
            "value_hidden_dim": config.model.value_hidden_dim,
        },
        "optimization": {
            "batch_size": config.optimization.batch_size,
            "learning_rate": config.optimization.learning_rate,
            "weight_decay": config.optimization.weight_decay,
            "grad_clip_norm": config.optimization.grad_clip_norm,
        },
        "training": {
            "max_epochs": config.training.max_epochs,
            "early_stopping_patience": config.training.early_stopping_patience,
            "selection_metric": config.training.selection_metric,
        },
        "loss": {
            "policy_weight": config.loss.policy_weight,
            "value_weight": config.loss.value_weight,
        },
        "evaluation": {
            "ranking_cutoffs": list(config.evaluation.ranking_cutoffs),
            "ece_bins": config.evaluation.ece_bins,
        },
        "versions": {
            "neural_pipeline_version": config.versions.neural_pipeline_version,
            "feature_schema_version": config.versions.feature_schema_version,
            "split_definition_version": config.versions.split_definition_version,
            "board_encoding_version": config.versions.board_encoding_version,
            "action_encoding_version": config.versions.action_encoding_version,
            "position_normalization_version": config.versions.position_normalization_version,
            "rating_band_definition_version": config.versions.rating_band_definition_version,
            "time_control_category_definition_version": (
                config.versions.time_control_category_definition_version
            ),
        },
    }
