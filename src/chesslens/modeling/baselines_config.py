"""Typed configuration loader for Phase 2.2 classical baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from chesslens.modeling.contracts import FEATURE_SCHEMA_VERSION, SPLIT_DEFINITION_VERSION

BASELINE_PIPELINE_VERSION = "classical_baselines_v1"
RATING_BAND_DEFINITION_VERSION = "rating_band_v1"
GAME_PHASE_DEFINITION_VERSION = "game_phase_v1"

SUPPORTED_TRAINING_POPULATION_MODES: tuple[str, ...] = (
    "temporal_all",
    "player_disjoint",
)
SUPPORTED_LEGALITY_SCOPES: tuple[str, ...] = ("selected", "full")
SUPPORTED_FREQUENCY_BACKOFF_LEVELS: tuple[str, ...] = (
    "rating_band_phase",
    "phase",
    "global",
)


@dataclass(frozen=True)
class BaselineInputConfig:
    modeling_manifest_path: Path


@dataclass(frozen=True)
class BaselineOutputConfig:
    output_root: Path
    prediction_artifact_row_limit: int


@dataclass(frozen=True)
class BaselineLimitsConfig:
    max_train_positions: int | None
    max_validation_positions: int | None
    max_test_positions: int | None
    max_negative_candidates_per_train_position: int | None
    evaluate_all_legal_candidates_validation: bool
    evaluate_all_legal_candidates_test: bool


@dataclass(frozen=True)
class BaselineTrainingPopulationConfig:
    mode: str


@dataclass(frozen=True)
class BaselinePreflightConfig:
    legality_scope: str


@dataclass(frozen=True)
class BaselineRuntimeConfig:
    seed: int
    threads: int
    early_stopping_rounds: int


@dataclass(frozen=True)
class BaselineEvalConfig:
    ranking_cutoffs: tuple[int, ...]
    ece_bins: int
    bootstrap_iterations: int


@dataclass(frozen=True)
class BaselineFrequencyConfig:
    backoff_levels: tuple[str, ...]


@dataclass(frozen=True)
class BaselineLogisticConfig:
    c: float
    max_iter: int
    solver: str


@dataclass(frozen=True)
class BaselineLightGBMConfig:
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    feature_fraction: float
    lambda_l2: float


@dataclass(frozen=True)
class BaselineVersionConfig:
    baseline_pipeline_version: str
    feature_schema_version: str
    split_definition_version: str
    rating_band_definition_version: str
    game_phase_definition_version: str


@dataclass(frozen=True)
class BaselineConfig:
    input: BaselineInputConfig
    output: BaselineOutputConfig
    limits: BaselineLimitsConfig
    training_population: BaselineTrainingPopulationConfig
    preflight: BaselinePreflightConfig
    runtime: BaselineRuntimeConfig
    evaluation: BaselineEvalConfig
    frequency: BaselineFrequencyConfig
    logistic: BaselineLogisticConfig
    lightgbm: BaselineLightGBMConfig
    versions: BaselineVersionConfig


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path.as_posix()}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Baseline config root must be an object")
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


def _as_positive_float(value: Any, *, field_name: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _as_fraction(value: Any, *, field_name: str) -> float:
    parsed = float(value)
    if parsed <= 0.0 or parsed > 1.0:
        raise ValueError(f"{field_name} must be in (0, 1]")
    return parsed


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _validate_cutoffs(cutoffs: list[Any]) -> tuple[int, ...]:
    parsed = tuple(
        _as_positive_int(item, field_name="evaluation.ranking_cutoffs[]")
        for item in cutoffs
    )
    if not parsed:
        raise ValueError("evaluation.ranking_cutoffs must be non-empty")
    if tuple(sorted(set(parsed))) != parsed:
        raise ValueError("evaluation.ranking_cutoffs must be strictly increasing and unique")
    return parsed


def _as_enum_value(value: Any, *, field_name: str, allowed: tuple[str, ...]) -> str:
    parsed = _as_non_empty_string(value, field_name=field_name)
    if parsed not in allowed:
        raise ValueError(
            f"{field_name} must be one of {', '.join(allowed)}; got {parsed!r}"
        )
    return parsed


def _validate_frequency_backoff_levels(levels_raw: Any) -> tuple[str, ...]:
    if not isinstance(levels_raw, list) or not levels_raw:
        raise ValueError("frequency.backoff_levels must be a non-empty list")

    levels = tuple(
        _as_non_empty_string(item, field_name="frequency.backoff_levels[]")
        for item in levels_raw
    )
    unsupported = sorted(set(levels) - set(SUPPORTED_FREQUENCY_BACKOFF_LEVELS))
    if unsupported:
        raise ValueError(
            "frequency.backoff_levels contains unsupported level(s): "
            + ", ".join(unsupported)
        )

    duplicates = sorted(level for level in set(levels) if levels.count(level) > 1)
    if duplicates:
        raise ValueError(
            "frequency.backoff_levels contains duplicate level(s): "
            + ", ".join(duplicates)
        )

    if "global" not in levels:
        raise ValueError("frequency.backoff_levels must include 'global' as final fallback")
    if levels[-1] != "global":
        raise ValueError("frequency.backoff_levels must end with 'global'")

    return levels


def load_baseline_config(path: str | Path) -> BaselineConfig:
    raw = _load_yaml_dict(Path(path))

    raw_input = _as_mapping(raw.get("input"), field_name="input")
    raw_output = _as_mapping(raw.get("output"), field_name="output")
    raw_limits = _as_mapping(raw.get("limits"), field_name="limits")
    raw_training_population = _as_mapping(
        raw.get("training_population"), field_name="training_population"
    )
    raw_preflight = _as_mapping(raw.get("preflight"), field_name="preflight")
    raw_runtime = _as_mapping(raw.get("runtime"), field_name="runtime")
    raw_eval = _as_mapping(raw.get("evaluation"), field_name="evaluation")
    raw_frequency = _as_mapping(raw.get("frequency"), field_name="frequency")
    raw_logistic = _as_mapping(raw.get("logistic"), field_name="logistic")
    raw_lgbm = _as_mapping(raw.get("lightgbm"), field_name="lightgbm")
    raw_versions = _as_mapping(raw.get("versions"), field_name="versions")

    input_config = BaselineInputConfig(
        modeling_manifest_path=_resolve_path(
            _as_non_empty_string(
                raw_input.get(
                    "modeling_manifest_path",
                    "data/modeling/datasets/fixture/_manifest.json",
                ),
                field_name="input.modeling_manifest_path",
            )
        )
    )

    output_config = BaselineOutputConfig(
        output_root=_resolve_path(
            _as_non_empty_string(
                raw_output.get("output_root", "data/modeling/baselines"),
                field_name="output.output_root",
            )
        ),
        prediction_artifact_row_limit=_as_positive_int(
            raw_output.get("prediction_artifact_row_limit", 2000),
            field_name="output.prediction_artifact_row_limit",
        ),
    )

    limits_config = BaselineLimitsConfig(
        max_train_positions=_as_optional_non_negative_int(
            raw_limits.get("max_train_positions"),
            field_name="limits.max_train_positions",
        ),
        max_validation_positions=_as_optional_non_negative_int(
            raw_limits.get("max_validation_positions"),
            field_name="limits.max_validation_positions",
        ),
        max_test_positions=_as_optional_non_negative_int(
            raw_limits.get("max_test_positions"),
            field_name="limits.max_test_positions",
        ),
        max_negative_candidates_per_train_position=_as_optional_non_negative_int(
            raw_limits.get("max_negative_candidates_per_train_position", 31),
            field_name="limits.max_negative_candidates_per_train_position",
        ),
        evaluate_all_legal_candidates_validation=_as_bool(
            raw_limits.get("evaluate_all_legal_candidates_validation", True),
            field_name="limits.evaluate_all_legal_candidates_validation",
        ),
        evaluate_all_legal_candidates_test=_as_bool(
            raw_limits.get("evaluate_all_legal_candidates_test", True),
            field_name="limits.evaluate_all_legal_candidates_test",
        ),
    )

    training_population_config = BaselineTrainingPopulationConfig(
        mode=_as_enum_value(
            raw_training_population.get("mode", "temporal_all"),
            field_name="training_population.mode",
            allowed=SUPPORTED_TRAINING_POPULATION_MODES,
        )
    )

    preflight_config = BaselinePreflightConfig(
        legality_scope=_as_enum_value(
            raw_preflight.get("legality_scope", "selected"),
            field_name="preflight.legality_scope",
            allowed=SUPPORTED_LEGALITY_SCOPES,
        )
    )

    runtime_config = BaselineRuntimeConfig(
        seed=_as_non_negative_int(raw_runtime.get("seed", 20261001), field_name="runtime.seed"),
        threads=_as_positive_int(raw_runtime.get("threads", 1), field_name="runtime.threads"),
        early_stopping_rounds=_as_positive_int(
            raw_runtime.get("early_stopping_rounds", 20),
            field_name="runtime.early_stopping_rounds",
        ),
    )

    cutoffs_raw = raw_eval.get("ranking_cutoffs", [1, 3, 5, 10])
    if not isinstance(cutoffs_raw, list):
        raise ValueError("evaluation.ranking_cutoffs must be a list")
    ece_bins = _as_positive_int(raw_eval.get("ece_bins", 10), field_name="evaluation.ece_bins")
    if ece_bins < 2:
        raise ValueError("evaluation.ece_bins must be >= 2")
    eval_config = BaselineEvalConfig(
        ranking_cutoffs=_validate_cutoffs(cutoffs_raw),
        ece_bins=ece_bins,
        bootstrap_iterations=_as_non_negative_int(
            raw_eval.get("bootstrap_iterations", 200),
            field_name="evaluation.bootstrap_iterations",
        ),
    )

    backoff_raw = raw_frequency.get(
        "backoff_levels",
        ["rating_band_phase", "phase", "global"],
    )
    frequency_config = BaselineFrequencyConfig(
        backoff_levels=_validate_frequency_backoff_levels(backoff_raw)
    )

    logistic_config = BaselineLogisticConfig(
        c=_as_positive_float(raw_logistic.get("c", 1.0), field_name="logistic.c"),
        max_iter=_as_positive_int(
            raw_logistic.get("max_iter", 400),
            field_name="logistic.max_iter",
        ),
        solver=_as_non_empty_string(
            raw_logistic.get("solver", "lbfgs"),
            field_name="logistic.solver",
        ),
    )

    lightgbm_config = BaselineLightGBMConfig(
        n_estimators=_as_positive_int(
            raw_lgbm.get("n_estimators", 300),
            field_name="lightgbm.n_estimators",
        ),
        learning_rate=_as_positive_float(
            raw_lgbm.get("learning_rate", 0.05),
            field_name="lightgbm.learning_rate",
        ),
        num_leaves=_as_positive_int(
            raw_lgbm.get("num_leaves", 31),
            field_name="lightgbm.num_leaves",
        ),
        min_data_in_leaf=_as_positive_int(
            raw_lgbm.get("min_data_in_leaf", 20), field_name="lightgbm.min_data_in_leaf"
        ),
        feature_fraction=_as_fraction(
            raw_lgbm.get("feature_fraction", 1.0),
            field_name="lightgbm.feature_fraction",
        ),
        lambda_l2=_as_non_negative_int(
            raw_lgbm.get("lambda_l2", 0),
            field_name="lightgbm.lambda_l2",
        ),
    )

    versions_config = BaselineVersionConfig(
        baseline_pipeline_version=_as_non_empty_string(
            raw_versions.get("baseline_pipeline_version", BASELINE_PIPELINE_VERSION),
            field_name="versions.baseline_pipeline_version",
        ),
        feature_schema_version=_as_non_empty_string(
            raw_versions.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
            field_name="versions.feature_schema_version",
        ),
        split_definition_version=_as_non_empty_string(
            raw_versions.get("split_definition_version", SPLIT_DEFINITION_VERSION),
            field_name="versions.split_definition_version",
        ),
        rating_band_definition_version=_as_non_empty_string(
            raw_versions.get("rating_band_definition_version", RATING_BAND_DEFINITION_VERSION),
            field_name="versions.rating_band_definition_version",
        ),
        game_phase_definition_version=_as_non_empty_string(
            raw_versions.get("game_phase_definition_version", GAME_PHASE_DEFINITION_VERSION),
            field_name="versions.game_phase_definition_version",
        ),
    )

    return BaselineConfig(
        input=input_config,
        output=output_config,
        limits=limits_config,
        training_population=training_population_config,
        preflight=preflight_config,
        runtime=runtime_config,
        evaluation=eval_config,
        frequency=frequency_config,
        logistic=logistic_config,
        lightgbm=lightgbm_config,
        versions=versions_config,
    )


def apply_baseline_cli_overrides(
    config: BaselineConfig,
    *,
    modeling_manifest_path: str | None,
    output_root: str | None,
    max_train_positions: int | None,
    max_validation_positions: int | None,
    max_test_positions: int | None,
    threads: int | None,
    seed: int | None,
) -> BaselineConfig:
    updated = config

    if modeling_manifest_path is not None:
        updated = replace(
            updated,
            input=replace(
                updated.input,
                modeling_manifest_path=_resolve_path(modeling_manifest_path),
            ),
        )

    if output_root is not None:
        updated = replace(
            updated,
            output=replace(updated.output, output_root=_resolve_path(output_root)),
        )

    limits = updated.limits
    if max_train_positions is not None:
        limits = replace(
            limits,
            max_train_positions=_as_optional_non_negative_int(
                max_train_positions,
                field_name="--max-train-positions",
            ),
        )
    if max_validation_positions is not None:
        limits = replace(
            limits,
            max_validation_positions=_as_optional_non_negative_int(
                max_validation_positions,
                field_name="--max-validation-positions",
            ),
        )
    if max_test_positions is not None:
        limits = replace(
            limits,
            max_test_positions=_as_optional_non_negative_int(
                max_test_positions,
                field_name="--max-test-positions",
            ),
        )

    runtime = updated.runtime
    if threads is not None:
        runtime = replace(runtime, threads=_as_positive_int(threads, field_name="--threads"))
    if seed is not None:
        runtime = replace(runtime, seed=_as_non_negative_int(seed, field_name="--seed"))

    updated = replace(updated, limits=limits, runtime=runtime)
    return updated


def baseline_config_identity_payload(config: BaselineConfig) -> dict[str, Any]:
    return {
        "input": {
            "modeling_manifest_path": config.input.modeling_manifest_path.as_posix(),
        },
        "output": {
            "prediction_artifact_row_limit": config.output.prediction_artifact_row_limit,
        },
        "limits": {
            "max_train_positions": config.limits.max_train_positions,
            "max_validation_positions": config.limits.max_validation_positions,
            "max_test_positions": config.limits.max_test_positions,
            "max_negative_candidates_per_train_position": (
                config.limits.max_negative_candidates_per_train_position
            ),
            "evaluate_all_legal_candidates_validation": (
                config.limits.evaluate_all_legal_candidates_validation
            ),
            "evaluate_all_legal_candidates_test": config.limits.evaluate_all_legal_candidates_test,
        },
        "training_population": {
            "mode": config.training_population.mode,
        },
        "preflight": {
            "legality_scope": config.preflight.legality_scope,
        },
        "runtime": {
            "seed": config.runtime.seed,
            "threads": config.runtime.threads,
            "early_stopping_rounds": config.runtime.early_stopping_rounds,
        },
        "evaluation": {
            "ranking_cutoffs": list(config.evaluation.ranking_cutoffs),
            "ece_bins": config.evaluation.ece_bins,
            "bootstrap_iterations": config.evaluation.bootstrap_iterations,
        },
        "frequency": {"backoff_levels": list(config.frequency.backoff_levels)},
        "logistic": {
            "c": config.logistic.c,
            "max_iter": config.logistic.max_iter,
            "solver": config.logistic.solver,
        },
        "lightgbm": {
            "n_estimators": config.lightgbm.n_estimators,
            "learning_rate": config.lightgbm.learning_rate,
            "num_leaves": config.lightgbm.num_leaves,
            "min_data_in_leaf": config.lightgbm.min_data_in_leaf,
            "feature_fraction": config.lightgbm.feature_fraction,
            "lambda_l2": config.lightgbm.lambda_l2,
        },
        "versions": {
            "baseline_pipeline_version": config.versions.baseline_pipeline_version,
            "feature_schema_version": config.versions.feature_schema_version,
            "split_definition_version": config.versions.split_definition_version,
            "rating_band_definition_version": config.versions.rating_band_definition_version,
            "game_phase_definition_version": config.versions.game_phase_definition_version,
        },
    }
