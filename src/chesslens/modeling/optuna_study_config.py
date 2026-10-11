"""Typed configuration for bounded Optuna HPO over Phase 3 neural training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_DEVICE_VALUES: tuple[str, ...] = ("cpu", "cuda", "auto")


@dataclass(frozen=True)
class HpoReferenceConfig:
    neural_config_path: Path
    modeling_manifest_path_override: Path | None
    baseline_reference_manifest_path_override: Path | None


@dataclass(frozen=True)
class HpoStudyConfig:
    name: str
    output_root: Path
    storage_path: Path
    evaluation_scope: str
    objective_metric: str


@dataclass(frozen=True)
class HpoBudgetConfig:
    max_total_trials: int
    max_new_trials_per_invocation: int


@dataclass(frozen=True)
class HpoSearchRange:
    low: float
    high: float
    log: bool


@dataclass(frozen=True)
class HpoSearchSpaceConfig:
    learning_rate: HpoSearchRange
    weight_decay: HpoSearchRange
    value_weight_choices: tuple[float, ...]


@dataclass(frozen=True)
class HpoRuntimeConfig:
    seed: int
    threads: int
    device: str


@dataclass(frozen=True)
class HpoPruningConfig:
    startup_trials: int
    warmup_epochs: int


@dataclass(frozen=True)
class HpoBehaviorConfig:
    strict: bool


@dataclass(frozen=True)
class HpoExecutionConfig:
    trial_concurrency: int


@dataclass(frozen=True)
class NeuralHpoConfig:
    reference: HpoReferenceConfig
    study: HpoStudyConfig
    budget: HpoBudgetConfig
    search_space: HpoSearchSpaceConfig
    runtime: HpoRuntimeConfig
    pruning: HpoPruningConfig
    behavior: HpoBehaviorConfig
    execution: HpoExecutionConfig


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"HPO config file does not exist: {path.as_posix()}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("HPO config root must be an object")
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


def _as_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def _as_positive_float(value: Any, *, field_name: str) -> float:
    parsed = _as_float(value, field_name=field_name)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _resolve_optional_path(value: Any, *, field_name: str) -> Path | None:
    if value is None:
        return None
    text = _as_non_empty_string(value, field_name=field_name)
    return _resolve_path(text)


def _load_search_range(raw: dict[str, Any], *, prefix: str) -> HpoSearchRange:
    low = _as_positive_float(raw.get("low"), field_name=f"{prefix}.low")
    high = _as_positive_float(raw.get("high"), field_name=f"{prefix}.high")
    if high <= low:
        raise ValueError(f"{prefix}.high must be greater than {prefix}.low")
    return HpoSearchRange(
        low=low,
        high=high,
        log=_as_bool(raw.get("log", True), field_name=f"{prefix}.log"),
    )


def load_neural_hpo_config(path: Path) -> NeuralHpoConfig:
    raw = _load_yaml_dict(path)

    raw_reference = _as_mapping(raw.get("reference"), field_name="reference")
    raw_study = _as_mapping(raw.get("study"), field_name="study")
    raw_budget = _as_mapping(raw.get("budget"), field_name="budget")
    raw_space = _as_mapping(raw.get("search_space"), field_name="search_space")
    raw_runtime = _as_mapping(raw.get("runtime"), field_name="runtime")
    raw_pruning = _as_mapping(raw.get("pruning"), field_name="pruning")
    raw_behavior = _as_mapping(raw.get("behavior"), field_name="behavior")
    raw_execution = _as_mapping(raw.get("execution"), field_name="execution")

    study_name = _as_non_empty_string(
        raw_study.get("name", "phase32_hpo"),
        field_name="study.name",
    )
    study_output_root = _resolve_path(
        _as_non_empty_string(
            raw_study.get("output_root", "data/neural_hpo"),
            field_name="study.output_root",
        )
    )

    default_storage = study_output_root / "studies" / study_name / "study.sqlite3"
    storage_value = raw_study.get("storage_path", default_storage.as_posix())

    learning_rate_raw = _as_mapping(
        raw_space.get("learning_rate"),
        field_name="search_space.learning_rate",
    )
    weight_decay_raw = _as_mapping(
        raw_space.get("weight_decay"),
        field_name="search_space.weight_decay",
    )
    value_weight_raw = raw_space.get("value_weight_choices", [0.2, 0.4, 0.6])
    if not isinstance(value_weight_raw, list) or not value_weight_raw:
        raise ValueError("search_space.value_weight_choices must be a non-empty list")

    value_weight_choices: list[float] = []
    for index, item in enumerate(value_weight_raw):
        parsed = _as_positive_float(
            item,
            field_name=f"search_space.value_weight_choices[{index}]",
        )
        value_weight_choices.append(parsed)

    if len(set(value_weight_choices)) != len(value_weight_choices):
        raise ValueError("search_space.value_weight_choices must be unique")

    runtime_device = _as_non_empty_string(
        raw_runtime.get("device", "cpu"),
        field_name="runtime.device",
    )
    if runtime_device not in SUPPORTED_DEVICE_VALUES:
        raise ValueError(
            "runtime.device must be one of "
            + ", ".join(SUPPORTED_DEVICE_VALUES)
            + f"; got {runtime_device!r}"
        )

    config = NeuralHpoConfig(
        reference=HpoReferenceConfig(
            neural_config_path=_resolve_path(
                _as_non_empty_string(
                    raw_reference.get("neural_config_path"),
                    field_name="reference.neural_config_path",
                )
            ),
            modeling_manifest_path_override=_resolve_optional_path(
                raw_reference.get("modeling_manifest_path_override"),
                field_name="reference.modeling_manifest_path_override",
            ),
            baseline_reference_manifest_path_override=_resolve_optional_path(
                raw_reference.get("baseline_reference_manifest_path_override"),
                field_name="reference.baseline_reference_manifest_path_override",
            ),
        ),
        study=HpoStudyConfig(
            name=study_name,
            output_root=study_output_root,
            storage_path=_resolve_path(
                _as_non_empty_string(storage_value, field_name="study.storage_path")
            ),
            evaluation_scope=_as_non_empty_string(
                raw_study.get("evaluation_scope", "validation_only"),
                field_name="study.evaluation_scope",
            ),
            objective_metric=_as_non_empty_string(
                raw_study.get("objective_metric", "validation.policy.mrr"),
                field_name="study.objective_metric",
            ),
        ),
        budget=HpoBudgetConfig(
            max_total_trials=_as_positive_int(
                raw_budget.get("max_total_trials", 8),
                field_name="budget.max_total_trials",
            ),
            max_new_trials_per_invocation=_as_positive_int(
                raw_budget.get("max_new_trials_per_invocation", 2),
                field_name="budget.max_new_trials_per_invocation",
            ),
        ),
        search_space=HpoSearchSpaceConfig(
            learning_rate=_load_search_range(
                learning_rate_raw,
                prefix="search_space.learning_rate",
            ),
            weight_decay=_load_search_range(
                weight_decay_raw,
                prefix="search_space.weight_decay",
            ),
            value_weight_choices=tuple(sorted(value_weight_choices)),
        ),
        runtime=HpoRuntimeConfig(
            seed=_as_non_negative_int(raw_runtime.get("seed", 20240131), field_name="runtime.seed"),
            threads=_as_positive_int(raw_runtime.get("threads", 1), field_name="runtime.threads"),
            device=runtime_device,
        ),
        pruning=HpoPruningConfig(
            startup_trials=_as_non_negative_int(
                raw_pruning.get("startup_trials", 1),
                field_name="pruning.startup_trials",
            ),
            warmup_epochs=_as_non_negative_int(
                raw_pruning.get("warmup_epochs", 1),
                field_name="pruning.warmup_epochs",
            ),
        ),
        behavior=HpoBehaviorConfig(
            strict=_as_bool(raw_behavior.get("strict", True), field_name="behavior.strict")
        ),
        execution=HpoExecutionConfig(
            trial_concurrency=_as_positive_int(
                raw_execution.get("trial_concurrency", 1),
                field_name="execution.trial_concurrency",
            )
        ),
    )

    if config.study.evaluation_scope != "validation_only":
        raise ValueError(
            "study.evaluation_scope must be 'validation_only' for Phase 3.2 HPO safety"
        )

    if config.study.objective_metric != "validation.policy.mrr":
        raise ValueError(
            "study.objective_metric must be 'validation.policy.mrr' for current implementation"
        )

    if config.execution.trial_concurrency != 1:
        raise ValueError(
            "execution.trial_concurrency must be 1; parallel trial execution is not supported"
        )

    return config


def hpo_config_identity_payload(
    *,
    config: NeuralHpoConfig,
    reference_neural_identity_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "reference_neural_identity": reference_neural_identity_payload,
        "study": {
            "name": config.study.name,
            "evaluation_scope": config.study.evaluation_scope,
            "objective_metric": config.study.objective_metric,
        },
        "budget": {
            "max_total_trials": config.budget.max_total_trials,
            "max_new_trials_per_invocation": config.budget.max_new_trials_per_invocation,
        },
        "search_space": {
            "learning_rate": {
                "low": config.search_space.learning_rate.low,
                "high": config.search_space.learning_rate.high,
                "log": config.search_space.learning_rate.log,
            },
            "weight_decay": {
                "low": config.search_space.weight_decay.low,
                "high": config.search_space.weight_decay.high,
                "log": config.search_space.weight_decay.log,
            },
            "value_weight_choices": list(config.search_space.value_weight_choices),
        },
        "runtime": {
            "seed": config.runtime.seed,
            "threads": config.runtime.threads,
            "device": config.runtime.device,
        },
        "execution": {
            "trial_concurrency": config.execution.trial_concurrency,
        },
        "pruning": {
            "startup_trials": config.pruning.startup_trials,
            "warmup_epochs": config.pruning.warmup_epochs,
        },
    }
