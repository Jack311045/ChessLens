"""Bounded Optuna HPO runner for Phase 3.2 validation-only neural tuning."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import socket
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import optuna
import psutil
import torch
import yaml
from optuna.trial import TrialState

from chesslens.ingestion.shard_manifest import atomic_write_json
from chesslens.modeling.mlflow_tracking import import_manifest_to_mlflow
from chesslens.modeling.neural_config import (
    apply_neural_cli_overrides,
    load_neural_config,
    neural_config_identity_payload,
)
from chesslens.modeling.neural_training import run_neural
from chesslens.modeling.optuna_study_config import (
    NeuralHpoConfig,
    hpo_config_identity_payload,
    load_neural_hpo_config,
)
from chesslens.modeling.validation import canonical_json, read_json_object, sha256_text


@dataclass(frozen=True)
class TrialExecutionResult:
    status: Literal["complete", "pruned", "failed", "interrupted"]
    objective_value: float | None
    experiment_id: str | None
    manifest_path: Path | None
    checkpoint_path: Path | None
    validation_policy_metrics: dict[str, Any] | None
    validation_value_metrics: dict[str, Any] | None
    message: str | None


class _StudyLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._acquired = False
        self._lock_id: str | None = None

    def __enter__(self) -> _StudyLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._path.exists():
            payload = self._read_payload()
            owner_status = _lock_owner_status(payload)
            if owner_status.status == "stale":
                self._path.unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    "Study lock already exists and ownership is "
                    f"{owner_status.status!r}; {owner_status.reason}. "
                    "Conservative lock policy prevents taking over this study lock."
                )

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(str(self._path), flags)
        except FileExistsError as exc:
            raise RuntimeError(
                "Another HPO writer is active for this study: " + self._path.as_posix()
            ) from exc

        lock_id = uuid.uuid4().hex
        payload = _current_lock_payload(lock_id=lock_id)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

        self._acquired = True
        self._lock_id = lock_id
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._acquired:
            payload = self._read_payload()
            if (
                isinstance(payload, dict)
                and payload.get("lock_id") == self._lock_id
            ):
                self._path.unlink(missing_ok=True)
            self._acquired = False
            self._lock_id = None

    def _read_payload(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None


@dataclass(frozen=True)
class _OwnerStatus:
    status: Literal["active", "stale", "uncertain"]
    reason: str


def _as_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _as_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _current_lock_payload(*, lock_id: str) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    cmdline_raw = process.cmdline()
    cmdline = " ".join(cmdline_raw) if isinstance(cmdline_raw, list) else ""

    payload: dict[str, Any] = {
        "lock_id": lock_id,
        "pid": process.pid,
        "pid_create_time_epoch": float(process.create_time()),
        "host": socket.gethostname(),
        "created_at_utc": _utc_now_iso(),
        "cwd": Path.cwd().as_posix(),
        "cmdline": cmdline,
    }
    try:
        payload["username"] = process.username()
    except (psutil.AccessDenied, psutil.ZombieProcess):
        payload["username"] = None
    return payload


def _lock_owner_status(payload: dict[str, Any] | None) -> _OwnerStatus:
    if payload is None:
        return _OwnerStatus(
            status="uncertain",
            reason="lock payload is missing or unreadable",
        )

    host = str(payload.get("host", "")).strip()
    if host and host != socket.gethostname():
        return _OwnerStatus(
            status="uncertain",
            reason=(
                "lock file host differs from current host; cannot safely verify ownership"
            ),
        )

    pid = _as_positive_int(payload.get("pid"))
    pid_create_time = _as_positive_float(payload.get("pid_create_time_epoch"))
    if pid is None or pid_create_time is None:
        return _OwnerStatus(
            status="uncertain",
            reason="lock payload is missing pid or pid_create_time_epoch",
        )

    if not psutil.pid_exists(pid):
        return _OwnerStatus(status="stale", reason=f"pid {pid} is not running")

    try:
        process = psutil.Process(pid)
        live_create_time = float(process.create_time())
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return _OwnerStatus(
            status="uncertain",
            reason=f"unable to query process metadata for pid {pid}",
        )

    if abs(live_create_time - pid_create_time) <= 1e-6:
        return _OwnerStatus(
            status="active",
            reason=f"pid {pid} with matching create-time is still active",
        )

    return _OwnerStatus(
        status="stale",
        reason=(
            "pid exists but create-time mismatch indicates pid reuse; "
            "lock owner is no longer active"
        ),
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    temp_path.write_bytes(payload)
    os.replace(temp_path, path)


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded, resume-safe Optuna HPO trials for ChessLens neural training "
            "using validation-only evaluation scope"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hpo/fixture_smoke.yaml"),
        help="Path to typed HPO config YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and report trial budget without running trials",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report study status, counters, and best trial",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete running trials before starting new ones",
    )
    parser.add_argument(
        "--max-new-trials",
        type=int,
        default=None,
        help="Optional per-invocation cap override for new trial starts",
    )
    parser.add_argument(
        "--export-best-config",
        type=Path,
        default=None,
        help="Write a neural YAML config with best HPO parameters",
    )
    parser.add_argument(
        "--export-final-output-root",
        type=Path,
        default=None,
        help=(
            "Optional output root embedded into exported best config for future final runs. "
            "Defaults to a study-scoped final_runs directory."
        ),
    )
    parser.add_argument(
        "--mlflow-tracking-config",
        type=Path,
        default=None,
        help="Optional MLflow tracking config path for idempotent trial-level tracking",
    )
    parser.add_argument(
        "--mlflow-run-origin",
        choices=("historical", "current"),
        default="current",
        help="Run-origin mode tag used for optional trial-level MLflow logging",
    )
    parser.add_argument(
        "--mlflow-include-model-weights",
        action="store_true",
        help="When MLflow logging is enabled, include model weights artifact",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON payload",
    )
    return parser


def _study_root(config: NeuralHpoConfig) -> Path:
    return config.study.output_root / "studies" / config.study.name


def _study_storage_uri(config: NeuralHpoConfig) -> str:
    return f"sqlite:///{config.study.storage_path.as_posix()}"


def _sampler_state_path(study_root: Path) -> Path:
    return study_root / "sampler_state.pkl"


def _identity_path(study_root: Path) -> Path:
    return study_root / "study_identity.json"


def _reports_root(study_root: Path) -> Path:
    return study_root / "reports"


def _trial_root(study_root: Path, trial_number: int) -> Path:
    return study_root / "trials" / f"trial-{trial_number:04d}"


def _trial_output_root(study_root: Path, trial_number: int) -> Path:
    return _trial_root(study_root, trial_number) / "neural_output"


def _reference_trial_values(config: NeuralHpoConfig) -> tuple[float, float, float]:
    reference = load_neural_config(config.reference.neural_config_path)
    return (
        float(reference.optimization.learning_rate),
        float(reference.optimization.weight_decay),
        float(reference.loss.value_weight),
    )


def _reference_preflight_config_payload(
    config: NeuralHpoConfig,
    study_root: Path,
) -> dict[str, Any]:
    raw = yaml.safe_load(config.reference.neural_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("Reference neural config must parse to a YAML object")

    input_raw = raw.setdefault("input", {})
    if not isinstance(input_raw, dict):
        raise RuntimeError("reference.input must be an object")

    if config.reference.modeling_manifest_path_override is not None:
        input_raw["modeling_manifest_path"] = (
            config.reference.modeling_manifest_path_override.as_posix()
        )
    if config.reference.baseline_reference_manifest_path_override is not None:
        input_raw["baseline_reference_manifest_path"] = (
            config.reference.baseline_reference_manifest_path_override.as_posix()
        )

    runtime_raw = raw.setdefault("runtime", {})
    if not isinstance(runtime_raw, dict):
        raise RuntimeError("reference.runtime must be an object")
    runtime_raw["threads"] = config.runtime.threads
    runtime_raw["device"] = config.runtime.device
    runtime_raw["seed"] = config.runtime.seed

    output_raw = raw.setdefault("output", {})
    if not isinstance(output_raw, dict):
        raise RuntimeError("reference.output must be an object")
    output_raw["output_root"] = (study_root / "preflight" / "neural_output").as_posix()

    return raw


def _load_reference_identity_payload(config: NeuralHpoConfig) -> dict[str, Any]:
    reference_payload = _reference_preflight_config_payload(config, _study_root(config))
    reference_path = _study_root(config) / "preflight" / "reference_effective_config.yaml"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(yaml.safe_dump(reference_payload, sort_keys=False), encoding="utf-8")

    reference_neural = load_neural_config(reference_path)
    reference_neural = apply_neural_cli_overrides(
        reference_neural,
        modeling_manifest_path=(
            None
            if config.reference.modeling_manifest_path_override is None
            else config.reference.modeling_manifest_path_override.as_posix()
        ),
        baseline_reference_manifest_path=(
            None
            if config.reference.baseline_reference_manifest_path_override is None
            else config.reference.baseline_reference_manifest_path_override.as_posix()
        ),
        output_root=None,
        max_train_positions=None,
        max_validation_positions=None,
        max_test_positions=None,
        device=config.runtime.device,
    )
    return neural_config_identity_payload(reference_neural)


def _reference_preflight_identity_payload(
    config: NeuralHpoConfig,
    study_root: Path,
) -> dict[str, Any]:
    preflight_config_path = study_root / "preflight" / "reference_dry_run.yaml"
    payload = _reference_preflight_config_payload(config, study_root)
    preflight_config_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = run_neural(
        config_path=preflight_config_path,
        dry_run=True,
        evaluation_scope="validation_only",
    )
    summary = result.summary or {}
    if str(summary.get("status", "")) != "dry_run":
        raise RuntimeError("Reference preflight dry-run did not return status='dry_run'")

    selection_fingerprints = summary.get("selection_fingerprints")
    counts = summary.get("counts")
    if not isinstance(selection_fingerprints, dict):
        raise RuntimeError("Reference preflight dry-run missing selection_fingerprints")
    if not isinstance(counts, dict):
        raise RuntimeError("Reference preflight dry-run missing counts")

    training_population = counts.get("training_population")
    if not isinstance(training_population, dict):
        raise RuntimeError("Reference preflight dry-run missing training_population")

    baseline_compatibility_raw = summary.get("baseline_compatibility")
    if baseline_compatibility_raw is None:
        baseline_compatibility: dict[str, Any] = {
            "compatible": True,
            "reason": "baseline_reference_not_provided",
        }
    elif isinstance(baseline_compatibility_raw, dict):
        baseline_compatibility = baseline_compatibility_raw
    else:
        raise RuntimeError("Reference preflight dry-run missing baseline_compatibility")
    if baseline_compatibility.get("compatible") is False:
        reason = baseline_compatibility.get("reason")
        raise RuntimeError(
            "Reference preflight baseline compatibility check failed: "
            + str(reason)
        )

    return {
        "reference_neural_experiment_id": result.experiment_id,
        "modeling_manifest_path": summary.get("modeling_manifest_path"),
        "modeling_dataset_id": summary.get("modeling_dataset_id"),
        "selection_fingerprints": selection_fingerprints,
        "training_population": training_population,
        "split_row_counts_selected": counts.get("split_row_counts_selected"),
        "split_row_counts_full": counts.get("split_row_counts_full"),
        "baseline_compatibility": baseline_compatibility,
    }


def _enforce_study_identity(
    config: NeuralHpoConfig,
    study_root: Path,
    reference_preflight_identity: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    reference_identity = _load_reference_identity_payload(config)
    payload = hpo_config_identity_payload(
        config=config,
        reference_neural_identity_payload=reference_identity,
    )
    payload["reference_preflight"] = reference_preflight_identity
    identity_hash = sha256_text(canonical_json(payload))

    identity_path = _identity_path(study_root)
    if identity_path.exists():
        existing = read_json_object(identity_path)
        existing_hash = str(existing.get("identity_hash", "")).strip()
        existing_payload = existing.get("identity_payload")

        if existing_hash != identity_hash or existing_payload != payload:
            raise RuntimeError(
                "HPO study identity mismatch. Existing study was created with a different "
                "reference config/search space/budget. Create a new study name or revert "
                "configuration changes."
            )
        return identity_hash, payload

    atomic_write_json(
        identity_path,
        {
            "identity_hash": identity_hash,
            "identity_payload": payload,
            "created_at_utc": _utc_now_iso(),
        },
    )
    return identity_hash, payload


def _build_sampler(
    config: NeuralHpoConfig,
    sampler_state_path: Path,
) -> optuna.samplers.BaseSampler:
    if sampler_state_path.exists():
        try:
            loaded = pickle.loads(sampler_state_path.read_bytes())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Sampler state is unreadable/corrupt; manual intervention required: "
                + sampler_state_path.as_posix()
            ) from exc

        if not isinstance(loaded, dict) or "sampler" not in loaded:
            raise RuntimeError(
                "Sampler state has incompatible format; manual intervention required: "
                + sampler_state_path.as_posix()
            )
        sampler = loaded.get("sampler")
        if not isinstance(sampler, optuna.samplers.BaseSampler):
            raise RuntimeError(
                "Sampler state payload is incompatible; manual intervention required: "
                + sampler_state_path.as_posix()
            )
        return sampler

    return optuna.samplers.TPESampler(seed=config.runtime.seed)


def _save_sampler_state(sampler: optuna.samplers.BaseSampler, sampler_state_path: Path) -> None:
    payload = {
        "saved_at_utc": _utc_now_iso(),
        "sampler_class": type(sampler).__name__,
        "optuna_version": optuna.__version__,
        "sampler": sampler,
    }
    _atomic_write_bytes(sampler_state_path, pickle.dumps(payload))


def _create_or_load_study(
    *,
    config: NeuralHpoConfig,
    study_root: Path,
) -> tuple[optuna.study.Study, optuna.samplers.BaseSampler]:
    config.study.storage_path.parent.mkdir(parents=True, exist_ok=True)
    sampler_state = _sampler_state_path(study_root)
    sampler = _build_sampler(config, sampler_state)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=config.pruning.startup_trials,
        n_warmup_steps=config.pruning.warmup_epochs,
    )

    study = optuna.create_study(
        study_name=config.study.name,
        storage=_study_storage_uri(config),
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    return study, sampler


def _started_trial_count(study: optuna.study.Study) -> int:
    return sum(
        1
        for trial in study.trials
        if trial.state
        in {TrialState.RUNNING, TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
    )


def _count_by_state(study: optuna.study.Study) -> dict[str, int]:
    counts: dict[str, int] = {
        "RUNNING": 0,
        "COMPLETE": 0,
        "PRUNED": 0,
        "FAIL": 0,
        "WAITING": 0,
    }
    for trial in study.trials:
        key = trial.state.name
        counts[key] = counts.get(key, 0) + 1
    return counts


def _maybe_enqueue_reference_trial(study: optuna.study.Study, config: NeuralHpoConfig) -> None:
    learning_rate, weight_decay, value_weight = _reference_trial_values(config)
    params = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "value_weight": value_weight,
    }

    if not (
        config.search_space.learning_rate.low
        <= params["learning_rate"]
        <= config.search_space.learning_rate.high
    ):
        raise RuntimeError(
            "Reference learning_rate is outside configured search space; "
            "update search_space.learning_rate or the reference config"
        )

    if not (
        config.search_space.weight_decay.low
        <= params["weight_decay"]
        <= config.search_space.weight_decay.high
    ):
        raise RuntimeError(
            "Reference weight_decay is outside configured search space; "
            "update search_space.weight_decay or the reference config"
        )

    if params["value_weight"] not in set(config.search_space.value_weight_choices):
        raise RuntimeError(
            "Reference value_weight is outside configured value_weight_choices; "
            "update search_space.value_weight_choices or the reference config"
        )

    for trial in study.trials:
        if (
            "learning_rate" in trial.params
            and "weight_decay" in trial.params
            and "value_weight" in trial.params
            and float(trial.params["learning_rate"]) == params["learning_rate"]
            and float(trial.params["weight_decay"]) == params["weight_decay"]
            and float(trial.params["value_weight"]) == params["value_weight"]
        ):
            return

    study.enqueue_trial(params)


def _trial_config_payload(
    *,
    config: NeuralHpoConfig,
    trial_number: int,
    learning_rate: float,
    weight_decay: float,
    value_weight: float,
) -> dict[str, Any]:
    raw = yaml.safe_load(config.reference.neural_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("Reference neural config must parse to a YAML object")

    input_raw = raw.setdefault("input", {})
    if not isinstance(input_raw, dict):
        raise RuntimeError("reference.input must be an object")

    if config.reference.modeling_manifest_path_override is not None:
        input_raw["modeling_manifest_path"] = (
            config.reference.modeling_manifest_path_override.as_posix()
        )

    if config.reference.baseline_reference_manifest_path_override is not None:
        input_raw["baseline_reference_manifest_path"] = (
            config.reference.baseline_reference_manifest_path_override.as_posix()
        )

    runtime_raw = raw.setdefault("runtime", {})
    if not isinstance(runtime_raw, dict):
        raise RuntimeError("reference.runtime must be an object")
    runtime_raw["threads"] = config.runtime.threads
    runtime_raw["device"] = config.runtime.device
    runtime_raw["seed"] = config.runtime.seed

    optimization_raw = raw.setdefault("optimization", {})
    if not isinstance(optimization_raw, dict):
        raise RuntimeError("reference.optimization must be an object")
    optimization_raw["learning_rate"] = float(learning_rate)
    optimization_raw["weight_decay"] = float(weight_decay)

    loss_raw = raw.setdefault("loss", {})
    if not isinstance(loss_raw, dict):
        raise RuntimeError("reference.loss must be an object")
    loss_raw["value_weight"] = float(value_weight)

    output_raw = raw.setdefault("output", {})
    if not isinstance(output_raw, dict):
        raise RuntimeError("reference.output must be an object")
    output_raw["output_root"] = _trial_output_root(_study_root(config), trial_number).as_posix()

    return raw


def _write_trial_config(
    *,
    config: NeuralHpoConfig,
    trial_number: int,
    learning_rate: float,
    weight_decay: float,
    value_weight: float,
) -> Path:
    trial_root = _trial_root(_study_root(config), trial_number)
    trial_root.mkdir(parents=True, exist_ok=True)

    payload = _trial_config_payload(
        config=config,
        trial_number=trial_number,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        value_weight=value_weight,
    )

    trial_config_path = trial_root / "trial_config.yaml"
    trial_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    return trial_config_path


def _latest_validation_metric(
    *,
    trial_output_root: Path,
    experiment_id: str,
) -> tuple[int, float]:
    checkpoint_path = trial_output_root / "work" / experiment_id / "checkpoints" / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Trial checkpoint is not a dictionary")

    history = checkpoint.get("history")
    if not isinstance(history, list) or not history:
        raise RuntimeError("Trial checkpoint history is missing or empty")

    latest = history[-1]
    if not isinstance(latest, dict):
        raise RuntimeError("Trial checkpoint history row is invalid")

    epoch = int(latest.get("epoch", len(history) - 1))
    metric = float(latest.get("validation_policy_mrr", 0.0))
    return epoch, metric


def _checkpoint_path_for_trial(trial_output_root: Path, experiment_id: str) -> Path:
    return trial_output_root / "work" / experiment_id / "checkpoints" / "last.pt"


def _objective_and_validation_metrics_from_manifest(
    manifest_path: Path,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    aggregate = read_json_object(manifest_path.parent / "aggregate_metrics.json")

    policy_payload = aggregate.get("policy")
    if not isinstance(policy_payload, dict):
        raise RuntimeError(
            "Completed trial is missing policy metrics in aggregate_metrics.json"
        )

    validation_policy_payload = policy_payload.get("validation")
    validation_policy_metrics = (
        validation_policy_payload if isinstance(validation_policy_payload, dict) else {}
    )

    value_payload = aggregate.get("value")
    validation_value_metrics: dict[str, Any] = {}
    if isinstance(value_payload, dict):
        value_validation = value_payload.get("validation")
        if isinstance(value_validation, dict):
            validation_value_metrics = value_validation

    best_metric = policy_payload.get("best_validation_policy_mrr")
    if isinstance(best_metric, (int, float)):
        return float(best_metric), validation_policy_metrics, validation_value_metrics

    if isinstance(validation_policy_metrics, dict):
        mrr = validation_policy_metrics.get("mrr")
        if isinstance(mrr, (int, float)):
            return float(mrr), validation_policy_metrics, validation_value_metrics

    raise RuntimeError("Unable to resolve objective metric from completed trial artifacts")


def _trial_id_for_number(study: optuna.study.Study, trial_number: int) -> int:
    return int(
        study._storage.get_trial_id_from_study_id_trial_number(  # pyright: ignore[reportPrivateUsage]
            study._study_id,  # pyright: ignore[reportPrivateUsage]
            trial_number,
        )
    )


def _live_trial_for_number(study: optuna.study.Study, trial_number: int) -> optuna.trial.Trial:
    trial_id = _trial_id_for_number(study, trial_number)
    return optuna.trial.Trial(study, trial_id)


def _set_trial_user_attr(
    study: optuna.study.Study,
    trial_number: int,
    key: str,
    value: Any,
) -> None:
    trial_id = _trial_id_for_number(study, trial_number)
    study._storage.set_trial_user_attr(  # pyright: ignore[reportPrivateUsage]
        trial_id,
        key,
        value,
    )


def _mark_trial_failed(study: optuna.study.Study, trial_number: int, reason: str) -> None:
    _set_trial_user_attr(study, trial_number, "failure_reason", reason)
    _set_trial_user_attr(study, trial_number, "failed_at_utc", _utc_now_iso())
    study.tell(trial_number, state=TrialState.FAIL)


def _attach_outcome_metadata(
    study: optuna.study.Study,
    trial_number: int,
    outcome: TrialExecutionResult,
) -> None:
    if outcome.experiment_id is not None:
        _set_trial_user_attr(study, trial_number, "neural_experiment_id", outcome.experiment_id)
    if outcome.manifest_path is not None:
        _set_trial_user_attr(
            study,
            trial_number,
            "neural_manifest_path",
            outcome.manifest_path.as_posix(),
        )
    if outcome.checkpoint_path is not None:
        _set_trial_user_attr(
            study,
            trial_number,
            "neural_checkpoint_path",
            outcome.checkpoint_path.as_posix(),
        )
    if outcome.validation_policy_metrics is not None:
        mrr = outcome.validation_policy_metrics.get("mrr")
        if isinstance(mrr, (int, float)):
            _set_trial_user_attr(study, trial_number, "validation_policy_mrr", float(mrr))
    if outcome.validation_value_metrics is not None:
        accuracy = outcome.validation_value_metrics.get("accuracy")
        macro_f1 = outcome.validation_value_metrics.get("macro_f1")
        ece = outcome.validation_value_metrics.get("ece")
        if isinstance(accuracy, (int, float)):
            _set_trial_user_attr(study, trial_number, "validation_value_accuracy", float(accuracy))
        if isinstance(macro_f1, (int, float)):
            _set_trial_user_attr(study, trial_number, "validation_value_macro_f1", float(macro_f1))
        if isinstance(ece, (int, float)):
            _set_trial_user_attr(study, trial_number, "validation_value_ece", float(ece))
    if outcome.message is not None:
        _set_trial_user_attr(study, trial_number, "last_message", outcome.message)


def _run_trial(
    *,
    config: NeuralHpoConfig,
    trial_number: int,
    learning_rate: float,
    weight_decay: float,
    value_weight: float,
    pruning_trial: optuna.trial.Trial | None,
    resume: bool,
) -> TrialExecutionResult:
    trial_config_path = _write_trial_config(
        config=config,
        trial_number=trial_number,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        value_weight=value_weight,
    )

    trial_output_root = _trial_output_root(_study_root(config), trial_number)
    resume_flag = resume
    experiment_id: str | None = None

    while True:
        try:
            result = run_neural(
                config_path=trial_config_path,
                resume=resume_flag,
                max_epochs_this_session=(1 if pruning_trial is not None else None),
                evaluation_scope="validation_only",
            )
        except Exception as exc:  # noqa: BLE001
            checkpoint_path: Path | None = None
            if experiment_id is not None:
                candidate = _checkpoint_path_for_trial(trial_output_root, experiment_id)
                if candidate.exists():
                    checkpoint_path = candidate
            return TrialExecutionResult(
                status="failed",
                objective_value=None,
                experiment_id=experiment_id,
                manifest_path=None,
                checkpoint_path=checkpoint_path,
                validation_policy_metrics=None,
                validation_value_metrics=None,
                message=f"Neural trial execution failed: {exc}",
            )

        experiment_id = result.experiment_id
        checkpoint_path = _checkpoint_path_for_trial(trial_output_root, experiment_id)
        if result.completed:
            objective, policy_metrics, value_metrics = (
                _objective_and_validation_metrics_from_manifest(result.manifest_path)
            )
            return TrialExecutionResult(
                status="complete",
                objective_value=objective,
                experiment_id=experiment_id,
                manifest_path=result.manifest_path,
                checkpoint_path=(checkpoint_path if checkpoint_path.exists() else None),
                validation_policy_metrics=policy_metrics,
                validation_value_metrics=value_metrics,
                message=None,
            )

        summary = result.summary or {}
        status = str(summary.get("status", "")).strip()
        if status == "interrupted":
            return TrialExecutionResult(
                status="interrupted",
                objective_value=None,
                experiment_id=experiment_id,
                manifest_path=None,
                checkpoint_path=(checkpoint_path if checkpoint_path.exists() else None),
                validation_policy_metrics=None,
                validation_value_metrics=None,
                message="Neural trial run was interrupted; checkpoint preserved for resume",
            )

        if status != "session_limit_reached":
            return TrialExecutionResult(
                status="failed",
                objective_value=None,
                experiment_id=experiment_id,
                manifest_path=None,
                checkpoint_path=(checkpoint_path if checkpoint_path.exists() else None),
                validation_policy_metrics=None,
                validation_value_metrics=None,
                message=f"Neural trial did not complete: status={status!r}",
            )

        try:
            epoch, metric = _latest_validation_metric(
                trial_output_root=trial_output_root,
                experiment_id=experiment_id,
            )
        except Exception as exc:  # noqa: BLE001
            return TrialExecutionResult(
                status="failed",
                objective_value=None,
                experiment_id=experiment_id,
                manifest_path=None,
                checkpoint_path=(checkpoint_path if checkpoint_path.exists() else None),
                validation_policy_metrics=None,
                validation_value_metrics=None,
                message=(
                    "Neural trial resume/checkpoint recovery failed: "
                    f"{exc}"
                ),
            )

        if pruning_trial is not None:
            pruning_trial.report(metric, step=epoch)
            if epoch >= config.pruning.warmup_epochs and pruning_trial.should_prune():
                return TrialExecutionResult(
                    status="pruned",
                    objective_value=None,
                    experiment_id=experiment_id,
                    manifest_path=None,
                    checkpoint_path=(checkpoint_path if checkpoint_path.exists() else None),
                    validation_policy_metrics=None,
                    validation_value_metrics=None,
                    message=f"Pruned at epoch {epoch} with validation_policy_mrr={metric:.6f}",
                )

        resume_flag = True


def _resume_running_trials(
    *,
    study: optuna.study.Study,
    config: NeuralHpoConfig,
    mlflow_tracking_config: Path | None,
    mlflow_run_origin: Literal["historical", "current"],
    mlflow_include_model_weights: bool,
) -> TrialExecutionResult | None:
    running_trials = [trial for trial in study.trials if trial.state == TrialState.RUNNING]

    for frozen in running_trials:
        params = frozen.params
        try:
            learning_rate = float(params["learning_rate"])
            weight_decay = float(params["weight_decay"])
            value_weight = float(params["value_weight"])
        except (KeyError, TypeError, ValueError):
            _mark_trial_failed(
                study,
                frozen.number,
                "Running trial is missing required hyperparameter values",
            )
            continue

        try:
            live_trial = _live_trial_for_number(study, frozen.number)
        except Exception as exc:  # noqa: BLE001
            _mark_trial_failed(
                study,
                frozen.number,
                f"Unable to recover running trial handle for resume: {exc}",
            )
            continue

        try:
            outcome = _run_trial(
                config=config,
                trial_number=frozen.number,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                value_weight=value_weight,
                pruning_trial=live_trial,
                resume=True,
            )
        except Exception as exc:  # noqa: BLE001
            outcome = TrialExecutionResult(
                status="failed",
                objective_value=None,
                experiment_id=None,
                manifest_path=None,
                checkpoint_path=None,
                validation_policy_metrics=None,
                validation_value_metrics=None,
                message=f"Unhandled exception while resuming running trial: {exc}",
            )
        _attach_outcome_metadata(study, frozen.number, outcome)

        if outcome.status == "complete":
            if outcome.objective_value is None:
                _mark_trial_failed(
                    study,
                    frozen.number,
                    "Completed trial outcome did not provide objective value",
                )
            else:
                _set_trial_user_attr(study, frozen.number, "completed_at_utc", _utc_now_iso())
                _track_trial_in_mlflow(
                    study=study,
                    trial_number=frozen.number,
                    outcome=outcome,
                    tracking_config_path=mlflow_tracking_config,
                    mode=mlflow_run_origin,
                    include_model_weights=mlflow_include_model_weights,
                )
                study.tell(frozen.number, outcome.objective_value)
            continue

        if outcome.status == "pruned":
            if outcome.message is not None:
                _set_trial_user_attr(study, frozen.number, "pruned_reason", outcome.message)
            study.tell(frozen.number, state=TrialState.PRUNED)
            continue

        if outcome.status == "failed":
            _mark_trial_failed(
                study,
                frozen.number,
                outcome.message or "Trial failed during resume",
            )
            continue

        if outcome.status == "interrupted":
            if outcome.message is not None:
                _set_trial_user_attr(study, frozen.number, "interrupted_reason", outcome.message)
            return outcome

    return None


def _track_trial_in_mlflow(
    *,
    study: optuna.study.Study,
    trial_number: int,
    outcome: TrialExecutionResult,
    tracking_config_path: Path | None,
    mode: Literal["historical", "current"],
    include_model_weights: bool,
) -> None:
    if tracking_config_path is None or outcome.manifest_path is None:
        return

    try:
        result = import_manifest_to_mlflow(
            manifest_path=outcome.manifest_path,
            pipeline_kind="neural",
            tracking_config_path=tracking_config_path,
            mode=mode,
            include_model_weights=include_model_weights,
        )
    except Exception as exc:  # noqa: BLE001
        _set_trial_user_attr(study, trial_number, "mlflow_import_status", "failed")
        _set_trial_user_attr(study, trial_number, "mlflow_import_error", str(exc))
        return

    _set_trial_user_attr(study, trial_number, "mlflow_import_status", "complete")
    _set_trial_user_attr(study, trial_number, "mlflow_run_id", result.mlflow_run_id)
    _set_trial_user_attr(study, trial_number, "mlflow_experiment_id", result.mlflow_experiment_id)
    _set_trial_user_attr(study, trial_number, "mlflow_reused_existing", result.reused_existing)


def _write_trial_reports(study: optuna.study.Study, study_root: Path) -> None:
    reports_root = _reports_root(study_root)
    reports_root.mkdir(parents=True, exist_ok=True)

    json_rows: list[dict[str, Any]] = []
    for trial in study.trials:
        json_rows.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": dict(sorted(trial.params.items())),
                "user_attrs": dict(sorted(trial.user_attrs.items())),
                "datetime_start": (
                    None if trial.datetime_start is None else trial.datetime_start.isoformat()
                ),
                "datetime_complete": (
                    None
                    if trial.datetime_complete is None
                    else trial.datetime_complete.isoformat()
                ),
            }
        )

    atomic_write_json(reports_root / "trials.json", {"trials": json_rows})

    csv_path = reports_root / "trials.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "number",
                "state",
                "value",
                "learning_rate",
                "weight_decay",
                "value_weight",
                "neural_experiment_id",
                "neural_manifest_path",
                "neural_checkpoint_path",
                "validation_policy_mrr",
                "validation_value_accuracy",
                "validation_value_macro_f1",
                "validation_value_ece",
                "mlflow_import_status",
                "mlflow_run_id",
                "failure_reason",
                "datetime_start",
                "datetime_complete",
            ],
        )
        writer.writeheader()
        for row in json_rows:
            params = row["params"] if isinstance(row["params"], dict) else {}
            user_attrs = row["user_attrs"] if isinstance(row["user_attrs"], dict) else {}
            writer.writerow(
                {
                    "number": row["number"],
                    "state": row["state"],
                    "value": row["value"],
                    "learning_rate": params.get("learning_rate"),
                    "weight_decay": params.get("weight_decay"),
                    "value_weight": params.get("value_weight"),
                    "neural_experiment_id": user_attrs.get("neural_experiment_id"),
                    "neural_manifest_path": user_attrs.get("neural_manifest_path"),
                    "neural_checkpoint_path": user_attrs.get("neural_checkpoint_path"),
                    "validation_policy_mrr": user_attrs.get("validation_policy_mrr"),
                    "validation_value_accuracy": user_attrs.get("validation_value_accuracy"),
                    "validation_value_macro_f1": user_attrs.get("validation_value_macro_f1"),
                    "validation_value_ece": user_attrs.get("validation_value_ece"),
                    "mlflow_import_status": user_attrs.get("mlflow_import_status"),
                    "mlflow_run_id": user_attrs.get("mlflow_run_id"),
                    "failure_reason": user_attrs.get("failure_reason"),
                    "datetime_start": row["datetime_start"],
                    "datetime_complete": row["datetime_complete"],
                }
            )


def _status_payload(
    *,
    study: optuna.study.Study,
    config: NeuralHpoConfig,
    study_root: Path,
    identity_hash: str,
) -> dict[str, Any]:
    started = _started_trial_count(study)
    remaining = max(config.budget.max_total_trials - started, 0)
    counts = _count_by_state(study)

    best_payload: dict[str, Any] | None = None
    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None

    if best_trial is not None and best_trial.state == TrialState.COMPLETE:
        best_payload = {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": dict(sorted(best_trial.params.items())),
            "user_attrs": dict(sorted(best_trial.user_attrs.items())),
        }

    tracking_failures = sum(
        1
        for trial in study.trials
        if str(trial.user_attrs.get("mlflow_import_status", "")).strip().lower() == "failed"
    )

    return {
        "study_name": config.study.name,
        "study_root": study_root.as_posix(),
        "storage_path": config.study.storage_path.as_posix(),
        "identity_hash": identity_hash,
        "max_total_trials": config.budget.max_total_trials,
        "max_new_trials_per_invocation": config.budget.max_new_trials_per_invocation,
        "started_trials": started,
        "remaining_trial_budget": remaining,
        "trial_state_counts": counts,
        "best_trial": best_payload,
        "execution": {
            "trial_concurrency": config.execution.trial_concurrency,
            "runtime_threads": config.runtime.threads,
        },
        "mlflow_tracking_failures": tracking_failures,
    }


def _export_best_config(
    *,
    config: NeuralHpoConfig,
    study: optuna.study.Study,
    study_root: Path,
    output_path: Path,
    final_output_root: Path | None,
) -> Path:
    try:
        best = study.best_trial
    except ValueError as exc:
        raise RuntimeError(
            "Cannot export best config: no completed best trial is available"
        ) from exc

    if best.state != TrialState.COMPLETE:
        raise RuntimeError("Cannot export best config: no completed best trial is available")

    trial_config_path = _trial_root(study_root, best.number) / "trial_config.yaml"
    if not trial_config_path.exists():
        raise RuntimeError(
            "Cannot export best config: winning trial config not found at "
            + trial_config_path.as_posix()
        )

    payload = yaml.safe_load(trial_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Winning trial config must parse to a YAML object")

    output_raw = payload.setdefault("output", {})
    if not isinstance(output_raw, dict):
        raise RuntimeError("Winning trial config output block must be an object")

    target_output_root = (
        final_output_root
        if final_output_root is not None
        else (study_root / "final_runs" / f"best_trial_{best.number:04d}")
    )
    output_raw["output_root"] = target_output_root.as_posix()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    manifest_path_text = best.user_attrs.get("neural_manifest_path")
    if not isinstance(manifest_path_text, str) or not manifest_path_text.strip():
        raise RuntimeError("Cannot verify export: winning trial manifest path is unavailable")
    winning_manifest = Path(manifest_path_text)
    if not winning_manifest.exists():
        raise RuntimeError(
            "Cannot verify export: winning trial manifest path does not exist: "
            + winning_manifest.as_posix()
        )
    winning_manifest_payload = read_json_object(winning_manifest)
    winning_fingerprints = winning_manifest_payload.get("selection_fingerprints")
    winning_counts = winning_manifest_payload.get("counts")
    if not isinstance(winning_fingerprints, dict) or not isinstance(winning_counts, dict):
        raise RuntimeError(
            "Cannot verify export: winning manifest missing selection fingerprints/counts"
        )

    dry_result = run_neural(
        config_path=output_path,
        dry_run=True,
        evaluation_scope="validation_only",
    )
    dry_summary = dry_result.summary or {}
    dry_fingerprints = dry_summary.get("selection_fingerprints")
    dry_counts = dry_summary.get("counts")
    if not isinstance(dry_fingerprints, dict) or not isinstance(dry_counts, dict):
        raise RuntimeError("Export verification dry-run did not return selection details")

    if dry_fingerprints != winning_fingerprints:
        raise RuntimeError(
            "Export verification failed: dry-run selection fingerprints differ from winning trial"
        )

    winning_training_population = winning_counts.get("training_population")
    dry_training_population = dry_counts.get("training_population")
    if dry_training_population != winning_training_population:
        raise RuntimeError(
            "Export verification failed: dry-run training population differs from winning trial"
        )

    return output_path


def _run_hpo(
    *,
    config: NeuralHpoConfig,
    resume_running_trials: bool,
    max_new_trials_override: int | None,
    dry_run: bool,
    status_only: bool,
    export_best_config: Path | None,
    export_final_output_root: Path | None,
    mlflow_tracking_config: Path | None,
    mlflow_run_origin: Literal["historical", "current"],
    mlflow_include_model_weights: bool,
) -> dict[str, Any]:
    study_root = _study_root(config)
    study_root.mkdir(parents=True, exist_ok=True)

    with _StudyLock(study_root / ".study.lock"):
        reference_preflight = _reference_preflight_identity_payload(config, study_root)
        identity_hash, _ = _enforce_study_identity(
            config,
            study_root,
            reference_preflight,
        )

        study, sampler = _create_or_load_study(config=config, study_root=study_root)
        status = _status_payload(
            study=study,
            config=config,
            study_root=study_root,
            identity_hash=identity_hash,
        )
        status["reference_preflight"] = reference_preflight

        if dry_run:
            new_cap = (
                config.budget.max_new_trials_per_invocation
                if max_new_trials_override is None
                else max_new_trials_override
            )
            if new_cap <= 0:
                raise RuntimeError("--max-new-trials must be positive when provided")
            status["dry_run"] = True
            status["requested_new_trial_cap"] = int(new_cap)
            status["would_start_new_trials"] = int(
                min(new_cap, max(config.budget.max_total_trials - status["started_trials"], 0))
            )
            _write_trial_reports(study, study_root)
            _save_sampler_state(sampler, _sampler_state_path(study_root))
            return status

        if status_only:
            if export_best_config is not None:
                exported = _export_best_config(
                    config=config,
                    study=study,
                    study_root=study_root,
                    output_path=export_best_config,
                    final_output_root=export_final_output_root,
                )
                status["exported_best_config_path"] = exported.as_posix()
            _write_trial_reports(study, study_root)
            _save_sampler_state(sampler, _sampler_state_path(study_root))
            return status

        running_before = [trial for trial in study.trials if trial.state == TrialState.RUNNING]
        if running_before and not resume_running_trials:
            raise RuntimeError(
                "Study has running trials. Re-run with --resume to continue these trials "
                "or use --status to inspect state."
            )

        if resume_running_trials:
            interrupted = _resume_running_trials(
                study=study,
                config=config,
                mlflow_tracking_config=mlflow_tracking_config,
                mlflow_run_origin=mlflow_run_origin,
                mlflow_include_model_weights=mlflow_include_model_weights,
            )
            if interrupted is not None:
                status_after_resume = _status_payload(
                    study=study,
                    config=config,
                    study_root=study_root,
                    identity_hash=identity_hash,
                )
                status_after_resume["reference_preflight"] = reference_preflight
                status_after_resume["status"] = "interrupted_during_resume"
                status_after_resume["message"] = interrupted.message
                _write_trial_reports(study, study_root)
                _save_sampler_state(sampler, _sampler_state_path(study_root))
                return status_after_resume

        _maybe_enqueue_reference_trial(study, config)

        started = _started_trial_count(study)
        remaining_total = max(config.budget.max_total_trials - started, 0)

        new_cap = (
            config.budget.max_new_trials_per_invocation
            if max_new_trials_override is None
            else max_new_trials_override
        )
        if new_cap <= 0:
            raise RuntimeError("--max-new-trials must be positive when provided")

        to_start = min(new_cap, remaining_total)

        started_this_invocation = 0
        interrupted_trial: TrialExecutionResult | None = None

        for _ in range(to_start):
            trial = study.ask()
            started_this_invocation += 1

            learning_rate = trial.suggest_float(
                "learning_rate",
                config.search_space.learning_rate.low,
                config.search_space.learning_rate.high,
                log=config.search_space.learning_rate.log,
            )
            weight_decay = trial.suggest_float(
                "weight_decay",
                config.search_space.weight_decay.low,
                config.search_space.weight_decay.high,
                log=config.search_space.weight_decay.log,
            )
            value_weight = trial.suggest_categorical(
                "value_weight",
                list(config.search_space.value_weight_choices),
            )

            trial_output = _trial_output_root(study_root, trial.number)
            trial.set_user_attr("trial_output_root", trial_output.as_posix())
            trial.set_user_attr("started_at_utc", _utc_now_iso())
            _save_sampler_state(sampler, _sampler_state_path(study_root))

            try:
                outcome = _run_trial(
                    config=config,
                    trial_number=trial.number,
                    learning_rate=float(learning_rate),
                    weight_decay=float(weight_decay),
                    value_weight=float(value_weight),
                    pruning_trial=trial,
                    resume=False,
                )
            except Exception as exc:  # noqa: BLE001
                outcome = TrialExecutionResult(
                    status="failed",
                    objective_value=None,
                    experiment_id=None,
                    manifest_path=None,
                    checkpoint_path=None,
                    validation_policy_metrics=None,
                    validation_value_metrics=None,
                    message=f"Unhandled exception while executing trial: {exc}",
                )
            _attach_outcome_metadata(study, trial.number, outcome)

            if outcome.status == "complete":
                if outcome.objective_value is None:
                    _mark_trial_failed(
                        study,
                        trial.number,
                        "Completed trial outcome did not provide objective value",
                    )
                else:
                    trial.set_user_attr("completed_at_utc", _utc_now_iso())
                    _track_trial_in_mlflow(
                        study=study,
                        trial_number=trial.number,
                        outcome=outcome,
                        tracking_config_path=mlflow_tracking_config,
                        mode=mlflow_run_origin,
                        include_model_weights=mlflow_include_model_weights,
                    )
                    study.tell(trial, outcome.objective_value)
            elif outcome.status == "pruned":
                if outcome.message is not None:
                    trial.set_user_attr("pruned_reason", outcome.message)
                study.tell(trial, state=TrialState.PRUNED)
            elif outcome.status == "failed":
                _mark_trial_failed(study, trial.number, outcome.message or "Trial failed")
            else:
                interrupted_trial = outcome
                if outcome.message is not None:
                    trial.set_user_attr("interrupted_reason", outcome.message)
                break

            _save_sampler_state(sampler, _sampler_state_path(study_root))

        _write_trial_reports(study, study_root)
        _save_sampler_state(sampler, _sampler_state_path(study_root))

        final_payload = _status_payload(
            study=study,
            config=config,
            study_root=study_root,
            identity_hash=identity_hash,
        )
        final_payload["reference_preflight"] = reference_preflight
        final_payload["started_new_trials_this_invocation"] = started_this_invocation

        if interrupted_trial is not None:
            final_payload["status"] = "interrupted"
            final_payload["message"] = interrupted_trial.message
        else:
            final_payload["status"] = "ok"

        if export_best_config is not None:
            exported = _export_best_config(
                config=config,
                study=study,
                study_root=study_root,
                output_path=export_best_config,
                final_output_root=export_final_output_root,
            )
            final_payload["exported_best_config_path"] = exported.as_posix()

        return final_payload


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_neural_hpo_config(args.config)
        payload = _run_hpo(
            config=config,
            resume_running_trials=bool(args.resume),
            max_new_trials_override=args.max_new_trials,
            dry_run=bool(args.dry_run),
            status_only=bool(args.status),
            export_best_config=args.export_best_config,
            export_final_output_root=args.export_final_output_root,
            mlflow_tracking_config=args.mlflow_tracking_config,
            mlflow_run_origin=args.mlflow_run_origin,
            mlflow_include_model_weights=bool(args.mlflow_include_model_weights),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"run_neural_hpo_error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"study_name={payload['study_name']}")
        print(f"status={payload.get('status', 'ok')}")
        print(f"started_trials={payload['started_trials']}")
        print(f"remaining_trial_budget={payload['remaining_trial_budget']}")
        if payload.get("best_trial") is not None:
            best = payload["best_trial"]
            if isinstance(best, dict):
                print(f"best_trial_number={best.get('number')}")
                print(f"best_trial_value={best.get('value')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
