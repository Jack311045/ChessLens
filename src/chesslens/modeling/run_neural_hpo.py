"""Bounded Optuna HPO runner for Phase 3.2 validation-only neural tuning."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import optuna
import torch
import yaml
from optuna.trial import TrialState

from chesslens.ingestion.shard_manifest import atomic_write_json
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
    message: str | None


class _StudyLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._acquired = False

    def __enter__(self) -> _StudyLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._path.exists():
            payload = self._read_payload()
            pid = payload.get("pid") if isinstance(payload, dict) else None
            if isinstance(pid, int) and not _pid_exists(pid):
                self._path.unlink(missing_ok=True)

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(str(self._path), flags)
        except FileExistsError as exc:
            raise RuntimeError(
                "Another HPO writer is active for this study: " + self._path.as_posix()
            ) from exc

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "created_at_utc": _utc_now_iso()}, handle, indent=2)
            handle.write("\n")

        self._acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._acquired:
            self._path.unlink(missing_ok=True)
            self._acquired = False

    def _read_payload(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


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


def _load_reference_identity_payload(config: NeuralHpoConfig) -> dict[str, Any]:
    reference_neural = load_neural_config(config.reference.neural_config_path)
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


def _enforce_study_identity(
    config: NeuralHpoConfig,
    study_root: Path,
) -> tuple[str, dict[str, Any]]:
    reference_identity = _load_reference_identity_payload(config)
    payload = hpo_config_identity_payload(
        config=config,
        reference_neural_identity_payload=reference_identity,
    )
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
        payload = pickle.loads(sampler_state_path.read_bytes())
        if isinstance(payload, optuna.samplers.BaseSampler):
            return payload

    return optuna.samplers.TPESampler(seed=config.runtime.seed)


def _save_sampler_state(sampler: optuna.samplers.BaseSampler, sampler_state_path: Path) -> None:
    sampler_state_path.parent.mkdir(parents=True, exist_ok=True)
    sampler_state_path.write_bytes(pickle.dumps(sampler))


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
    reference = load_neural_config(config.reference.neural_config_path)

    params = {
        "learning_rate": float(reference.optimization.learning_rate),
        "weight_decay": float(reference.optimization.weight_decay),
        "value_weight": float(reference.loss.value_weight),
    }

    if not (
        config.search_space.learning_rate.low
        <= params["learning_rate"]
        <= config.search_space.learning_rate.high
    ):
        return

    if not (
        config.search_space.weight_decay.low
        <= params["weight_decay"]
        <= config.search_space.weight_decay.high
    ):
        return

    if params["value_weight"] not in set(config.search_space.value_weight_choices):
        return

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


def _objective_from_completed_manifest(manifest_path: Path) -> float:
    aggregate = read_json_object(manifest_path.parent / "aggregate_metrics.json")

    policy_payload = aggregate.get("policy")
    if not isinstance(policy_payload, dict):
        raise RuntimeError(
            "Completed trial is missing policy metrics in aggregate_metrics.json"
        )

    best_metric = policy_payload.get("best_validation_policy_mrr")
    if isinstance(best_metric, (int, float)):
        return float(best_metric)

    validation_payload = policy_payload.get("validation")
    if isinstance(validation_payload, dict):
        mrr = validation_payload.get("mrr")
        if isinstance(mrr, (int, float)):
            return float(mrr)

    raise RuntimeError("Unable to resolve objective metric from completed trial artifacts")


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

    while True:
        result = run_neural(
            config_path=trial_config_path,
            resume=resume_flag,
            max_epochs_this_session=(1 if pruning_trial is not None else None),
            evaluation_scope="validation_only",
        )

        experiment_id = result.experiment_id
        if result.completed:
            objective = _objective_from_completed_manifest(result.manifest_path)
            return TrialExecutionResult(
                status="complete",
                objective_value=objective,
                experiment_id=experiment_id,
                message=None,
            )

        summary = result.summary or {}
        status = str(summary.get("status", "")).strip()
        if status == "interrupted":
            return TrialExecutionResult(
                status="interrupted",
                objective_value=None,
                experiment_id=experiment_id,
                message="Neural trial run was interrupted; checkpoint preserved for resume",
            )

        if status != "session_limit_reached":
            return TrialExecutionResult(
                status="failed",
                objective_value=None,
                experiment_id=experiment_id,
                message=f"Neural trial did not complete: status={status!r}",
            )

        epoch, metric = _latest_validation_metric(
            trial_output_root=trial_output_root,
            experiment_id=experiment_id,
        )

        if pruning_trial is not None:
            pruning_trial.report(metric, step=epoch)
            if epoch >= config.pruning.warmup_epochs and pruning_trial.should_prune():
                return TrialExecutionResult(
                    status="pruned",
                    objective_value=None,
                    experiment_id=experiment_id,
                    message=f"Pruned at epoch {epoch} with validation_policy_mrr={metric:.6f}",
                )

        resume_flag = True


def _resume_running_trials(
    *,
    study: optuna.study.Study,
    config: NeuralHpoConfig,
) -> TrialExecutionResult | None:
    running_trials = [trial for trial in study.trials if trial.state == TrialState.RUNNING]

    for frozen in running_trials:
        params = frozen.params
        try:
            learning_rate = float(params["learning_rate"])
            weight_decay = float(params["weight_decay"])
            value_weight = float(params["value_weight"])
        except (KeyError, TypeError, ValueError):
            study.tell(frozen.number, state=TrialState.FAIL)
            continue

        outcome = _run_trial(
            config=config,
            trial_number=frozen.number,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            value_weight=value_weight,
            pruning_trial=None,
            resume=True,
        )

        if outcome.status == "complete":
            if outcome.objective_value is None:
                study.tell(frozen.number, state=TrialState.FAIL)
            else:
                study.tell(frozen.number, outcome.objective_value)
            continue

        if outcome.status == "pruned":
            study.tell(frozen.number, state=TrialState.PRUNED)
            continue

        if outcome.status == "failed":
            study.tell(frozen.number, state=TrialState.FAIL)
            continue

        if outcome.status == "interrupted":
            return outcome

    return None


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
                "datetime_start",
                "datetime_complete",
            ],
        )
        writer.writeheader()
        for row in json_rows:
            params = row["params"] if isinstance(row["params"], dict) else {}
            writer.writerow(
                {
                    "number": row["number"],
                    "state": row["state"],
                    "value": row["value"],
                    "learning_rate": params.get("learning_rate"),
                    "weight_decay": params.get("weight_decay"),
                    "value_weight": params.get("value_weight"),
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
        }

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
    }


def _export_best_config(
    *,
    config: NeuralHpoConfig,
    study: optuna.study.Study,
    output_path: Path,
) -> Path:
    try:
        best = study.best_trial
    except ValueError as exc:
        raise RuntimeError(
            "Cannot export best config: no completed best trial is available"
        ) from exc

    if best.state != TrialState.COMPLETE:
        raise RuntimeError("Cannot export best config: no completed best trial is available")

    learning_rate = float(best.params["learning_rate"])
    weight_decay = float(best.params["weight_decay"])
    value_weight = float(best.params["value_weight"])

    payload = yaml.safe_load(config.reference.neural_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Reference neural config must parse to a YAML object")

    optimization_raw = payload.setdefault("optimization", {})
    if not isinstance(optimization_raw, dict):
        raise RuntimeError("reference.optimization must be an object")
    optimization_raw["learning_rate"] = learning_rate
    optimization_raw["weight_decay"] = weight_decay

    loss_raw = payload.setdefault("loss", {})
    if not isinstance(loss_raw, dict):
        raise RuntimeError("reference.loss must be an object")
    loss_raw["value_weight"] = value_weight

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def _run_hpo(
    *,
    config: NeuralHpoConfig,
    resume_running_trials: bool,
    max_new_trials_override: int | None,
    dry_run: bool,
    status_only: bool,
    export_best_config: Path | None,
) -> dict[str, Any]:
    study_root = _study_root(config)
    study_root.mkdir(parents=True, exist_ok=True)

    identity_hash, _ = _enforce_study_identity(config, study_root)

    study, sampler = _create_or_load_study(config=config, study_root=study_root)

    status = _status_payload(
        study=study,
        config=config,
        study_root=study_root,
        identity_hash=identity_hash,
    )

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
        return status

    if status_only:
        if export_best_config is not None:
            exported = _export_best_config(
                config=config,
                study=study,
                output_path=export_best_config,
            )
            status["exported_best_config_path"] = exported.as_posix()
        return status

    with _StudyLock(study_root / ".study.lock"):
        running_before = [trial for trial in study.trials if trial.state == TrialState.RUNNING]
        if running_before and not resume_running_trials:
            raise RuntimeError(
                "Study has running trials. Re-run with --resume to continue these trials "
                "or use --status to inspect state."
            )

        if resume_running_trials:
            interrupted = _resume_running_trials(study=study, config=config)
            if interrupted is not None:
                status_after_resume = _status_payload(
                    study=study,
                    config=config,
                    study_root=study_root,
                    identity_hash=identity_hash,
                )
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

            outcome = _run_trial(
                config=config,
                trial_number=trial.number,
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                value_weight=float(value_weight),
                pruning_trial=trial,
                resume=False,
            )

            started_this_invocation += 1

            if outcome.status == "complete":
                if outcome.objective_value is None:
                    study.tell(trial, state=TrialState.FAIL)
                else:
                    study.tell(trial, outcome.objective_value)
            elif outcome.status == "pruned":
                study.tell(trial, state=TrialState.PRUNED)
            elif outcome.status == "failed":
                study.tell(trial, state=TrialState.FAIL)
            else:
                interrupted_trial = outcome
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
                output_path=export_best_config,
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
