"""MLflow import and tracking helpers for ChessLens experiments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import mlflow

from chesslens.modeling.mlflow_tracking_config import (
    MlflowTrackingConfig,
    load_mlflow_tracking_config,
)
from chesslens.modeling.validation import read_json_object

PipelineKind = Literal["baseline", "neural"]
ImportMode = Literal["historical", "current"]

_BASELINE_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "effective_config.json",
    "aggregate_metrics.json",
    "subgroup_metrics.json",
    "bootstrap_confidence_intervals.json",
    "lightgbm_model.txt",
    "logistic_pipeline.pkl",
    "frequency_model.json",
    "leakage_audit.json",
    "leakage_audit.md",
    "baseline_report.md",
)

_NEURAL_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "effective_config.json",
    "experiment_identity_payload.json",
    "feature_preprocessing.json",
    "aggregate_metrics.json",
    "subgroup_metrics.json",
    "leakage_audit.json",
    "leakage_audit.md",
    "training_history.json",
    "checkpoint_last.pt",
    "checkpoint_best.pt",
    "model_state_best.pt",
    "model_state_last.pt",
    "environment.json",
)

_JSON_ARTIFACT_EXTENSIONS: tuple[str, ...] = (".json", ".md", ".txt")
_IMPORT_STATUS_TAG = "chesslens_import_status"
_IMPORT_STATUS_COMPLETE = "complete"
_IMPORT_STATUS_IN_PROGRESS = "in_progress"
_IMPORT_STATUS_FAILED = "failed"


@dataclass(frozen=True)
class MlflowImportResult:
    pipeline_kind: PipelineKind
    experiment_id: str
    manifest_path: Path
    mlflow_experiment_id: str
    mlflow_run_id: str
    reused_existing: bool
    mode: ImportMode


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_artifacts_for_kind(kind: PipelineKind) -> tuple[str, ...]:
    if kind == "baseline":
        return _BASELINE_REQUIRED_ARTIFACTS
    return _NEURAL_REQUIRED_ARTIFACTS


def _validate_manifest_and_artifacts(
    *,
    manifest_path: Path,
    pipeline_kind: PipelineKind,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest = read_json_object(manifest_path)
    if str(manifest.get("status", "")).strip() != "complete":
        raise RuntimeError(
            "Only complete experiments are importable to MLflow; "
            f"manifest status={manifest.get('status')!r}"
        )

    experiment_root = manifest_path.parent
    success_path = experiment_root / "_SUCCESS"
    if not success_path.exists():
        raise RuntimeError("Experiment import rejected: _SUCCESS marker not found")

    artifact_integrity = manifest.get("artifact_integrity")
    if not isinstance(artifact_integrity, dict):
        raise RuntimeError("Experiment import rejected: manifest missing artifact_integrity")

    required = _required_artifacts_for_kind(pipeline_kind)
    missing_required = [name for name in required if name not in artifact_integrity]
    if missing_required:
        raise RuntimeError(
            "Experiment import rejected: missing artifact_integrity entries: "
            + ", ".join(sorted(missing_required))
        )

    for file_name, record in sorted(artifact_integrity.items()):
        if not isinstance(record, dict):
            raise RuntimeError(f"Invalid artifact_integrity record for {file_name!r}")

        expected_size = record.get("size_bytes")
        expected_sha = str(record.get("sha256", "")).strip().lower()
        if not isinstance(expected_size, int) or expected_size < 0:
            raise RuntimeError(
                "Invalid artifact_integrity.size_bytes for "
                f"{file_name!r}: {expected_size!r}"
            )
        if len(expected_sha) != 64:
            raise RuntimeError(
                "Invalid artifact_integrity.sha256 for "
                f"{file_name!r}: {record.get('sha256')!r}"
            )

        artifact_path = experiment_root / file_name
        if not artifact_path.exists():
            raise RuntimeError(
                "Experiment import rejected: missing artifact file "
                f"{file_name!r}"
            )

        actual_size = artifact_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                "Experiment import rejected: artifact size mismatch for "
                f"{file_name!r} expected={expected_size} actual={actual_size}"
            )

        actual_sha = _file_sha256(artifact_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                "Experiment import rejected: artifact sha256 mismatch for "
                f"{file_name!r} expected={expected_sha} actual={actual_sha}"
            )

    return manifest, experiment_root, artifact_integrity


def _ensure_experiment(config: MlflowTrackingConfig) -> str:
    config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    config.artifact_root.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(config.tracking_uri)

    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is not None:
        return str(experiment.experiment_id)

    experiment_id = mlflow.create_experiment(
        name=config.experiment_name,
        artifact_location=config.artifact_root.as_posix(),
    )
    return str(experiment_id)


def _flatten_params(payload: Any, *, prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}

    if isinstance(payload, dict):
        for key, value in sorted(payload.items()):
            key_text = str(key)
            path = key_text if not prefix else f"{prefix}.{key_text}"
            result.update(_flatten_params(value, prefix=path))
        return result

    if isinstance(payload, list):
        result[prefix] = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return result

    if payload is None:
        result[prefix] = "null"
        return result

    result[prefix] = str(payload)
    return result


def _flatten_numeric_metrics(payload: Any, *, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}

    if isinstance(payload, dict):
        for key, value in sorted(payload.items()):
            key_text = str(key)
            path = key_text if not prefix else f"{prefix}.{key_text}"
            result.update(_flatten_numeric_metrics(value, prefix=path))
        return result

    if isinstance(payload, list):
        return result

    if isinstance(payload, bool):
        return result

    if isinstance(payload, (int, float)):
        value = float(payload)
        if math.isfinite(value):
            result[prefix] = value
    return result


def _history_epoch_metrics(history_row: dict[str, Any]) -> dict[str, float]:
    mapping: dict[str, str] = {
        "train_total_loss": "train.total_loss",
        "train_policy_loss": "train.policy_loss",
        "train_value_loss": "train.value_loss",
        "validation_total_loss": "validation.total_loss",
        "validation_policy_loss": "validation.policy_loss",
        "validation_value_loss": "validation.value_loss",
        "validation_policy_mrr": "epoch.validation.policy.mrr",
        "duration_seconds": "epoch.duration_seconds",
    }

    metrics: dict[str, float] = {}
    for source_key, metric_key in mapping.items():
        value = history_row.get(source_key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metric_value = float(value)
            if math.isfinite(metric_value):
                metrics[metric_key] = metric_value
    return metrics


def _log_epoch_history_if_present(experiment_root: Path) -> None:
    history_path = experiment_root / "training_history.json"
    if not history_path.exists():
        return

    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return

    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        epoch_value = row.get("epoch", index)
        try:
            step = int(epoch_value)
        except (TypeError, ValueError):
            step = index

        for metric_key, metric_value in _history_epoch_metrics(row).items():
            mlflow.log_metric(metric_key, metric_value, step=step)


def _namespaced_aggregate_metrics(
    *,
    aggregate_payload: dict[str, Any],
    pipeline_kind: PipelineKind,
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    if pipeline_kind == "neural":
        policy_payload = aggregate_payload.get("policy")
        value_payload = aggregate_payload.get("value")
        losses_payload = aggregate_payload.get("losses")

        if isinstance(policy_payload, dict):
            best = policy_payload.get("best_validation_policy_mrr")
            if isinstance(best, (int, float)) and math.isfinite(float(best)):
                metrics["validation.policy.best_mrr"] = float(best)
            for split in ("validation", "test"):
                split_payload = policy_payload.get(split)
                if isinstance(split_payload, dict):
                    metrics.update(
                        _flatten_numeric_metrics(
                            split_payload,
                            prefix=f"{split}.policy",
                        )
                    )

        if isinstance(value_payload, dict):
            for split in ("validation", "test"):
                split_payload = value_payload.get(split)
                if isinstance(split_payload, dict):
                    metrics.update(
                        _flatten_numeric_metrics(
                            split_payload,
                            prefix=f"{split}.value",
                        )
                    )

        if isinstance(losses_payload, dict):
            for split in ("validation", "test"):
                for suffix in ("total_loss", "policy_loss", "value_loss"):
                    key = f"{split}_{suffix}"
                    value = losses_payload.get(key)
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        metrics[f"{split}.loss.{suffix}"] = float(value)

        return metrics

    lightgbm_payload = aggregate_payload.get("lightgbm_policy")
    if isinstance(lightgbm_payload, dict):
        for split in ("validation", "test"):
            split_payload = lightgbm_payload.get(split)
            if isinstance(split_payload, dict):
                metrics.update(
                    _flatten_numeric_metrics(split_payload, prefix=f"{split}.policy")
                )

    logistic_payload = aggregate_payload.get("logistic_wdl")
    if isinstance(logistic_payload, dict):
        for split in ("validation", "test"):
            split_payload = logistic_payload.get(split)
            if isinstance(split_payload, dict):
                metrics.update(
                    _flatten_numeric_metrics(split_payload, prefix=f"{split}.value")
                )

    return metrics


def _log_default_artifacts(
    *,
    experiment_root: Path,
    artifact_integrity: dict[str, Any],
    pipeline_kind: PipelineKind,
    include_model_weights: bool,
) -> None:
    for file_name in sorted(artifact_integrity):
        suffix = Path(file_name).suffix.lower()
        if suffix not in _JSON_ARTIFACT_EXTENSIONS:
            continue
        mlflow.log_artifact((experiment_root / file_name).as_posix(), artifact_path="reports")

    if include_model_weights and pipeline_kind == "neural":
        model_state_path = experiment_root / "model_state_best.pt"
        if model_state_path.exists():
            mlflow.log_artifact(model_state_path.as_posix(), artifact_path="weights")


def _existing_run_for_experiment(
    *,
    mlflow_experiment_id: str,
    pipeline_kind: PipelineKind,
    experiment_id: str,
) -> Any | None:
    filter_string = (
        f"tags.chesslens_pipeline_kind = '{pipeline_kind}' "
        f"and tags.chesslens_experiment_id = '{experiment_id}'"
    )
    runs = mlflow.search_runs(
        experiment_ids=[mlflow_experiment_id],
        filter_string=filter_string,
        output_format="list",
    )

    if not runs:
        return None
    if len(runs) > 1:
        raise RuntimeError(
            "MLflow conflict: multiple runs already exist for ChessLens experiment identity "
            f"{experiment_id!r} in MLflow experiment {mlflow_experiment_id!r}"
        )
    return runs[0]


def import_manifest_to_mlflow(
    *,
    manifest_path: Path,
    pipeline_kind: PipelineKind,
    tracking_config_path: Path,
    mode: ImportMode,
    include_model_weights: bool | None = None,
) -> MlflowImportResult:
    config = load_mlflow_tracking_config(tracking_config_path)

    manifest, experiment_root, artifact_integrity = _validate_manifest_and_artifacts(
        manifest_path=manifest_path,
        pipeline_kind=pipeline_kind,
    )

    experiment_id = str(manifest.get("experiment_id", "")).strip()
    if not experiment_id:
        raise RuntimeError("Experiment import rejected: manifest missing experiment_id")

    identity_hash = str(manifest.get("identity_payload_sha256", "")).strip()
    if len(identity_hash) != 64:
        raise RuntimeError(
            "Experiment import rejected: invalid identity_payload_sha256 in manifest"
        )

    mlflow_experiment_id = _ensure_experiment(config)
    mode_tag = "historical_import" if mode == "historical" else "current_code"
    manifest_sha256 = _file_sha256(manifest_path)

    existing = _existing_run_for_experiment(
        mlflow_experiment_id=mlflow_experiment_id,
        pipeline_kind=pipeline_kind,
        experiment_id=experiment_id,
    )
    if existing is not None:
        tags = dict(existing.data.tags)

        existing_identity = str(tags.get("chesslens_identity_payload_sha256", "")).strip()
        if existing_identity and existing_identity != identity_hash:
            raise RuntimeError(
                "MLflow conflict: existing run has mismatched identity hash for "
                f"ChessLens experiment {experiment_id!r}"
            )

        existing_mode = str(tags.get("chesslens_run_origin", "")).strip()
        if existing_mode and existing_mode != mode_tag:
            raise RuntimeError(
                "MLflow conflict: existing run has different run origin tag; "
                f"existing={existing_mode!r}, requested={mode_tag!r}"
            )

        existing_import_status = str(tags.get(_IMPORT_STATUS_TAG, "")).strip().lower()
        existing_manifest_sha = str(tags.get("chesslens_source_manifest_sha256", "")).strip()
        if existing_import_status != _IMPORT_STATUS_COMPLETE:
            raise RuntimeError(
                "MLflow run exists but import is incomplete/failed. "
                "Resolve the partial run explicitly before retrying import. "
                f"run_id={existing.info.run_id} status={existing_import_status!r}"
            )
        if str(existing.info.status).strip().upper() != "FINISHED":
            raise RuntimeError(
                "MLflow run exists but is not FINISHED. Resolve run state before reuse: "
                f"run_id={existing.info.run_id} status={existing.info.status!r}"
            )
        if existing_manifest_sha and existing_manifest_sha != manifest_sha256:
            raise RuntimeError(
                "MLflow conflict: existing complete import run references a different "
                "source manifest checksum"
            )

        return MlflowImportResult(
            pipeline_kind=pipeline_kind,
            experiment_id=experiment_id,
            manifest_path=manifest_path,
            mlflow_experiment_id=mlflow_experiment_id,
            mlflow_run_id=str(existing.info.run_id),
            reused_existing=True,
            mode=mode,
        )

    include_weights_flag = (
        config.default_include_model_weights
        if include_model_weights is None
        else include_model_weights
    )

    with mlflow.start_run(
        experiment_id=mlflow_experiment_id,
        run_name=f"{pipeline_kind}-{experiment_id[:12]}",
    ) as run:
        mlflow.set_tags(
            {
                "chesslens_pipeline_kind": pipeline_kind,
                "chesslens_experiment_id": experiment_id,
                "chesslens_identity_payload_sha256": identity_hash,
                "chesslens_run_origin": mode_tag,
                "chesslens_imported_historical": str(mode == "historical").lower(),
                "chesslens_imported_at_utc": _utc_now_iso(),
                "chesslens_manifest_path": manifest_path.as_posix(),
                "chesslens_git_commit": str(manifest.get("git_commit") or ""),
                "chesslens_source_manifest_sha256": manifest_sha256,
                _IMPORT_STATUS_TAG: _IMPORT_STATUS_IN_PROGRESS,
            }
        )

        try:
            identity_payload = manifest.get("identity_payload")
            if isinstance(identity_payload, dict):
                identity_params = _flatten_params(identity_payload, prefix="identity")
                for key, value in sorted(identity_params.items()):
                    if key:
                        mlflow.log_param(key, value)

            modeling_input = manifest.get("modeling_input")
            if isinstance(modeling_input, dict):
                modeling_params = _flatten_params(modeling_input, prefix="modeling")
                for key, value in sorted(modeling_params.items()):
                    if key:
                        mlflow.log_param(key, value)

            aggregate_metrics_path = experiment_root / "aggregate_metrics.json"
            if aggregate_metrics_path.exists():
                aggregate_payload = json.loads(aggregate_metrics_path.read_text(encoding="utf-8"))
                if isinstance(aggregate_payload, dict):
                    aggregate_metrics = _namespaced_aggregate_metrics(
                        aggregate_payload=aggregate_payload,
                        pipeline_kind=pipeline_kind,
                    )
                    for metric_key, metric_value in sorted(aggregate_metrics.items()):
                        if metric_key:
                            mlflow.log_metric(metric_key, metric_value)

            _log_epoch_history_if_present(experiment_root)

            _log_default_artifacts(
                experiment_root=experiment_root,
                artifact_integrity=artifact_integrity,
                pipeline_kind=pipeline_kind,
                include_model_weights=include_weights_flag,
            )
        except Exception as exc:  # noqa: BLE001
            mlflow.set_tags(
                {
                    _IMPORT_STATUS_TAG: _IMPORT_STATUS_FAILED,
                    "chesslens_import_failed_at_utc": _utc_now_iso(),
                    "chesslens_import_error": str(exc),
                }
            )
            raise

        mlflow.set_tags(
            {
                _IMPORT_STATUS_TAG: _IMPORT_STATUS_COMPLETE,
                "chesslens_import_completed_at_utc": _utc_now_iso(),
                "chesslens_import_source_identity_sha256": identity_hash,
            }
        )

        return MlflowImportResult(
            pipeline_kind=pipeline_kind,
            experiment_id=experiment_id,
            manifest_path=manifest_path,
            mlflow_experiment_id=mlflow_experiment_id,
            mlflow_run_id=str(run.info.run_id),
            reused_existing=False,
            mode=mode,
        )


def mlflow_import_result_to_json(result: MlflowImportResult) -> dict[str, Any]:
    return {
        "pipeline_kind": result.pipeline_kind,
        "experiment_id": result.experiment_id,
        "manifest_path": result.manifest_path.as_posix(),
        "mlflow_experiment_id": result.mlflow_experiment_id,
        "mlflow_run_id": result.mlflow_run_id,
        "reused_existing": result.reused_existing,
        "mode": result.mode,
    }
