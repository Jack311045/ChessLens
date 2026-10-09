"""Phase 3.1 supervised neural training orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import psutil
import torch

from chesslens.ingestion.rss import PeakRssSampler
from chesslens.ingestion.shard_manifest import atomic_write_json
from chesslens.modeling.baselines_metrics import RankingGroup, compute_ranking_metrics
from chesslens.modeling.neural_config import (
    NeuralConfig,
    apply_neural_cli_overrides,
    load_neural_config,
    neural_config_identity_payload,
)
from chesslens.modeling.neural_data import (
    WDL_CLASS_ORDER,
    EncodedExample,
    NeuralPreprocessor,
    SelectionFingerprints,
    compare_selection_fingerprints,
    encode_examples,
    fit_preprocessor,
    selected_key_hash,
    selection_fingerprints,
)
from chesslens.modeling.neural_losses import LossWeights, compute_supervised_losses
from chesslens.modeling.neural_network import (
    NeuralModelSizes,
    PolicyValueResidualCNN,
    parameter_count,
)
from chesslens.modeling.run_baselines import (
    PositionExample,
    PreflightSummary,
    TrainingPopulationSummary,
    _annotate_player_holdout_subgroup,
    _load_position_examples,
    _logistic_metrics,
    _logistic_subgroups,
    _novel_position_ids,
    _player_holdout_interpretation,
    _preflight_modeling_dataset,
    _ranking_subgroups,
    _select_training_population,
)
from chesslens.modeling.validation import (
    canonical_json,
    read_json_object,
    sha256_text,
    validate_manifest_identity,
)
from chesslens.runtime_paths import resolve_modeling_manifest_override_or_env

EXCLUDED_TRAINING_FEATURES: tuple[str, ...] = ("eco", "opening_family", "opening")
ALLOWED_CONTEXT_FEATURES: tuple[str, ...] = (
    "mover_rating",
    "opponent_rating",
    "rating_difference",
    "ply",
    "mover_rating_band",
    "time_control_category",
)
NEURAL_REQUIRED_ARTIFACTS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class NeuralRunResult:
    run_id: str
    experiment_id: str
    reused_existing: bool
    output_path: Path
    manifest_path: Path
    duration_seconds: float | None
    peak_rss_bytes: int | None
    dry_run: bool
    validate_only: bool
    status_only: bool
    completed: bool
    summary: dict[str, Any] | None


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    train_total_loss: float
    train_policy_loss: float
    train_value_loss: float
    validation_total_loss: float
    validation_policy_loss: float
    validation_value_loss: float
    validation_policy_mrr: float
    duration_seconds: float


@dataclass(frozen=True)
class EvalSummary:
    total_loss: float
    policy_loss: float
    value_loss: float
    policy_metrics: dict[str, Any]
    value_metrics: dict[str, Any]
    ranking_groups: list[RankingGroup]
    value_probs: np.ndarray


class _SingleWriterLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._acquired = False

    def __enter__(self) -> _SingleWriterLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._path.exists():
            payload = self._read_payload()
            pid = payload.get("pid") if isinstance(payload, dict) else None
            if isinstance(pid, int) and not psutil.pid_exists(pid):
                self._path.unlink(missing_ok=True)

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(str(self._path), flags)
        except FileExistsError as exc:
            raise RuntimeError(
                f"Another writer is active for this experiment lock: {self._path.as_posix()}"
            ) from exc

        payload = {
            "pid": os.getpid(),
            "created_at_utc": _utc_now_iso(),
        }
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
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


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _git_commit() -> str | None:
    head = Path(".git") / "HEAD"
    if not head.exists():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref = text[4:].strip()
        ref_path = Path(".git") / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()[:40]
        return None
    return text[:40] if text else None


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(requested: str) -> torch.device:
    lowered = requested.strip().lower()
    if lowered == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if lowered == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but CUDA is not available")
        return torch.device("cuda")
    if lowered == "cpu":
        return torch.device("cpu")
    raise RuntimeError(f"Unsupported device selection: {requested!r}")


def _set_runtime_threads(threads: int) -> None:
    torch.set_num_threads(max(1, int(threads)))


def _modeling_manifest_override(config: NeuralConfig, override: str | None) -> NeuralConfig:
    resolved = resolve_modeling_manifest_override_or_env(override)
    if resolved is None:
        return config
    return apply_neural_cli_overrides(
        config,
        modeling_manifest_path=resolved.as_posix(),
        baseline_reference_manifest_path=None,
        output_root=None,
        max_train_positions=None,
        max_validation_positions=None,
        max_test_positions=None,
        device=None,
    )


def _neural_preflight(config: NeuralConfig) -> PreflightSummary:
    shim = SimpleNamespace(
        input=SimpleNamespace(modeling_manifest_path=config.input.modeling_manifest_path),
        versions=SimpleNamespace(
            feature_schema_version=config.versions.feature_schema_version,
            split_definition_version=config.versions.split_definition_version,
        ),
    )
    return _preflight_modeling_dataset(cast(Any, shim))


def _selection_inputs(
    config: NeuralConfig,
    preflight: PreflightSummary,
) -> tuple[
    list[PositionExample],
    list[PositionExample],
    list[PositionExample],
    list[PositionExample],
    TrainingPopulationSummary,
    SelectionFingerprints,
]:
    novel_position_ids = _novel_position_ids(preflight.novel_position_paths)

    train_examples = _load_position_examples(
        preflight.policy_paths_by_split["train"],
        split="train",
        limit=config.selection.max_train_positions,
        seed=config.runtime.seed,
        novel_position_ids=novel_position_ids,
    )
    validation_examples = _load_position_examples(
        preflight.policy_paths_by_split["validation"],
        split="validation",
        limit=config.selection.max_validation_positions,
        seed=config.runtime.seed,
        novel_position_ids=novel_position_ids,
    )
    test_examples = _load_position_examples(
        preflight.policy_paths_by_split["test"],
        split="test",
        limit=config.selection.max_test_positions,
        seed=config.runtime.seed,
        novel_position_ids=novel_position_ids,
    )

    if not train_examples:
        raise RuntimeError("No train examples selected for neural training")
    if not validation_examples:
        raise RuntimeError("No validation examples selected for neural training")
    if not test_examples:
        raise RuntimeError("No test examples selected for neural training")

    selected_examples_by_split = {
        "train": train_examples,
        "validation": validation_examples,
        "test": test_examples,
    }

    training_examples, training_population = _select_training_population(
        train_examples,
        mode=config.selection.training_population_mode,
        all_selected_examples_by_split=selected_examples_by_split,
        game_assignment_paths_by_split=preflight.game_assignment_paths_by_split,
        player_holdout_seed=preflight.player_holdout_seed,
        player_holdout_hash_modulus=preflight.player_holdout_hash_modulus,
        player_holdout_hash_threshold=preflight.player_holdout_hash_threshold,
    )

    fingerprints = selection_fingerprints(
        train_selected=train_examples,
        validation_selected=validation_examples,
        test_selected=test_examples,
        train_population_used=training_examples,
    )

    return (
        train_examples,
        validation_examples,
        test_examples,
        training_examples,
        training_population,
        fingerprints,
    )


def _identity_payload(
    *,
    config: NeuralConfig,
    preflight: PreflightSummary,
    fingerprints: SelectionFingerprints,
    git_commit: str | None,
) -> dict[str, Any]:
    effective = neural_config_identity_payload(config)
    effective["input"].pop("modeling_manifest_path", None)
    effective["input"].pop("baseline_reference_manifest_path", None)

    return {
        "modeling_dataset_id": preflight.modeling_dataset_id,
        "modeling_manifest_sha256": preflight.modeling_manifest_sha256,
        "feature_schema_version": preflight.feature_schema_version,
        "split_definition_version": preflight.split_definition_version,
        "selection_fingerprints": {
            "split_key_hashes": fingerprints.split_key_hashes,
            "train_population_key_hash": fingerprints.train_population_key_hash,
        },
        "excluded_training_features": list(EXCLUDED_TRAINING_FEATURES),
        "allowed_context_features": list(ALLOWED_CONTEXT_FEATURES),
        "effective_config": effective,
        "git_commit": git_commit,
    }


def _experiment_id(identity_payload: dict[str, Any]) -> str:
    return sha256_text(canonical_json(identity_payload))


def _validate_existing_experiment(
    *,
    experiment_path: Path,
    expected_experiment_id: str,
    expected_identity_hash: str,
) -> dict[str, Any]:
    manifest = read_json_object(experiment_path / "_manifest.json")
    validate_manifest_identity(
        manifest=manifest,
        expected_paths={
            ("status",): "complete",
            ("experiment_id",): expected_experiment_id,
            ("identity_payload_sha256",): expected_identity_hash,
        },
    )

    if not (experiment_path / "_SUCCESS").exists():
        raise RuntimeError("Existing neural experiment missing _SUCCESS marker")

    integrity_payload = manifest.get("artifact_integrity")
    if not isinstance(integrity_payload, dict):
        raise RuntimeError("Existing neural experiment manifest missing artifact_integrity")

    for required_file in NEURAL_REQUIRED_ARTIFACTS:
        if required_file not in integrity_payload:
            raise RuntimeError(
                "Existing neural experiment missing artifact_integrity entry for "
                + required_file
            )

    for file_name, record in integrity_payload.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"Invalid artifact_integrity entry for {file_name!r}")

        expected_sha = str(record.get("sha256", "")).strip()
        expected_size = record.get("size_bytes")
        if not expected_sha:
            raise RuntimeError(f"Missing sha256 for artifact {file_name!r}")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise RuntimeError(f"Invalid size for artifact {file_name!r}")

        artifact_path = experiment_path / file_name
        if not artifact_path.exists():
            raise RuntimeError(f"Missing artifact file {file_name!r}")

        actual_size = artifact_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Existing experiment artifact integrity mismatch (size): {file_name!r}"
            )

        actual_sha = _file_sha256(artifact_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Existing experiment artifact integrity mismatch (sha256): {file_name!r}"
            )

    return manifest


def _batches_for_epoch(
    *,
    size: int,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> list[np.ndarray]:
    indices = np.arange(size, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(indices)

    batches: list[np.ndarray] = []
    for start in range(0, size, batch_size):
        batches.append(indices[start : start + batch_size])
    return batches


def _batch_tensors(
    examples: list[EncodedExample],
    index_batch: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch_examples = [examples[int(idx)] for idx in index_batch]

    board = torch.tensor(
        np.stack([item.board_tensor for item in batch_examples]),
        dtype=torch.float32,
        device=device,
    )
    legal_mask = torch.tensor(
        np.stack([item.legal_mask for item in batch_examples]),
        dtype=torch.bool,
        device=device,
    )
    policy_targets = torch.tensor(
        [item.policy_target_action_index for item in batch_examples],
        dtype=torch.int64,
        device=device,
    )
    value_targets = torch.tensor(
        [item.value_target_index for item in batch_examples],
        dtype=torch.int64,
        device=device,
    )
    continuous = torch.tensor(
        np.stack([item.continuous_context for item in batch_examples]),
        dtype=torch.float32,
        device=device,
    )
    rating_band = torch.tensor(
        [item.rating_band_index for item in batch_examples],
        dtype=torch.int64,
        device=device,
    )
    time_control = torch.tensor(
        [item.time_control_index for item in batch_examples],
        dtype=torch.int64,
        device=device,
    )

    return {
        "board": board,
        "legal_mask": legal_mask,
        "policy_targets": policy_targets,
        "value_targets": value_targets,
        "continuous": continuous,
        "rating_band": rating_band,
        "time_control": time_control,
    }


def _evaluate_examples(
    *,
    model: PolicyValueResidualCNN,
    examples_encoded: list[EncodedExample],
    examples_raw: list[PositionExample],
    batch_size: int,
    seed: int,
    epoch: int,
    device: torch.device,
    ranking_cutoffs: tuple[int, ...],
    ece_bins: int,
    loss_weights: LossWeights,
) -> EvalSummary:
    model.eval()

    all_policy_logits: list[np.ndarray] = []
    all_value_probs: list[np.ndarray] = []

    total_rows = 0
    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0

    with torch.no_grad():
        for batch_indices in _batches_for_epoch(
            size=len(examples_encoded),
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
            shuffle=False,
        ):
            tensors = _batch_tensors(examples_encoded, batch_indices, device=device)

            policy_logits, value_logits = model(
                tensors["board"],
                tensors["continuous"],
                tensors["rating_band"],
                tensors["time_control"],
            )
            losses = compute_supervised_losses(
                policy_logits=policy_logits,
                value_logits=value_logits,
                legal_mask=tensors["legal_mask"],
                policy_targets=tensors["policy_targets"],
                value_targets=tensors["value_targets"],
                weights=loss_weights,
            )

            row_count = int(batch_indices.shape[0])
            total_rows += row_count
            total_loss_sum += float(losses.total.detach().cpu().item()) * row_count
            policy_loss_sum += float(losses.policy.detach().cpu().item()) * row_count
            value_loss_sum += float(losses.value.detach().cpu().item()) * row_count

            all_policy_logits.append(policy_logits.detach().cpu().numpy())
            all_value_probs.append(torch.softmax(value_logits, dim=1).detach().cpu().numpy())

    if total_rows == 0:
        raise RuntimeError("Evaluation received zero rows")

    policy_logits_np = np.concatenate(all_policy_logits, axis=0)
    value_probs_np = np.concatenate(all_value_probs, axis=0)

    ranking_groups: list[RankingGroup] = []
    for idx, encoded in enumerate(examples_encoded):
        legal_indices = np.flatnonzero(encoded.legal_mask).astype(np.int64)
        if legal_indices.size == 0:
            raise RuntimeError("Encoded example has no legal actions")

        labels = tuple(
            1 if int(action_index) == encoded.policy_target_action_index else 0
            for action_index in legal_indices.tolist()
        )
        scores = tuple(
            float(policy_logits_np[idx, int(action_index)])
            for action_index in legal_indices
        )

        ranking_groups.append(
            RankingGroup(
                game_id=encoded.game_id,
                group_id=f"{encoded.game_id}:{encoded.ply}",
                action_indices=tuple(int(action_index) for action_index in legal_indices.tolist()),
                labels=labels,
                scores=scores,
                metadata=encoded.metadata,
            )
        )

    policy_metrics = compute_ranking_metrics(ranking_groups, ranking_cutoffs)
    value_metrics = _logistic_metrics(
        examples=examples_raw,
        probs=value_probs_np,
        ece_bins=ece_bins,
    )

    return EvalSummary(
        total_loss=(total_loss_sum / total_rows),
        policy_loss=(policy_loss_sum / total_rows),
        value_loss=(value_loss_sum / total_rows),
        policy_metrics=policy_metrics,
        value_metrics=value_metrics,
        ranking_groups=ranking_groups,
        value_probs=value_probs_np,
    )


def _checkpoint_state(
    *,
    experiment_id: str,
    identity_hash: str,
    next_epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int | None,
    no_improve_epochs: int,
    model: PolicyValueResidualCNN,
    optimizer: torch.optim.Optimizer,
    preprocessor: NeuralPreprocessor,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "identity_hash": identity_hash,
        "next_epoch": next_epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "no_improve_epochs": no_improve_epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "preprocessor": preprocessor.to_dict(),
        "history": history,
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Checkpoint payload is not a dictionary")
    return payload


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _status_payload(
    *,
    work_root: Path,
    experiment_root: Path,
    experiment_id: str,
) -> dict[str, Any]:
    manifest_path = experiment_root / "_manifest.json"
    if manifest_path.exists() and (experiment_root / "_SUCCESS").exists():
        manifest = read_json_object(manifest_path)
        return {
            "status": "complete",
            "experiment_id": experiment_id,
            "output_path": _display_path(experiment_root),
            "manifest_path": _display_path(manifest_path),
            "manifest": manifest,
        }

    progress_path = work_root / "progress.json"
    if progress_path.exists():
        return {
            "status": "in_progress_or_interrupted",
            "experiment_id": experiment_id,
            "work_root": _display_path(work_root),
            "progress": read_json_object(progress_path),
        }

    return {
        "status": "not_started",
        "experiment_id": experiment_id,
        "work_root": _display_path(work_root),
    }


def _baseline_compatibility(
    *,
    config: NeuralConfig,
    preflight: PreflightSummary,
    actual_fingerprints: SelectionFingerprints,
) -> dict[str, Any] | None:
    baseline_path = config.input.baseline_reference_manifest_path
    if baseline_path is None:
        return None
    if not baseline_path.exists():
        return {
            "compatible": False,
            "reason": (
                "baseline_reference_manifest_path does not exist: "
                f"{baseline_path.as_posix()}"
            ),
        }

    baseline_manifest = read_json_object(baseline_path)
    identity_payload = baseline_manifest.get("identity_payload")
    if not isinstance(identity_payload, dict):
        return {
            "compatible": False,
            "reason": "Baseline manifest missing identity_payload",
        }

    baseline_manifest_sha = str(identity_payload.get("modeling_manifest_sha256", "")).strip()
    if baseline_manifest_sha != preflight.modeling_manifest_sha256:
        return {
            "compatible": False,
            "reason": (
                "Modeling manifest identity mismatch against baseline reference: "
                "baseline="
                f"{baseline_manifest_sha!r}, "
                "current="
                f"{preflight.modeling_manifest_sha256!r}"
            ),
        }

    effective = identity_payload.get("effective_config")
    if not isinstance(effective, dict):
        return {
            "compatible": False,
            "reason": "Baseline manifest identity payload missing effective_config",
        }

    try:
        limits = cast(dict[str, Any], effective["limits"])
        training_population = cast(dict[str, Any], effective["training_population"])
        runtime = cast(dict[str, Any], effective["runtime"])
    except KeyError as exc:
        return {
            "compatible": False,
            "reason": f"Baseline effective_config missing key: {exc}",
        }

    baseline_seed = int(runtime.get("seed", 0))
    baseline_mode = str(training_population.get("mode", ""))
    baseline_train_limit = cast(int | None, limits.get("max_train_positions"))
    baseline_validation_limit = cast(int | None, limits.get("max_validation_positions"))
    baseline_test_limit = cast(int | None, limits.get("max_test_positions"))

    novel_position_ids = _novel_position_ids(preflight.novel_position_paths)
    train_examples = _load_position_examples(
        preflight.policy_paths_by_split["train"],
        split="train",
        limit=baseline_train_limit,
        seed=baseline_seed,
        novel_position_ids=novel_position_ids,
    )
    validation_examples = _load_position_examples(
        preflight.policy_paths_by_split["validation"],
        split="validation",
        limit=baseline_validation_limit,
        seed=baseline_seed,
        novel_position_ids=novel_position_ids,
    )
    test_examples = _load_position_examples(
        preflight.policy_paths_by_split["test"],
        split="test",
        limit=baseline_test_limit,
        seed=baseline_seed,
        novel_position_ids=novel_position_ids,
    )

    baseline_train_population, _ = _select_training_population(
        train_examples,
        mode=baseline_mode,
        all_selected_examples_by_split={
            "train": train_examples,
            "validation": validation_examples,
            "test": test_examples,
        },
        game_assignment_paths_by_split=preflight.game_assignment_paths_by_split,
        player_holdout_seed=preflight.player_holdout_seed,
        player_holdout_hash_modulus=preflight.player_holdout_hash_modulus,
        player_holdout_hash_threshold=preflight.player_holdout_hash_threshold,
    )

    expected = {
        "train": selected_key_hash(train_examples),
        "validation": selected_key_hash(validation_examples),
        "test": selected_key_hash(test_examples),
    }
    expected_train_hash = selected_key_hash(baseline_train_population)

    report = compare_selection_fingerprints(
        expected_split_key_hashes=expected,
        expected_train_population_key_hash=expected_train_hash,
        actual=actual_fingerprints,
    )

    return {
        "compatible": report.compatible,
        "reason": report.reason,
        "expected": {
            "split_key_hashes": expected,
            "train_population_key_hash": expected_train_hash,
        },
        "actual": {
            "split_key_hashes": actual_fingerprints.split_key_hashes,
            "train_population_key_hash": actual_fingerprints.train_population_key_hash,
        },
    }


def _result_payload(result: NeuralRunResult) -> dict[str, Any]:
    payload = {
        "run_id": result.run_id,
        "experiment_id": result.experiment_id,
        "reused_existing": result.reused_existing,
        "output_path": _display_path(result.output_path),
        "manifest_path": _display_path(result.manifest_path),
        "duration_seconds": result.duration_seconds,
        "peak_rss_bytes": result.peak_rss_bytes,
        "dry_run": result.dry_run,
        "validate_only": result.validate_only,
        "status_only": result.status_only,
        "completed": result.completed,
    }
    if result.summary is not None:
        payload["summary"] = result.summary
    return payload


def run_neural(
    *,
    config_path: Path,
    modeling_manifest_override: str | None = None,
    baseline_reference_manifest_override: str | None = None,
    output_root_override: str | None = None,
    max_train_positions_override: int | None = None,
    max_validation_positions_override: int | None = None,
    max_test_positions_override: int | None = None,
    device_override: str | None = None,
    dry_run: bool = False,
    validate_only: bool = False,
    resume: bool = False,
    status_only: bool = False,
    max_epochs_this_session: int | None = None,
) -> NeuralRunResult:
    config = load_neural_config(config_path)
    config = apply_neural_cli_overrides(
        config,
        modeling_manifest_path=modeling_manifest_override,
        baseline_reference_manifest_path=baseline_reference_manifest_override,
        output_root=output_root_override,
        max_train_positions=max_train_positions_override,
        max_validation_positions=max_validation_positions_override,
        max_test_positions=max_test_positions_override,
        device=device_override,
    )
    config = _modeling_manifest_override(config, modeling_manifest_override)

    preflight = _neural_preflight(config)
    (
        train_examples,
        validation_examples,
        test_examples,
        training_examples,
        training_population,
        fingerprints,
    ) = _selection_inputs(config, preflight)

    git_commit = _git_commit()
    identity_payload = _identity_payload(
        config=config,
        preflight=preflight,
        fingerprints=fingerprints,
        git_commit=git_commit,
    )
    identity_hash = sha256_text(canonical_json(identity_payload))
    experiment_id = _experiment_id(identity_payload)

    run_id = f"neural-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    output_root = config.output.output_root
    work_root = output_root / "work" / experiment_id
    experiment_root = output_root / "experiments" / experiment_id
    manifest_path = experiment_root / "_manifest.json"

    baseline_compatibility = _baseline_compatibility(
        config=config,
        preflight=preflight,
        actual_fingerprints=fingerprints,
    )

    if status_only:
        summary = _status_payload(
            work_root=work_root,
            experiment_root=experiment_root,
            experiment_id=experiment_id,
        )
        summary["baseline_compatibility"] = baseline_compatibility
        return NeuralRunResult(
            run_id=run_id,
            experiment_id=experiment_id,
            reused_existing=False,
            output_path=experiment_root,
            manifest_path=manifest_path,
            duration_seconds=None,
            peak_rss_bytes=None,
            dry_run=False,
            validate_only=False,
            status_only=True,
            completed=(summary.get("status") == "complete"),
            summary=summary,
        )

    if experiment_root.exists():
        _validate_existing_experiment(
            experiment_path=experiment_root,
            expected_experiment_id=experiment_id,
            expected_identity_hash=identity_hash,
        )
        summary = {
            "status": "reused_existing_complete_experiment",
            "baseline_compatibility": baseline_compatibility,
        }
        return NeuralRunResult(
            run_id=run_id,
            experiment_id=experiment_id,
            reused_existing=True,
            output_path=experiment_root,
            manifest_path=manifest_path,
            duration_seconds=None,
            peak_rss_bytes=None,
            dry_run=False,
            validate_only=validate_only,
            status_only=False,
            completed=True,
            summary=summary,
        )

    if dry_run:
        summary = {
            "status": "dry_run",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "modeling_manifest_path": _display_path(preflight.modeling_manifest_path),
            "modeling_dataset_id": preflight.modeling_dataset_id,
            "selection_fingerprints": {
                "split_key_hashes": fingerprints.split_key_hashes,
                "train_population_key_hash": fingerprints.train_population_key_hash,
            },
            "counts": {
                "split_row_counts_full": preflight.split_row_counts,
                "split_row_counts_selected": {
                    "train": len(train_examples),
                    "validation": len(validation_examples),
                    "test": len(test_examples),
                },
                "training_population": asdict(training_population),
            },
            "baseline_compatibility": baseline_compatibility,
            "output_path": _display_path(experiment_root),
        }
        return NeuralRunResult(
            run_id=run_id,
            experiment_id=experiment_id,
            reused_existing=False,
            output_path=experiment_root,
            manifest_path=manifest_path,
            duration_seconds=None,
            peak_rss_bytes=None,
            dry_run=True,
            validate_only=False,
            status_only=False,
            completed=False,
            summary=summary,
        )

    preprocessor = fit_preprocessor(training_examples)
    train_encoded = encode_examples(training_examples, split="train", preprocessor=preprocessor)
    validation_encoded = encode_examples(
        validation_examples,
        split="validation",
        preprocessor=preprocessor,
    )
    test_encoded = encode_examples(test_examples, split="test", preprocessor=preprocessor)

    if validate_only:
        summary = {
            "status": "validate_only_pass",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "counts": {
                "split_row_counts_full": preflight.split_row_counts,
                "split_row_counts_selected": {
                    "train": len(train_examples),
                    "validation": len(validation_examples),
                    "test": len(test_examples),
                },
                "training_population": asdict(training_population),
            },
            "selection_fingerprints": {
                "split_key_hashes": fingerprints.split_key_hashes,
                "train_population_key_hash": fingerprints.train_population_key_hash,
            },
            "baseline_compatibility": baseline_compatibility,
        }
        return NeuralRunResult(
            run_id=run_id,
            experiment_id=experiment_id,
            reused_existing=False,
            output_path=experiment_root,
            manifest_path=manifest_path,
            duration_seconds=None,
            peak_rss_bytes=None,
            dry_run=False,
            validate_only=True,
            status_only=False,
            completed=False,
            summary=summary,
        )

    if max_epochs_this_session is not None and max_epochs_this_session <= 0:
        raise RuntimeError("max_epochs_this_session must be positive when provided")

    _set_runtime_threads(config.runtime.threads)
    device = _resolve_device(config.runtime.device)

    torch.manual_seed(config.runtime.seed)
    np.random.seed(config.runtime.seed)

    model = PolicyValueResidualCNN(
        trunk_channels=config.model.trunk_channels,
        residual_blocks=config.model.residual_blocks,
        context_embedding_dim=config.model.context_embedding_dim,
        context_hidden_dim=config.model.context_hidden_dim,
        value_hidden_dim=config.model.value_hidden_dim,
        sizes=NeuralModelSizes(
            continuous_features=int(train_encoded[0].continuous_context.shape[0]),
            rating_band_vocab_size=len(preprocessor.rating_band_to_index),
            time_control_vocab_size=len(preprocessor.time_control_to_index),
        ),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )

    work_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root = work_root / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    last_checkpoint_path = checkpoints_root / "last.pt"
    best_checkpoint_path = checkpoints_root / "best.pt"
    progress_path = work_root / "progress.json"

    start_epoch = 0
    global_step = 0
    best_metric = float("-inf")
    best_epoch: int | None = None
    no_improve_epochs = 0
    history: list[dict[str, Any]] = []

    if resume:
        if not last_checkpoint_path.exists():
            raise RuntimeError(
                "--resume requested but no last checkpoint exists at "
                f"{last_checkpoint_path.as_posix()}"
            )
        checkpoint = _load_checkpoint(last_checkpoint_path)
        if checkpoint.get("experiment_id") != experiment_id:
            raise RuntimeError("Incompatible resume: experiment_id mismatch")
        if checkpoint.get("identity_hash") != identity_hash:
            raise RuntimeError("Incompatible resume: identity hash mismatch")

        model.load_state_dict(cast(dict[str, Any], checkpoint["model_state"]))
        optimizer.load_state_dict(cast(dict[str, Any], checkpoint["optimizer_state"]))
        restored_preprocessor = NeuralPreprocessor.from_dict(
            cast(dict[str, Any], checkpoint["preprocessor"])
        )
        if restored_preprocessor.to_dict() != preprocessor.to_dict():
            raise RuntimeError("Incompatible resume: preprocessor mismatch")

        start_epoch = int(checkpoint["next_epoch"])
        global_step = int(checkpoint["global_step"])
        best_metric = float(checkpoint["best_metric"])
        best_epoch_payload = checkpoint.get("best_epoch")
        best_epoch = None if best_epoch_payload is None else int(best_epoch_payload)
        no_improve_epochs = int(checkpoint["no_improve_epochs"])
        history = [dict(item) for item in cast(list[dict[str, Any]], checkpoint["history"])]

        np_state = checkpoint.get("numpy_random_state")
        if np_state is not None:
            np.random.set_state(np_state)
        torch_state = checkpoint.get("torch_random_state")
        if isinstance(torch_state, torch.Tensor):
            torch.set_rng_state(torch_state)
    else:
        if last_checkpoint_path.exists():
            raise RuntimeError(
                "Existing incomplete checkpoint found; use --resume to continue or "
                "choose a different output root/identity"
            )

    if start_epoch >= config.training.max_epochs:
        raise RuntimeError(
            "Resume checkpoint next_epoch is already >= configured max_epochs; "
            "no training budget remains"
        )

    weights = LossWeights(policy=config.loss.policy_weight, value=config.loss.value_weight)

    published = False
    session_epoch_count = 0
    training_started = time.perf_counter()
    sampler = PeakRssSampler(interval_seconds=0.05)
    sampler.start()

    with _SingleWriterLock(work_root / ".training.lock"):
        try:
            for epoch in range(start_epoch, config.training.max_epochs):
                if (
                    max_epochs_this_session is not None
                    and session_epoch_count >= max_epochs_this_session
                ):
                    break

                model.train()
                epoch_start = time.perf_counter()
                train_rows = 0
                train_total_sum = 0.0
                train_policy_sum = 0.0
                train_value_sum = 0.0

                last_progress_emit = time.perf_counter()

                for batch_indices in _batches_for_epoch(
                    size=len(train_encoded),
                    batch_size=config.optimization.batch_size,
                    seed=config.runtime.seed,
                    epoch=epoch,
                    shuffle=True,
                ):
                    tensors = _batch_tensors(train_encoded, batch_indices, device=device)

                    optimizer.zero_grad(set_to_none=True)
                    policy_logits, value_logits = model(
                        tensors["board"],
                        tensors["continuous"],
                        tensors["rating_band"],
                        tensors["time_control"],
                    )

                    losses = compute_supervised_losses(
                        policy_logits=policy_logits,
                        value_logits=value_logits,
                        legal_mask=tensors["legal_mask"],
                        policy_targets=tensors["policy_targets"],
                        value_targets=tensors["value_targets"],
                        weights=weights,
                    )

                    losses.total.backward()
                    if config.optimization.grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=config.optimization.grad_clip_norm,
                        )
                    optimizer.step()

                    row_count = int(batch_indices.shape[0])
                    train_rows += row_count
                    train_total_sum += float(losses.total.detach().cpu().item()) * row_count
                    train_policy_sum += float(losses.policy.detach().cpu().item()) * row_count
                    train_value_sum += float(losses.value.detach().cpu().item()) * row_count
                    global_step += 1

                    now = time.perf_counter()
                    if (now - last_progress_emit) >= config.output.progress_update_seconds:
                        _write_progress(
                            progress_path,
                            {
                                "status": "running",
                                "updated_at_utc": _utc_now_iso(),
                                "run_id": run_id,
                                "experiment_id": experiment_id,
                                "epoch": epoch,
                                "global_step": global_step,
                                "train_rows_processed": train_rows,
                                "last_checkpoint": _display_path(last_checkpoint_path),
                            },
                        )
                        last_progress_emit = now

                if train_rows == 0:
                    raise RuntimeError("Training epoch processed zero rows")

                validation_eval = _evaluate_examples(
                    model=model,
                    examples_encoded=validation_encoded,
                    examples_raw=validation_examples,
                    batch_size=config.optimization.batch_size,
                    seed=config.runtime.seed,
                    epoch=epoch,
                    device=device,
                    ranking_cutoffs=config.evaluation.ranking_cutoffs,
                    ece_bins=config.evaluation.ece_bins,
                    loss_weights=weights,
                )

                epoch_duration = max(time.perf_counter() - epoch_start, 0.0)
                entry = EpochSummary(
                    epoch=epoch,
                    train_total_loss=(train_total_sum / train_rows),
                    train_policy_loss=(train_policy_sum / train_rows),
                    train_value_loss=(train_value_sum / train_rows),
                    validation_total_loss=validation_eval.total_loss,
                    validation_policy_loss=validation_eval.policy_loss,
                    validation_value_loss=validation_eval.value_loss,
                    validation_policy_mrr=float(validation_eval.policy_metrics.get("mrr", 0.0)),
                    duration_seconds=epoch_duration,
                )
                history.append(asdict(entry))

                current_metric = entry.validation_policy_mrr
                improved = current_metric > best_metric
                if improved:
                    best_metric = current_metric
                    best_epoch = epoch
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                checkpoint_payload = _checkpoint_state(
                    experiment_id=experiment_id,
                    identity_hash=identity_hash,
                    next_epoch=epoch + 1,
                    global_step=global_step,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    no_improve_epochs=no_improve_epochs,
                    model=model,
                    optimizer=optimizer,
                    preprocessor=preprocessor,
                    history=history,
                )
                _save_checkpoint(last_checkpoint_path, checkpoint_payload)
                if improved:
                    _save_checkpoint(best_checkpoint_path, checkpoint_payload)

                _write_progress(
                    progress_path,
                    {
                        "status": "running",
                        "updated_at_utc": _utc_now_iso(),
                        "run_id": run_id,
                        "experiment_id": experiment_id,
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        "last_checkpoint": _display_path(last_checkpoint_path),
                        "best_checkpoint": _display_path(best_checkpoint_path),
                    },
                )

                session_epoch_count += 1

                if no_improve_epochs > config.training.early_stopping_patience:
                    break

            completed = (
                (start_epoch + session_epoch_count) >= config.training.max_epochs
                or (no_improve_epochs > config.training.early_stopping_patience)
            )

            if not completed:
                _write_progress(
                    progress_path,
                    {
                        "status": "session_limit_reached",
                        "updated_at_utc": _utc_now_iso(),
                        "run_id": run_id,
                        "experiment_id": experiment_id,
                        "epochs_completed_this_session": session_epoch_count,
                        "next_epoch": start_epoch + session_epoch_count,
                        "last_checkpoint": _display_path(last_checkpoint_path),
                    },
                )
                duration_seconds = max(time.perf_counter() - training_started, 0.0)
                peak_rss = sampler.stop()
                return NeuralRunResult(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    reused_existing=False,
                    output_path=experiment_root,
                    manifest_path=manifest_path,
                    duration_seconds=round(duration_seconds, 6),
                    peak_rss_bytes=int(peak_rss),
                    dry_run=False,
                    validate_only=False,
                    status_only=False,
                    completed=False,
                    summary={
                        "status": "session_limit_reached",
                        "next_epoch": start_epoch + session_epoch_count,
                        "baseline_compatibility": baseline_compatibility,
                    },
                )

            if not best_checkpoint_path.exists():
                raise RuntimeError("Training completed but best checkpoint was not written")

            best_payload = _load_checkpoint(best_checkpoint_path)
            model.load_state_dict(cast(dict[str, Any], best_payload["model_state"]))

            test_eval = _evaluate_examples(
                model=model,
                examples_encoded=test_encoded,
                examples_raw=test_examples,
                batch_size=config.optimization.batch_size,
                seed=config.runtime.seed,
                epoch=(best_epoch if best_epoch is not None else 0),
                device=device,
                ranking_cutoffs=config.evaluation.ranking_cutoffs,
                ece_bins=config.evaluation.ece_bins,
                loss_weights=weights,
            )

            player_holdout_interpretation = _player_holdout_interpretation(
                training_population.mode
            )
            subgroup_metrics = {
                "neural_policy_test": _annotate_player_holdout_subgroup(
                    _ranking_subgroups(
                        test_eval.ranking_groups,
                        config.evaluation.ranking_cutoffs,
                    ),
                    interpretation=player_holdout_interpretation,
                ),
                "neural_value_test": _annotate_player_holdout_subgroup(
                    _logistic_subgroups(
                        test_examples,
                        test_eval.value_probs,
                        ece_bins=config.evaluation.ece_bins,
                    ),
                    interpretation=player_holdout_interpretation,
                ),
            }

            aggregate_metrics = {
                "training_population": asdict(training_population),
                "policy": {
                    "test": test_eval.policy_metrics,
                    "selection_metric": config.training.selection_metric,
                    "best_validation_policy_mrr": best_metric,
                    "best_epoch": best_epoch,
                },
                "value": {
                    "test": test_eval.value_metrics,
                    "class_order": list(WDL_CLASS_ORDER),
                },
                "losses": {
                    "test_total_loss": test_eval.total_loss,
                    "test_policy_loss": test_eval.policy_loss,
                    "test_value_loss": test_eval.value_loss,
                },
            }

            leakage_audit = {
                "game_id_overlap_counts": preflight.game_id_overlap_counts,
                "normalized_fen_overlap_counts": preflight.normalized_fen_overlap_counts,
                "player_overlap_counts": preflight.player_overlap_counts,
                "selection_fingerprints": {
                    "split_key_hashes": fingerprints.split_key_hashes,
                    "train_population_key_hash": fingerprints.train_population_key_hash,
                },
                "excluded_training_features": list(EXCLUDED_TRAINING_FEATURES),
                "allowed_context_features": list(ALLOWED_CONTEXT_FEATURES),
                "confirmations": {
                    "train_only_preprocessing": True,
                    "validation_test_not_used_for_fit": True,
                    "opening_metadata_excluded_from_features": True,
                },
                "baseline_compatibility": baseline_compatibility,
            }

            leakage_md = "\n".join(
                [
                    "# Neural Leakage Audit",
                    "",
                    "## Split overlap checks",
                    "- game_id overlaps: "
                    + json.dumps(preflight.game_id_overlap_counts, sort_keys=True),
                    "- normalized FEN overlaps: "
                    + json.dumps(preflight.normalized_fen_overlap_counts, sort_keys=True),
                    "- player overlaps: "
                    + json.dumps(preflight.player_overlap_counts, sort_keys=True),
                    "",
                    "## Feature policy",
                    "- excluded training features: " + ", ".join(EXCLUDED_TRAINING_FEATURES),
                    "- allowed context features: " + ", ".join(ALLOWED_CONTEXT_FEATURES),
                    "",
                    "## Confirmations",
                    "- train-only preprocessing: yes",
                    "- validation/test not used for fit: yes",
                    "- opening metadata excluded from training features: yes",
                ]
            )

            staging_root = output_root / "staging" / f"{experiment_id}__{run_id}"
            _safe_rmtree(staging_root)
            staging_root.mkdir(parents=True, exist_ok=False)

            effective_config = neural_config_identity_payload(config)
            (staging_root / "effective_config.json").write_text(
                json.dumps(effective_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (staging_root / "experiment_identity_payload.json").write_text(
                json.dumps(identity_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (staging_root / "feature_preprocessing.json").write_text(
                json.dumps(preprocessor.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (staging_root / "aggregate_metrics.json").write_text(
                json.dumps(aggregate_metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (staging_root / "subgroup_metrics.json").write_text(
                json.dumps(subgroup_metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (staging_root / "leakage_audit.json").write_text(
                json.dumps(leakage_audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (staging_root / "leakage_audit.md").write_text(leakage_md + "\n", encoding="utf-8")
            (staging_root / "training_history.json").write_text(
                json.dumps(history, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            shutil.copy2(last_checkpoint_path, staging_root / "checkpoint_last.pt")
            shutil.copy2(best_checkpoint_path, staging_root / "checkpoint_best.pt")

            best_model_state = cast(dict[str, Any], best_payload["model_state"])
            torch.save(best_model_state, staging_root / "model_state_best.pt")
            torch.save(
                cast(dict[str, Any], _load_checkpoint(last_checkpoint_path)["model_state"]),
                staging_root / "model_state_last.pt",
            )

            peak_rss = sampler.stop()
            duration_seconds = max(time.perf_counter() - training_started, 0.0)
            version_info = sys.version_info
            environment_payload = {
                "python_version": (
                    f"{version_info.major}."
                    f"{version_info.minor}."
                    f"{version_info.micro}"
                ),
                "torch_version": torch.__version__,
                "device": str(device),
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "threads": config.runtime.threads,
                "parameter_count": parameter_count(model),
                "git_commit": git_commit,
                "duration_seconds": round(duration_seconds, 6),
                "peak_rss_bytes": int(peak_rss),
            }
            (staging_root / "environment.json").write_text(
                json.dumps(environment_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            artifact_files = sorted(
                file.name
                for file in staging_root.iterdir()
                if file.is_file() and file.name not in {"_manifest.json", "_SUCCESS"}
            )
            artifact_integrity = {
                file_name: {
                    "size_bytes": int((staging_root / file_name).stat().st_size),
                    "sha256": _file_sha256(staging_root / file_name),
                }
                for file_name in artifact_files
            }

            manifest = {
                "status": "complete",
                "run_id": run_id,
                "experiment_id": experiment_id,
                "identity_payload_sha256": identity_hash,
                "identity_payload": identity_payload,
                "modeling_input": {
                    "manifest_path": _display_path(preflight.modeling_manifest_path),
                    "modeling_dataset_id": preflight.modeling_dataset_id,
                    "modeling_manifest_sha256": preflight.modeling_manifest_sha256,
                    "collection_id": preflight.collection_id,
                    "source_months": list(preflight.source_months),
                    "feature_schema_version": preflight.feature_schema_version,
                    "split_definition_version": preflight.split_definition_version,
                },
                "counts": {
                    "split_row_counts_full": preflight.split_row_counts,
                    "split_row_counts_selected": {
                        "train": len(train_examples),
                        "validation": len(validation_examples),
                        "test": len(test_examples),
                    },
                    "training_population": asdict(training_population),
                },
                "selection_fingerprints": {
                    "split_key_hashes": fingerprints.split_key_hashes,
                    "train_population_key_hash": fingerprints.train_population_key_hash,
                },
                "versions": {
                    "neural_pipeline_version": config.versions.neural_pipeline_version,
                    "feature_schema_version": config.versions.feature_schema_version,
                    "split_definition_version": config.versions.split_definition_version,
                    "board_encoding_version": config.versions.board_encoding_version,
                    "action_encoding_version": config.versions.action_encoding_version,
                    "position_normalization_version": (
                        config.versions.position_normalization_version
                    ),
                },
                "performance": {
                    "duration_seconds": round(duration_seconds, 6),
                    "peak_rss_bytes": int(peak_rss),
                },
                "created_at_utc": _utc_now_iso(),
                "git_commit": git_commit,
                "artifacts": artifact_files,
                "artifact_integrity": artifact_integrity,
                "baseline_compatibility": baseline_compatibility,
            }

            (staging_root / "_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (staging_root / "_SUCCESS").write_text("ok\n", encoding="utf-8")

            if experiment_root.exists():
                raise RuntimeError(
                    "Cannot publish complete neural experiment; destination already exists: "
                    f"{experiment_root.as_posix()}"
                )

            experiment_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_root, experiment_root)
            published = True

            _write_progress(
                progress_path,
                {
                    "status": "complete",
                    "updated_at_utc": _utc_now_iso(),
                    "run_id": run_id,
                    "experiment_id": experiment_id,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "published_manifest_path": _display_path(experiment_root / "_manifest.json"),
                },
            )

            summary = {
                "status": "complete",
                "best_epoch": best_epoch,
                "best_validation_policy_mrr": best_metric,
                "baseline_compatibility": baseline_compatibility,
            }

            return NeuralRunResult(
                run_id=run_id,
                experiment_id=experiment_id,
                reused_existing=False,
                output_path=experiment_root,
                manifest_path=experiment_root / "_manifest.json",
                duration_seconds=round(duration_seconds, 6),
                peak_rss_bytes=int(peak_rss),
                dry_run=False,
                validate_only=False,
                status_only=False,
                completed=True,
                summary=summary,
            )

        except KeyboardInterrupt:
            peak_rss = sampler.stop()
            _write_progress(
                progress_path,
                {
                    "status": "interrupted",
                    "updated_at_utc": _utc_now_iso(),
                    "run_id": run_id,
                    "experiment_id": experiment_id,
                    "last_checkpoint": _display_path(last_checkpoint_path),
                },
            )
            return NeuralRunResult(
                run_id=run_id,
                experiment_id=experiment_id,
                reused_existing=False,
                output_path=experiment_root,
                manifest_path=manifest_path,
                duration_seconds=None,
                peak_rss_bytes=int(peak_rss),
                dry_run=False,
                validate_only=False,
                status_only=False,
                completed=False,
                summary={
                    "status": "interrupted",
                    "baseline_compatibility": baseline_compatibility,
                },
            )
        finally:
            if not published:
                sampler.stop()


def run_neural_result_json(result: NeuralRunResult) -> dict[str, Any]:
    return _result_payload(result)
