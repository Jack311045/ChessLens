from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest
import torch
import yaml

import chesslens.modeling.run_neural_hpo as run_neural_hpo_module
from chesslens.modeling.optuna_study_config import load_neural_hpo_config
from tests.integration.test_run_baselines import _write_modeling_fixture
from tests.integration.test_run_neural import _write_neural_config

_SPLIT_KEY_HASHES: dict[str, str] = {
    "train": "train-hash-fixture",
    "validation": "validation-hash-fixture",
    "test": "test-hash-fixture",
}

_TRAINING_POPULATION: dict[str, Any] = {
    "mode": "player_disjoint",
    "full_train_row_count": 10,
    "eligible_train_row_count": 8,
    "training_row_count_used": 8,
}

_SPLIT_ROW_COUNTS_FULL: dict[str, int] = {
    "train": 120,
    "validation": 40,
    "test": 40,
}

_SPLIT_ROW_COUNTS_SELECTED: dict[str, int] = {
    "train": 10,
    "validation": 4,
    "test": 4,
}


def _write_hpo_config(
    path: Path,
    *,
    reference_neural_config: Path,
    study_output_root: Path,
    max_total_trials: int = 4,
    max_new_trials_per_invocation: int = 2,
    learning_rate_high: float = 0.004,
) -> None:
    path.write_text(
        "\n".join(
            [
                "reference:",
                f"  neural_config_path: {reference_neural_config.as_posix()}",
                "study:",
                "  name: fixture_hpo",
                f"  output_root: {study_output_root.as_posix()}",
                "  evaluation_scope: validation_only",
                "  objective_metric: validation.policy.mrr",
                "budget:",
                f"  max_total_trials: {max_total_trials}",
                f"  max_new_trials_per_invocation: {max_new_trials_per_invocation}",
                "search_space:",
                "  learning_rate:",
                "    low: 0.0005",
                f"    high: {learning_rate_high}",
                "    log: true",
                "  weight_decay:",
                "    low: 0.00001",
                "    high: 0.0005",
                "    log: true",
                "  value_weight_choices:",
                "    - 0.2",
                "    - 0.4",
                "    - 0.6",
                "runtime:",
                "  seed: 20240131",
                "  threads: 1",
                "  device: cpu",
                "pruning:",
                "  startup_trials: 1",
                "  warmup_epochs: 1",
                "behavior:",
                "  strict: true",
                "execution:",
                "  trial_concurrency: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_hpo(
    *,
    config: Any,
    resume_running_trials: bool,
    max_new_trials_override: int | None,
    dry_run: bool,
    status_only: bool,
    export_best_config: Path | None,
    export_final_output_root: Path | None = None,
) -> dict[str, Any]:
    return run_neural_hpo_module._run_hpo(
        config=config,
        resume_running_trials=resume_running_trials,
        max_new_trials_override=max_new_trials_override,
        dry_run=dry_run,
        status_only=status_only,
        export_best_config=export_best_config,
        export_final_output_root=export_final_output_root,
        mlflow_tracking_config=None,
        mlflow_run_origin="current",
        mlflow_include_model_weights=False,
    )


def _trial_params_by_number(study_root: Path, trial_number: int) -> dict[str, Any]:
    payload = json.loads((study_root / "reports" / "trials.json").read_text(encoding="utf-8"))
    trials = payload.get("trials")
    if not isinstance(trials, list):
        raise AssertionError("trials.json missing trials list")
    for row in trials:
        if isinstance(row, dict) and int(row.get("number", -1)) == trial_number:
            params = row.get("params")
            if isinstance(params, dict):
                return params
    raise AssertionError(f"trial #{trial_number} not found")


def _write_completed_trial_artifacts(
    *,
    manifest_path: Path,
    metric: float,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "selection_fingerprints": {
                    "split_key_hashes": dict(_SPLIT_KEY_HASHES),
                    "train_population_key_hash": "train-pop-hash-fixture",
                },
                "counts": {
                    "training_population": dict(_TRAINING_POPULATION),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (manifest_path.parent / "aggregate_metrics.json").write_text(
        json.dumps(
            {
                "policy": {
                    "best_validation_policy_mrr": metric,
                    "validation": {"mrr": metric},
                },
                "value": {
                    "validation": {
                        "accuracy": 0.5,
                        "macro_f1": 0.4,
                        "ece": 0.1,
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_neural_stub(
    *,
    interrupt_first_trial: bool,
    session_limit_on_resume: bool = False,
    write_checkpoint_on_session_limit: bool = True,
) -> tuple[Any, list[dict[str, Any]]]:
    calls: dict[str, int] = {}
    call_log: list[dict[str, Any]] = []

    def _stub_run_neural(
        *,
        config_path: Path,
        dry_run: bool = False,
        resume: bool = False,
        max_epochs_this_session: int | None = None,
        **_: Any,
    ) -> Any:
        trial_key = config_path.parent.name
        calls[trial_key] = calls.get(trial_key, 0) + 1

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise AssertionError("stub expected YAML object payload")
        output_root = Path(str(payload["output"]["output_root"]))
        modeling_manifest_path = str(payload.get("input", {}).get("modeling_manifest_path", ""))

        call_log.append(
            {
                "config_path": config_path.as_posix(),
                "dry_run": dry_run,
                "resume": resume,
                "max_epochs_this_session": max_epochs_this_session,
            }
        )

        if dry_run:
            return SimpleNamespace(
                completed=False,
                experiment_id=f"dry-{trial_key}",
                manifest_path=output_root / "experiments" / f"dry-{trial_key}" / "_manifest.json",
                summary={
                    "status": "dry_run",
                    "modeling_manifest_path": modeling_manifest_path,
                    "modeling_dataset_id": "fixture-modeling-dataset",
                    "selection_fingerprints": {
                        "split_key_hashes": dict(_SPLIT_KEY_HASHES),
                        "train_population_key_hash": "train-pop-hash-fixture",
                    },
                    "counts": {
                        "split_row_counts_full": dict(_SPLIT_ROW_COUNTS_FULL),
                        "split_row_counts_selected": dict(_SPLIT_ROW_COUNTS_SELECTED),
                        "training_population": dict(_TRAINING_POPULATION),
                    },
                    "baseline_compatibility": {
                        "compatible": True,
                        "reason": "selection_fingerprints_match",
                    },
                },
            )

        experiment_id = f"exp-{trial_key}"
        manifest_path = output_root / "experiments" / experiment_id / "_manifest.json"
        checkpoint_path = (
            output_root / "work" / experiment_id / "checkpoints" / "last.pt"
        )

        if (
            interrupt_first_trial
            and trial_key == "trial-0000"
            and calls[trial_key] == 1
            and not resume
        ):
            return SimpleNamespace(
                completed=False,
                experiment_id=experiment_id,
                manifest_path=manifest_path,
                summary={"status": "interrupted"},
            )

        if session_limit_on_resume and resume and trial_key == "trial-0000":
            if write_checkpoint_on_session_limit:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "history": [
                            {
                                "epoch": 1,
                                "validation_policy_mrr": 0.61,
                            }
                        ]
                    },
                    checkpoint_path,
                )
            return SimpleNamespace(
                completed=False,
                experiment_id=experiment_id,
                manifest_path=manifest_path,
                summary={"status": "session_limit_reached"},
            )

        learning_rate = float(payload["optimization"]["learning_rate"])
        value_weight = float(payload["loss"]["value_weight"])
        metric = float(round(0.7 + value_weight * 0.1 - learning_rate, 6))

        _write_completed_trial_artifacts(manifest_path=manifest_path, metric=metric)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "history": [
                    {
                        "epoch": 1,
                        "validation_policy_mrr": metric,
                    }
                ]
            },
            checkpoint_path,
        )

        return SimpleNamespace(
            completed=True,
            experiment_id=experiment_id,
            manifest_path=manifest_path,
            summary={"status": "complete"},
        )

    return _stub_run_neural, call_log


def test_run_neural_hpo_dry_run_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)
    reference_config_path = tmp_path / "reference_neural.yaml"
    _write_neural_config(
        reference_config_path,
        manifest_path=modeling_manifest,
        output_root=tmp_path / "neural",
    )

    hpo_config_path = tmp_path / "hpo.yaml"
    _write_hpo_config(
        hpo_config_path,
        reference_neural_config=reference_config_path,
        study_output_root=tmp_path / "hpo_output",
    )

    config = load_neural_hpo_config(hpo_config_path)
    stub, _ = _build_neural_stub(interrupt_first_trial=False)
    monkeypatch.setattr(run_neural_hpo_module, "run_neural", stub)

    dry = _run_hpo(
        config=config,
        resume_running_trials=False,
        max_new_trials_override=None,
        dry_run=True,
        status_only=False,
        export_best_config=None,
    )
    assert dry["dry_run"] is True
    assert dry["would_start_new_trials"] == 2

    status = _run_hpo(
        config=config,
        resume_running_trials=False,
        max_new_trials_override=None,
        dry_run=False,
        status_only=True,
        export_best_config=None,
    )
    assert status["started_trials"] == 0
    assert status["remaining_trial_budget"] == 4


def test_run_neural_hpo_resume_and_budget_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)
    reference_config_path = tmp_path / "reference_neural.yaml"
    _write_neural_config(
        reference_config_path,
        manifest_path=modeling_manifest,
        output_root=tmp_path / "neural",
    )

    hpo_config_path = tmp_path / "hpo.yaml"
    _write_hpo_config(
        hpo_config_path,
        reference_neural_config=reference_config_path,
        study_output_root=tmp_path / "hpo_output",
        max_total_trials=3,
        max_new_trials_per_invocation=2,
    )

    config = load_neural_hpo_config(hpo_config_path)
    stub, _ = _build_neural_stub(interrupt_first_trial=True)
    monkeypatch.setattr(run_neural_hpo_module, "run_neural", stub)

    interrupted = _run_hpo(
        config=config,
        resume_running_trials=False,
        max_new_trials_override=1,
        dry_run=False,
        status_only=False,
        export_best_config=None,
    )
    assert interrupted["status"] == "interrupted"
    assert interrupted["trial_state_counts"]["RUNNING"] == 1

    with pytest.raises(RuntimeError, match="running trials"):
        _run_hpo(
            config=config,
            resume_running_trials=False,
            max_new_trials_override=None,
            dry_run=False,
            status_only=False,
            export_best_config=None,
        )

    resumed = _run_hpo(
        config=config,
        resume_running_trials=True,
        max_new_trials_override=None,
        dry_run=False,
        status_only=False,
        export_best_config=None,
    )
    assert resumed["status"] == "ok"
    assert resumed["started_trials"] == 3
    assert resumed["remaining_trial_budget"] == 0
    assert resumed["trial_state_counts"]["RUNNING"] == 0
    assert resumed["trial_state_counts"]["COMPLETE"] == 3


def test_run_neural_hpo_rejects_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)
    reference_config_path = tmp_path / "reference_neural.yaml"
    _write_neural_config(
        reference_config_path,
        manifest_path=modeling_manifest,
        output_root=tmp_path / "neural",
    )

    hpo_config_path = tmp_path / "hpo.yaml"
    _write_hpo_config(
        hpo_config_path,
        reference_neural_config=reference_config_path,
        study_output_root=tmp_path / "hpo_output",
    )

    config = load_neural_hpo_config(hpo_config_path)
    stub, _ = _build_neural_stub(interrupt_first_trial=False)
    monkeypatch.setattr(run_neural_hpo_module, "run_neural", stub)

    _run_hpo(
        config=config,
        resume_running_trials=False,
        max_new_trials_override=None,
        dry_run=True,
        status_only=False,
        export_best_config=None,
    )

    _write_hpo_config(
        hpo_config_path,
        reference_neural_config=reference_config_path,
        study_output_root=tmp_path / "hpo_output",
        learning_rate_high=0.005,
    )
    changed = load_neural_hpo_config(hpo_config_path)

    with pytest.raises(RuntimeError, match="identity mismatch"):
        _run_hpo(
            config=changed,
            resume_running_trials=False,
            max_new_trials_override=None,
            dry_run=True,
            status_only=False,
            export_best_config=None,
        )


def test_run_neural_hpo_single_writer_lock_active_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)
    reference_config_path = tmp_path / "reference_neural.yaml"
    _write_neural_config(
        reference_config_path,
        manifest_path=modeling_manifest,
        output_root=tmp_path / "neural",
    )

    hpo_config_path = tmp_path / "hpo.yaml"
    _write_hpo_config(
        hpo_config_path,
        reference_neural_config=reference_config_path,
        study_output_root=tmp_path / "hpo_output",
    )

    config = load_neural_hpo_config(hpo_config_path)
    stub, _ = _build_neural_stub(interrupt_first_trial=False)
    monkeypatch.setattr(run_neural_hpo_module, "run_neural", stub)

    process = psutil.Process(os.getpid())
    study_root = run_neural_hpo_module._study_root(config)
    lock_path = study_root / ".study.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "lock_id": "foreign-lock",
                "pid": process.pid,
                "pid_create_time_epoch": float(process.create_time()),
                "host": socket.gethostname(),
                "created_at_utc": "2024-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="ownership is 'active'"):
        _run_hpo(
            config=config,
            resume_running_trials=False,
            max_new_trials_override=1,
            dry_run=False,
            status_only=False,
            export_best_config=None,
        )


def test_run_neural_hpo_single_writer_lock_uncertain_owner_conservative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)
    reference_config_path = tmp_path / "reference_neural.yaml"
    _write_neural_config(
        reference_config_path,
        manifest_path=modeling_manifest,
        output_root=tmp_path / "neural",
    )

    hpo_config_path = tmp_path / "hpo.yaml"
    _write_hpo_config(
        hpo_config_path,
        reference_neural_config=reference_config_path,
        study_output_root=tmp_path / "hpo_output",
    )

    config = load_neural_hpo_config(hpo_config_path)
    stub, _ = _build_neural_stub(interrupt_first_trial=False)
    monkeypatch.setattr(run_neural_hpo_module, "run_neural", stub)

    study_root = run_neural_hpo_module._study_root(config)
    lock_path = study_root / ".study.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "created_at_utc": "2024-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="ownership is 'uncertain'"):
        _run_hpo(
            config=config,
            resume_running_trials=False,
            max_new_trials_override=1,
            dry_run=False,
            status_only=False,
            export_best_config=None,
        )


def test_run_neural_hpo_preflight_failure_stops_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)
    reference_config_path = tmp_path / "reference_neural.yaml"
    _write_neural_config(
        reference_config_path,
        manifest_path=modeling_manifest,
        output_root=tmp_path / "neural",
    )

    hpo_config_path = tmp_path / "hpo.yaml"
    _write_hpo_config(
        hpo_config_path,
        reference_neural_config=reference_config_path,
        study_output_root=tmp_path / "hpo_output",
    )
    config = load_neural_hpo_config(hpo_config_path)

    dry_run_calls: list[bool] = []

    def _failing_preflight_stub(*, dry_run: bool = False, **_: Any) -> Any:
        dry_run_calls.append(dry_run)
        if dry_run:
            raise RuntimeError("JSON file not found: missing modeling manifest")
        raise AssertionError("trial execution should not start after dry-run preflight failure")

    monkeypatch.setattr(run_neural_hpo_module, "run_neural", _failing_preflight_stub)

    with pytest.raises(RuntimeError, match="JSON file not found"):
        _run_hpo(
            config=config,
            resume_running_trials=False,
            max_new_trials_override=1,
            dry_run=False,
            status_only=False,
            export_best_config=None,
        )

    assert dry_run_calls == [True]


def test_run_neural_hpo_export_best_config_preserves_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)
    reference_config_path = tmp_path / "reference_neural.yaml"
    _write_neural_config(
        reference_config_path,
        manifest_path=modeling_manifest,
        output_root=tmp_path / "neural",
    )

    hpo_config_path = tmp_path / "hpo.yaml"
    _write_hpo_config(
        hpo_config_path,
        reference_neural_config=reference_config_path,
        study_output_root=tmp_path / "hpo_output",
        max_total_trials=2,
        max_new_trials_per_invocation=1,
    )
    config = load_neural_hpo_config(hpo_config_path)

    stub, call_log = _build_neural_stub(interrupt_first_trial=False)
    monkeypatch.setattr(run_neural_hpo_module, "run_neural", stub)

    first = _run_hpo(
        config=config,
        resume_running_trials=False,
        max_new_trials_override=1,
        dry_run=False,
        status_only=False,
        export_best_config=None,
    )
    assert first["status"] == "ok"

    export_path = tmp_path / "exports" / "best.yaml"
    final_output_root = tmp_path / "final_run_output"
    exported = _run_hpo(
        config=config,
        resume_running_trials=False,
        max_new_trials_override=None,
        dry_run=False,
        status_only=True,
        export_best_config=export_path,
        export_final_output_root=final_output_root,
    )

    assert exported["exported_best_config_path"] == export_path.as_posix()

    exported_payload = yaml.safe_load(export_path.read_text(encoding="utf-8"))
    assert isinstance(exported_payload, dict)
    assert isinstance(exported_payload.get("output"), dict)
    assert str(exported_payload["output"]["output_root"]) == final_output_root.as_posix()

    dry_run_paths = {
        entry["config_path"]
        for entry in call_log
        if entry.get("dry_run") is True
    }
    assert export_path.as_posix() in dry_run_paths


def test_run_neural_hpo_resume_running_trial_without_checkpoint_marks_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)
    reference_config_path = tmp_path / "reference_neural.yaml"
    _write_neural_config(
        reference_config_path,
        manifest_path=modeling_manifest,
        output_root=tmp_path / "neural",
    )

    hpo_config_path = tmp_path / "hpo.yaml"
    _write_hpo_config(
        hpo_config_path,
        reference_neural_config=reference_config_path,
        study_output_root=tmp_path / "hpo_output",
        max_total_trials=1,
        max_new_trials_per_invocation=1,
    )
    config = load_neural_hpo_config(hpo_config_path)

    first_stub, _ = _build_neural_stub(interrupt_first_trial=True)
    monkeypatch.setattr(run_neural_hpo_module, "run_neural", first_stub)
    first = _run_hpo(
        config=config,
        resume_running_trials=False,
        max_new_trials_override=1,
        dry_run=False,
        status_only=False,
        export_best_config=None,
    )
    assert first["status"] == "interrupted"
    assert first["trial_state_counts"]["RUNNING"] == 1

    second_stub, _ = _build_neural_stub(
        interrupt_first_trial=False,
        session_limit_on_resume=True,
        write_checkpoint_on_session_limit=False,
    )
    monkeypatch.setattr(run_neural_hpo_module, "run_neural", second_stub)

    resumed = _run_hpo(
        config=config,
        resume_running_trials=True,
        max_new_trials_override=None,
        dry_run=False,
        status_only=False,
        export_best_config=None,
    )
    assert resumed["trial_state_counts"]["RUNNING"] == 0
    assert resumed["trial_state_counts"]["FAIL"] == 1


def test_run_neural_hpo_resume_running_trial_can_prune(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)
    reference_config_path = tmp_path / "reference_neural.yaml"
    _write_neural_config(
        reference_config_path,
        manifest_path=modeling_manifest,
        output_root=tmp_path / "neural",
    )

    hpo_config_path = tmp_path / "hpo.yaml"
    _write_hpo_config(
        hpo_config_path,
        reference_neural_config=reference_config_path,
        study_output_root=tmp_path / "hpo_output",
        max_total_trials=1,
        max_new_trials_per_invocation=1,
    )
    config = load_neural_hpo_config(hpo_config_path)

    first_stub, _ = _build_neural_stub(interrupt_first_trial=True)
    monkeypatch.setattr(run_neural_hpo_module, "run_neural", first_stub)

    interrupted = _run_hpo(
        config=config,
        resume_running_trials=False,
        max_new_trials_override=1,
        dry_run=False,
        status_only=False,
        export_best_config=None,
    )
    assert interrupted["status"] == "interrupted"

    preflight_stub, _ = _build_neural_stub(interrupt_first_trial=False)
    monkeypatch.setattr(run_neural_hpo_module, "run_neural", preflight_stub)

    def _pruned_run_trial(**_: Any) -> run_neural_hpo_module.TrialExecutionResult:
        return run_neural_hpo_module.TrialExecutionResult(
            status="pruned",
            objective_value=None,
            experiment_id="exp-trial-0000",
            manifest_path=None,
            checkpoint_path=Path("checkpoint.pt"),
            validation_policy_metrics=None,
            validation_value_metrics=None,
            message="Pruned after resume",
        )

    monkeypatch.setattr(run_neural_hpo_module, "_run_trial", _pruned_run_trial)

    resumed = _run_hpo(
        config=config,
        resume_running_trials=True,
        max_new_trials_override=None,
        dry_run=False,
        status_only=False,
        export_best_config=None,
    )
    assert resumed["trial_state_counts"]["RUNNING"] == 0
    assert resumed["trial_state_counts"]["PRUNED"] == 1


def test_run_neural_hpo_rejects_corrupt_sampler_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)
    reference_config_path = tmp_path / "reference_neural.yaml"
    _write_neural_config(
        reference_config_path,
        manifest_path=modeling_manifest,
        output_root=tmp_path / "neural",
    )

    hpo_config_path = tmp_path / "hpo.yaml"
    _write_hpo_config(
        hpo_config_path,
        reference_neural_config=reference_config_path,
        study_output_root=tmp_path / "hpo_output",
    )
    config = load_neural_hpo_config(hpo_config_path)

    stub, _ = _build_neural_stub(interrupt_first_trial=False)
    monkeypatch.setattr(run_neural_hpo_module, "run_neural", stub)

    study_root = run_neural_hpo_module._study_root(config)
    study_root.mkdir(parents=True, exist_ok=True)
    (study_root / "sampler_state.pkl").write_bytes(b"not-a-valid-pickle")

    with pytest.raises(RuntimeError, match="Sampler state is unreadable/corrupt"):
        _run_hpo(
            config=config,
            resume_running_trials=False,
            max_new_trials_override=None,
            dry_run=True,
            status_only=False,
            export_best_config=None,
        )


def test_run_neural_hpo_sampler_reproducible_after_interrupt_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modeling_manifest = _write_modeling_fixture(tmp_path)

    reference_a = tmp_path / "reference_a.yaml"
    reference_b = tmp_path / "reference_b.yaml"
    _write_neural_config(
        reference_a,
        manifest_path=modeling_manifest,
        output_root=tmp_path / "neural_a",
    )
    _write_neural_config(
        reference_b,
        manifest_path=modeling_manifest,
        output_root=tmp_path / "neural_b",
    )

    hpo_a = tmp_path / "hpo_a.yaml"
    hpo_b = tmp_path / "hpo_b.yaml"
    _write_hpo_config(
        hpo_a,
        reference_neural_config=reference_a,
        study_output_root=tmp_path / "hpo_output_a",
        max_total_trials=2,
        max_new_trials_per_invocation=2,
    )
    _write_hpo_config(
        hpo_b,
        reference_neural_config=reference_b,
        study_output_root=tmp_path / "hpo_output_b",
        max_total_trials=2,
        max_new_trials_per_invocation=2,
    )

    config_a = load_neural_hpo_config(hpo_a)
    config_b = load_neural_hpo_config(hpo_b)

    interrupted_stub, _ = _build_neural_stub(interrupt_first_trial=True)
    monkeypatch.setattr(run_neural_hpo_module, "run_neural", interrupted_stub)

    first = _run_hpo(
        config=config_a,
        resume_running_trials=False,
        max_new_trials_override=1,
        dry_run=False,
        status_only=False,
        export_best_config=None,
    )
    assert first["status"] == "interrupted"

    resumed_stub, _ = _build_neural_stub(interrupt_first_trial=False)
    monkeypatch.setattr(run_neural_hpo_module, "run_neural", resumed_stub)
    second = _run_hpo(
        config=config_a,
        resume_running_trials=True,
        max_new_trials_override=1,
        dry_run=False,
        status_only=False,
        export_best_config=None,
    )
    assert second["status"] == "ok"
    assert second["trial_state_counts"]["COMPLETE"] == 2

    uninterrupted_stub, _ = _build_neural_stub(interrupt_first_trial=False)
    monkeypatch.setattr(run_neural_hpo_module, "run_neural", uninterrupted_stub)
    direct = _run_hpo(
        config=config_b,
        resume_running_trials=False,
        max_new_trials_override=2,
        dry_run=False,
        status_only=False,
        export_best_config=None,
    )
    assert direct["status"] == "ok"
    assert direct["trial_state_counts"]["COMPLETE"] == 2

    params_resumed = _trial_params_by_number(run_neural_hpo_module._study_root(config_a), 1)
    params_direct = _trial_params_by_number(run_neural_hpo_module._study_root(config_b), 1)
    assert params_resumed == params_direct
