from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import chesslens.modeling.run_neural_hpo as run_neural_hpo_module
from chesslens.modeling.optuna_study_config import load_neural_hpo_config
from tests.integration.test_run_baselines import _write_modeling_fixture
from tests.integration.test_run_neural import _write_neural_config


def _write_hpo_config(
    path: Path,
    *,
    reference_neural_config: Path,
    study_output_root: Path,
    max_total_trials: int = 4,
    max_new_trials_per_invocation: int = 2,
    learning_rate_high: float = 0.002,
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
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _build_neural_stub(*, interrupt_first_trial: bool) -> Any:
    calls: dict[str, int] = {}

    def _stub_run_neural(*, config_path: Path, resume: bool = False, **_: Any) -> Any:
        trial_key = config_path.parent.name
        calls[trial_key] = calls.get(trial_key, 0) + 1

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        output_root = Path(str(payload["output"]["output_root"]))
        experiment_id = f"exp-{trial_key}"
        manifest_path = output_root / "experiments" / experiment_id / "_manifest.json"

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

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}\n", encoding="utf-8")

        learning_rate = float(payload["optimization"]["learning_rate"])
        value_weight = float(payload["loss"]["value_weight"])
        metric = float(round(0.7 + value_weight * 0.1 - learning_rate, 6))

        (manifest_path.parent / "aggregate_metrics.json").write_text(
            json.dumps({"policy": {"best_validation_policy_mrr": metric}}, indent=2)
            + "\n",
            encoding="utf-8",
        )

        return SimpleNamespace(
            completed=True,
            experiment_id=experiment_id,
            manifest_path=manifest_path,
            summary={"status": "complete"},
        )

    return _stub_run_neural


def test_run_neural_hpo_dry_run_and_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(
        run_neural_hpo_module,
        "run_neural",
        _build_neural_stub(interrupt_first_trial=False),
    )

    dry = run_neural_hpo_module._run_hpo(
        config=config,
        resume_running_trials=False,
        max_new_trials_override=None,
        dry_run=True,
        status_only=False,
        export_best_config=None,
    )
    assert dry["dry_run"] is True
    assert dry["would_start_new_trials"] == 2

    status = run_neural_hpo_module._run_hpo(
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
    monkeypatch.setattr(
        run_neural_hpo_module,
        "run_neural",
        _build_neural_stub(interrupt_first_trial=True),
    )

    interrupted = run_neural_hpo_module._run_hpo(
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
        run_neural_hpo_module._run_hpo(
            config=config,
            resume_running_trials=False,
            max_new_trials_override=None,
            dry_run=False,
            status_only=False,
            export_best_config=None,
        )

    resumed = run_neural_hpo_module._run_hpo(
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
    monkeypatch.setattr(
        run_neural_hpo_module,
        "run_neural",
        _build_neural_stub(interrupt_first_trial=False),
    )

    run_neural_hpo_module._run_hpo(
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
        learning_rate_high=0.003,
    )
    changed = load_neural_hpo_config(hpo_config_path)

    with pytest.raises(RuntimeError, match="identity mismatch"):
        run_neural_hpo_module._run_hpo(
            config=changed,
            resume_running_trials=False,
            max_new_trials_override=None,
            dry_run=True,
            status_only=False,
            export_best_config=None,
        )


def test_run_neural_hpo_single_writer_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(
        run_neural_hpo_module,
        "run_neural",
        _build_neural_stub(interrupt_first_trial=False),
    )

    study_root = run_neural_hpo_module._study_root(config)
    lock_path = study_root / ".study.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "created_at_utc": "2024-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Another HPO writer"):
        run_neural_hpo_module._run_hpo(
            config=config,
            resume_running_trials=False,
            max_new_trials_override=1,
            dry_run=False,
            status_only=False,
            export_best_config=None,
        )
