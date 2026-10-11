from __future__ import annotations

from pathlib import Path

import pytest

from chesslens.modeling.optuna_study_config import load_neural_hpo_config


def _write_hpo_config(
    path: Path,
    *,
    threads: int = 1,
    trial_concurrency: int = 1,
    objective_metric: str = "validation.policy.mrr",
) -> None:
    path.write_text(
        "\n".join(
            [
                "reference:",
                "  neural_config_path: configs/neural/fixture_smoke.yaml",
                "study:",
                "  name: fixture_hpo",
                "  output_root: data/neural_hpo",
                "  evaluation_scope: validation_only",
                f"  objective_metric: {objective_metric}",
                "budget:",
                "  max_total_trials: 8",
                "  max_new_trials_per_invocation: 2",
                "search_space:",
                "  learning_rate:",
                "    low: 0.0003",
                "    high: 0.003",
                "    log: true",
                "  weight_decay:",
                "    low: 0.00001",
                "    high: 0.001",
                "    log: true",
                "  value_weight_choices:",
                "    - 0.2",
                "    - 0.4",
                "    - 0.6",
                "runtime:",
                "  seed: 20240131",
                f"  threads: {threads}",
                "  device: cpu",
                "pruning:",
                "  startup_trials: 1",
                "  warmup_epochs: 1",
                "behavior:",
                "  strict: true",
                "execution:",
                f"  trial_concurrency: {trial_concurrency}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_neural_hpo_config_success(tmp_path: Path) -> None:
    config_path = tmp_path / "hpo.yaml"
    _write_hpo_config(config_path)

    config = load_neural_hpo_config(config_path)

    assert config.study.name == "fixture_hpo"
    assert config.study.evaluation_scope == "validation_only"
    assert config.budget.max_total_trials == 8
    assert config.runtime.threads == 1
    assert config.execution.trial_concurrency == 1


def test_load_neural_hpo_config_allows_runtime_threads_above_one(tmp_path: Path) -> None:
    config_path = tmp_path / "hpo.yaml"
    _write_hpo_config(config_path, threads=8)

    config = load_neural_hpo_config(config_path)

    assert config.runtime.threads == 8


def test_load_neural_hpo_config_rejects_parallel_trial_concurrency(tmp_path: Path) -> None:
    config_path = tmp_path / "hpo.yaml"
    _write_hpo_config(config_path, trial_concurrency=2)

    with pytest.raises(ValueError, match="execution.trial_concurrency must be 1"):
        load_neural_hpo_config(config_path)


def test_load_neural_hpo_config_rejects_unsupported_objective_metric(tmp_path: Path) -> None:
    config_path = tmp_path / "hpo.yaml"
    _write_hpo_config(config_path, objective_metric="validation.value.accuracy")

    with pytest.raises(ValueError, match="study.objective_metric must be 'validation.policy.mrr'"):
        load_neural_hpo_config(config_path)
