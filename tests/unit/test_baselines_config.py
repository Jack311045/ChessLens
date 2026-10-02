from __future__ import annotations

from pathlib import Path

import pytest

from chesslens.modeling.baselines_config import (
    apply_baseline_cli_overrides,
    load_baseline_config,
)


def _write_config(path: Path, *, ece_bins: int = 15) -> None:
    path.write_text(
        "\n".join(
            [
                "input:",
                "  modeling_manifest_path: data/modeling/datasets/example/_manifest.json",
                "versions:",
                "  baseline_pipeline_version: classical_baselines_v1",
                "  feature_schema_version: policy_value_features_v1",
                "  split_definition_version: temporal_game_split_v1",
                "  rating_band_definition_version: rating_band_v1",
                "  game_phase_definition_version: game_phase_v1",
                "runtime:",
                "  seed: 20240131",
                "  threads: 4",
                "  early_stopping_rounds: 50",
                "limits:",
                "  max_train_positions: 5000",
                "  max_validation_positions: 2000",
                "  max_test_positions: 2000",
                "  max_negative_candidates_per_train_position: 128",
                "  evaluate_all_legal_candidates_validation: true",
                "  evaluate_all_legal_candidates_test: true",
                "training_population:",
                "  mode: temporal_all",
                "preflight:",
                "  legality_scope: selected",
                "frequency:",
                "  backoff_levels:",
                "    - rating_band_phase",
                "    - phase",
                "    - global",
                "lightgbm:",
                "  n_estimators: 800",
                "  learning_rate: 0.05",
                "  num_leaves: 63",
                "  min_data_in_leaf: 50",
                "  feature_fraction: 0.9",
                "  lambda_l2: 1.0",
                "logistic:",
                "  solver: lbfgs",
                "  c: 1.0",
                "  max_iter: 1000",
                "evaluation:",
                "  ranking_cutoffs:",
                "    - 1",
                "    - 3",
                "    - 5",
                f"  ece_bins: {ece_bins}",
                "  bootstrap_iterations: 200",
                "output:",
                "  output_root: data/baselines",
                "  prediction_artifact_row_limit: 50000",
                "behavior:",
                "  strict: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_baseline_config_success(tmp_path: Path) -> None:
    config_path = tmp_path / "baseline.yaml"
    _write_config(config_path)

    config = load_baseline_config(config_path)

    assert config.runtime.seed == 20240131
    assert config.runtime.threads == 4
    assert config.limits.max_train_positions == 5000
    assert config.evaluation.ranking_cutoffs == (1, 3, 5)
    assert config.output.prediction_artifact_row_limit == 50000
    assert config.training_population.mode == "temporal_all"
    assert config.preflight.legality_scope == "selected"


def test_apply_baseline_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "baseline.yaml"
    _write_config(config_path)
    config = load_baseline_config(config_path)

    overridden = apply_baseline_cli_overrides(
        config,
        modeling_manifest_path="data/modeling/datasets/override/_manifest.json",
        output_root="data/other_baselines",
        max_train_positions=111,
        max_validation_positions=222,
        max_test_positions=333,
        threads=8,
        seed=999,
    )

    assert overridden.input.modeling_manifest_path.as_posix().endswith(
        "data/modeling/datasets/override/_manifest.json"
    )
    assert overridden.output.output_root.name == "other_baselines"
    assert overridden.limits.max_train_positions == 111
    assert overridden.limits.max_validation_positions == 222
    assert overridden.limits.max_test_positions == 333
    assert overridden.runtime.threads == 8
    assert overridden.runtime.seed == 999


def test_load_baseline_config_rejects_invalid_ece_bins(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-baseline.yaml"
    _write_config(config_path, ece_bins=1)

    with pytest.raises(ValueError, match="ece_bins"):
        load_baseline_config(config_path)


def test_load_baseline_config_rejects_invalid_training_population_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-training-mode.yaml"
    _write_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace("mode: temporal_all", "mode: invalid_mode"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training_population.mode"):
        load_baseline_config(config_path)


def test_load_baseline_config_rejects_invalid_legality_scope(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-legality-scope.yaml"
    _write_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace("legality_scope: selected", "legality_scope: invalid_scope"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="preflight.legality_scope"):
        load_baseline_config(config_path)


def test_load_baseline_config_rejects_duplicate_frequency_levels(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-frequency-duplicate.yaml"
    _write_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace("    - phase\n", "    - phase\n    - phase\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_baseline_config(config_path)


def test_load_baseline_config_rejects_unsupported_frequency_level(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-frequency-unsupported.yaml"
    _write_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace("    - phase", "    - unsupported_level"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported"):
        load_baseline_config(config_path)


def test_load_baseline_config_rejects_non_final_global_level(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-frequency-global-order.yaml"
    _write_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace(
            "    - rating_band_phase\n    - phase\n    - global",
            "    - global\n    - rating_band_phase\n    - phase",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must end with 'global'"):
        load_baseline_config(config_path)
