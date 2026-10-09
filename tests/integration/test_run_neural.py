from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from chesslens.modeling.neural_training import run_neural
from chesslens.modeling.run_baselines import run_baselines
from tests.integration.test_run_baselines import (
    _write_baseline_config,
    _write_modeling_fixture,
)


def _write_neural_config(
    path: Path,
    *,
    manifest_path: Path,
    output_root: Path,
    baseline_reference_manifest_path: Path | None = None,
    max_epochs: int = 3,
) -> None:
    baseline_line = (
        "  baseline_reference_manifest_path: null"
        if baseline_reference_manifest_path is None
        else (
            "  baseline_reference_manifest_path: "
            + baseline_reference_manifest_path.as_posix()
        )
    )

    path.write_text(
        "\n".join(
            [
                "input:",
                f"  modeling_manifest_path: {manifest_path.as_posix()}",
                baseline_line,
                "versions:",
                "  neural_pipeline_version: neural_supervised_v1",
                "  feature_schema_version: policy_value_features_v1",
                "  split_definition_version: temporal_game_split_v1",
                "  board_encoding_version: board18_abs_v1",
                "  action_encoding_version: action8x8x73_v1",
                "  position_normalization_version: fen4_legal_ep_v1",
                "runtime:",
                "  seed: 7",
                "  threads: 1",
                "  device: cpu",
                "selection:",
                "  max_train_positions: 100",
                "  max_validation_positions: 100",
                "  max_test_positions: 100",
                "  training_population_mode: temporal_all",
                "model:",
                "  trunk_channels: 16",
                "  residual_blocks: 1",
                "  context_embedding_dim: 8",
                "  context_hidden_dim: 16",
                "  value_hidden_dim: 32",
                "optimization:",
                "  optimizer: adamw",
                "  learning_rate: 0.003",
                "  weight_decay: 0.0001",
                "  batch_size: 2",
                "  grad_clip_norm: 1.0",
                "training:",
                f"  max_epochs: {max_epochs}",
                "  early_stopping_patience: 10",
                "  selection_metric: validation_policy_mrr",
                "loss:",
                "  policy_weight: 1.0",
                "  value_weight: 0.4",
                "evaluation:",
                "  ranking_cutoffs:",
                "    - 1",
                "    - 3",
                "    - 5",
                "  ece_bins: 5",
                "output:",
                f"  output_root: {output_root.as_posix()}",
                "  checkpoint_every_epochs: 1",
                "  keep_top_k_checkpoints: 2",
                "  progress_update_seconds: 1",
                "behavior:",
                "  strict: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_neural_complete_reuse_and_status(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)
    output_root = tmp_path / "neural_output"
    config_path = tmp_path / "neural.yaml"
    _write_neural_config(config_path, manifest_path=manifest_path, output_root=output_root)

    first = run_neural(config_path=config_path)
    assert first.reused_existing is False
    assert first.completed is True
    assert first.output_path.exists()
    assert first.manifest_path.exists()

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["experiment_id"] == first.experiment_id

    second = run_neural(config_path=config_path)
    assert second.reused_existing is True
    assert second.completed is True

    status = run_neural(config_path=config_path, status_only=True)
    assert status.completed is True
    assert status.summary is not None
    assert status.summary["status"] == "complete"


def test_run_neural_resume_after_session_limit(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)
    output_root = tmp_path / "neural_output"
    config_path = tmp_path / "neural.yaml"
    _write_neural_config(
        config_path,
        manifest_path=manifest_path,
        output_root=output_root,
        max_epochs=3,
    )

    first = run_neural(config_path=config_path, max_epochs_this_session=1)
    assert first.completed is False
    assert first.summary is not None
    assert first.summary["status"] == "session_limit_reached"

    resumed = run_neural(config_path=config_path, resume=True)
    assert resumed.completed is True
    assert resumed.reused_existing is False


def test_run_neural_resume_rejects_identity_hash_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)
    output_root = tmp_path / "neural_output"
    config_path = tmp_path / "neural.yaml"
    _write_neural_config(
        config_path,
        manifest_path=manifest_path,
        output_root=output_root,
        max_epochs=3,
    )

    first = run_neural(config_path=config_path, max_epochs_this_session=1)
    assert first.completed is False

    checkpoint_path = (
        output_root
        / "work"
        / first.experiment_id
        / "checkpoints"
        / "last.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["identity_hash"] = "tampered"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="identity hash mismatch"):
        run_neural(config_path=config_path, resume=True)


def test_run_neural_detects_tampered_existing_artifact(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)
    output_root = tmp_path / "neural_output"
    config_path = tmp_path / "neural.yaml"
    _write_neural_config(config_path, manifest_path=manifest_path, output_root=output_root)

    first = run_neural(config_path=config_path)
    assert first.completed is True

    history_path = first.output_path / "training_history.json"
    history = history_path.read_text(encoding="utf-8")
    history_path.write_text(history + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact integrity mismatch"):
        run_neural(config_path=config_path)


def test_run_neural_baseline_compatibility_compare(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)

    baseline_output_root = tmp_path / "baselines_output"
    baseline_config_path = tmp_path / "baseline.yaml"
    _write_baseline_config(
        baseline_config_path,
        manifest_path=manifest_path,
        output_root=baseline_output_root,
    )
    baseline_result = run_baselines(config_path=baseline_config_path)

    neural_output_root = tmp_path / "neural_output"
    neural_config_path = tmp_path / "neural.yaml"
    _write_neural_config(
        neural_config_path,
        manifest_path=manifest_path,
        output_root=neural_output_root,
        baseline_reference_manifest_path=baseline_result.manifest_path,
    )

    result = run_neural(config_path=neural_config_path, validate_only=True)
    assert result.summary is not None
    compatibility = result.summary["baseline_compatibility"]
    assert isinstance(compatibility, dict)
    assert compatibility["compatible"] is True
