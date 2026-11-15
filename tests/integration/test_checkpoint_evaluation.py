from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

import chesslens.modeling.run_checkpoint_evaluation as evaluation
from chesslens.modeling.neural_config import load_neural_config
from chesslens.modeling.neural_data import selected_key_hash
from chesslens.modeling.neural_training import run_neural
from chesslens.modeling.run_baselines import _load_position_examples
from tests.integration.test_run_baselines import _write_modeling_fixture, _write_parquet
from tests.integration.test_run_neural import _write_neural_config


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    modeling = _write_modeling_fixture(tmp_path)
    validation = modeling.parent / "policy_examples" / "validation.parquet"
    original = pq.read_table(validation).to_pylist()[0]
    rows = []
    games: list[dict[str, object]] = []
    for idx in range(12):
        for ply in (3, 5):
            rows.append(
                {
                    **original,
                    "game_id": f"validation_{idx:02d}",
                    "ply": ply,
                    "value_target_wdl": ("win", "draw", "loss")[idx % 3],
                }
            )
        games.append(
            {
                "game_id": f"validation_{idx:02d}",
                "temporal_split": "validation",
                "white_player_hash": f"validation_white_{idx}",
                "black_player_hash": f"validation_black_{idx}",
            }
        )
    _write_parquet(validation, rows)
    _write_parquet(modeling.parent / "game_assignments" / "validation.parquet", games)
    config = tmp_path / "neural.yaml"
    _write_neural_config(
        config,
        manifest_path=modeling,
        output_root=tmp_path / "neural",
        max_epochs=1,
    )
    frozen = run_neural(config_path=config, evaluation_scope="validation_only")
    return frozen.manifest_path, modeling, config


@pytest.fixture(scope="module")
def frozen(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    return _fixture(tmp_path_factory.mktemp("frozen_checkpoint"))


def test_game_split_stable_and_disjoint(frozen: tuple[Path, Path, Path]) -> None:
    _, modeling, config_path = frozen
    config = load_neural_config(config_path)
    rows = _load_position_examples(
        [modeling.parent / "policy_examples" / "validation.parquet"],
        split="validation",
        limit=100,
        seed=config.runtime.seed,
        novel_position_ids=set(),
    )
    fit, check = evaluation.split_validation_games(rows, seed=7, max_positions=24)
    fit_again, check_again = evaluation.split_validation_games(
        list(reversed(rows)),
        seed=7,
        max_positions=24,
    )
    assert selected_key_hash(fit) == selected_key_hash(fit_again)
    assert selected_key_hash(check) == selected_key_hash(check_again)
    assert not {r.game_id for r in fit} & {r.game_id for r in check}
    assert len(fit) + len(check) == len(rows)
    with pytest.raises(RuntimeError, match="two distinct games"):
        evaluation.split_validation_games(rows[:1], seed=7, max_positions=24)


def test_only_fit_labels_and_validation_loader(
    frozen: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, modeling, _ = frozen
    fitter = evaluation.fit_temperature
    loader = evaluation._load_position_examples
    seen: dict[str, Any] = {}

    def spy_fit(logits: torch.Tensor, targets: torch.Tensor) -> float:
        seen["logits"] = logits.clone()
        seen["targets"] = targets.clone()
        assert logits.device.type == "cpu"
        assert logits.dtype == torch.float64
        return fitter(logits, targets)

    def spy_loader(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["split"] == "validation"
        return loader(*args, **kwargs)

    monkeypatch.setattr(evaluation, "fit_temperature", spy_fit)
    monkeypatch.setattr(evaluation, "_load_position_examples", spy_loader)
    report = evaluation.run_checkpoint_evaluation(
        neural_manifest=manifest,
        modeling_manifest=modeling,
        output=tmp_path / "pilot",
        selection_rationale="Frozen fixture checkpoint; no test selection",
        seed=7,
        batch_size=3,
    )
    with np.load(tmp_path / "pilot" / "predictions.npz", allow_pickle=False) as predictions:
        mask = predictions["partition"] == "fit"
        np.testing.assert_array_equal(seen["targets"].numpy(), predictions["targets"][mask])
        np.testing.assert_array_equal(seen["logits"].numpy(), predictions["logits"][mask])
        assert int(mask.sum()) == report["fit"]["positions"]
        fit_games = set(predictions["game_id"][mask])
        check_games = set(predictions["game_id"][~mask])
        assert not fit_games & check_games
    assert report["argmax_difference_count"] == 0
    assert report["lineage"]["test_label_access"] is False
    assert report["raw"]["accuracy"] == report["calibrated"]["accuracy"]
    assert set(report["raw"]["reliability_bins"]) == {"win", "draw", "loss"}
    assert (tmp_path / "pilot" / "_SUCCESS").is_file()


def test_frozen_checkpoint_cli_repeat_and_output_integrity(
    frozen: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    manifest, modeling, _ = frozen
    output = tmp_path / "cli"
    command = [
        sys.executable,
        "-m",
        "chesslens.modeling.run_checkpoint_evaluation",
        "--neural-manifest",
        str(manifest),
        "--modeling-manifest",
        str(modeling),
        "--output",
        str(output),
        "--selection-rationale",
        "Preselected frozen fixture",
        "--seed",
        "7",
        "--batch-size",
        "3",
        "--threads",
        "1",
        "--json",
    ]
    env = {**os.environ, "PYTHONPATH": str(Path(evaluation.__file__).parents[2])}
    for _ in range(2):
        completed = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
        report = json.loads(completed.stdout)
        assert report["mode"] == "exploratory_validation_pilot"
        assert all(
            metric in report["raw"]
            for metric in (
                "nll",
                "multiclass_brier_score",
                "ece",
                "accuracy",
                "reliability_bins",
            )
        )
        assert report == json.loads((output / "report.json").read_text(encoding="utf-8"))
    (output / "report.md").write_text("tampered", encoding="utf-8")
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert completed.returncode == 1
    assert "integrity mismatch" in completed.stderr


def test_rejects_mismatched_data_checkpoint_and_class_order(
    frozen: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _, _ = frozen
    wrong_modeling = _write_modeling_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="selection fingerprints mismatch"):
        evaluation.run_checkpoint_evaluation(
            neural_manifest=manifest,
            modeling_manifest=wrong_modeling,
            output=tmp_path / "bad",
            selection_rationale="Fixture",
        )
    wrong_payload = json.loads(wrong_modeling.read_text(encoding="utf-8"))
    wrong_payload["upstream"]["collection_id"] = "different-collection"
    wrong_modeling.write_text(json.dumps(wrong_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="modeling manifest SHA256 mismatch"):
        evaluation.run_checkpoint_evaluation(
            neural_manifest=manifest,
            modeling_manifest=wrong_modeling,
            output=tmp_path / "bad-manifest",
            selection_rationale="Fixture",
        )
    wrong_payload["modeling_dataset_id"] = "different-dataset"
    wrong_modeling.write_text(json.dumps(wrong_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="dataset ID mismatch"):
        evaluation.run_checkpoint_evaluation(
            neural_manifest=manifest,
            modeling_manifest=wrong_modeling,
            output=tmp_path / "bad-dataset",
            selection_rationale="Fixture",
        )
    original_verify = evaluation._verify_selection

    def mismatched_keys(*args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs = dict(kwargs)
        expected = json.loads(json.dumps(args[2]))
        expected["split_key_hashes"]["validation"] = "wrong"
        return original_verify(args[0], args[1], expected)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "_verify_selection", mismatched_keys)
        with pytest.raises(RuntimeError, match="selection fingerprints mismatch"):
            evaluation.run_checkpoint_evaluation(
                neural_manifest=manifest,
                output=tmp_path / "bad-keys",
                selection_rationale="Fixture",
            )
    with monkeypatch.context() as patch:
        original_context = evaluation._validation_context

        def wrong_order(root: Path) -> dict[str, Any]:
            context = original_context(root)
            context["value"]["class_order"] = ["loss", "draw", "win"]
            return context

        patch.setattr(evaluation, "_validation_context", wrong_order)
        with pytest.raises(RuntimeError, match="WDL class order mismatch"):
            evaluation.run_checkpoint_evaluation(
                neural_manifest=manifest,
                output=tmp_path / "bad-order",
                selection_rationale="Fixture",
            )

    copied = tmp_path / "copied"
    shutil.copytree(manifest.parent, copied)
    checkpoint = copied / "model_state_best.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="artifact integrity mismatch"):
        evaluation.run_checkpoint_evaluation(
            neural_manifest=copied / "_manifest.json",
            output=tmp_path / "bad-checkpoint",
            selection_rationale="Fixture",
        )


def test_rejects_wrong_state_shapes_and_missing_completion(
    frozen: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _, _ = frozen
    actual_load = torch.load

    def wrong_shape(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["weights_only"] is True
        state = actual_load(*args, **kwargs)
        state["value_head.2.weight"] = torch.zeros((4, 32))
        return state

    monkeypatch.setattr(evaluation.torch, "load", wrong_shape)
    with pytest.raises(RuntimeError, match="size mismatch"):
        evaluation.run_checkpoint_evaluation(
            neural_manifest=manifest,
            output=tmp_path / "bad-shape",
            selection_rationale="Fixture",
        )
    copied = tmp_path / "incomplete"
    shutil.copytree(manifest.parent, copied)
    (copied / "_SUCCESS").unlink()
    with pytest.raises(RuntimeError, match="missing _SUCCESS"):
        evaluation._validate_neural_manifest(copied / "_manifest.json")

    (copied / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    payload = json.loads((copied / "_manifest.json").read_text(encoding="utf-8"))
    del payload["versions"]["board_encoding_version"]
    (copied / "_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="version board_encoding_version mismatch"):
        evaluation.run_checkpoint_evaluation(
            neural_manifest=copied / "_manifest.json",
            output=tmp_path / "bad-version",
            selection_rationale="Fixture",
        )


def test_scaling_does_not_change_frozen_model(
    frozen: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    manifest, _, _ = frozen
    files = ("model_state_best.pt", "feature_preprocessing.json")
    before = {name: evaluation._file_sha256(manifest.parent / name) for name in files}
    evaluation.run_checkpoint_evaluation(
        neural_manifest=manifest,
        output=tmp_path / "unchanged",
        selection_rationale="Fixture",
    )
    assert before == {name: evaluation._file_sha256(manifest.parent / name) for name in files}


def test_hpo_selection_evidence_is_pretest_and_rejects_nonwinner(
    frozen: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    manifest, _, _ = frozen
    experiment_id = json.loads(manifest.read_text(encoding="utf-8"))["experiment_id"]
    snapshot = tmp_path / "trials.json"
    snapshot.write_text(
        json.dumps(
            {
                "trials": [
                    {
                        "number": 0,
                        "state": "COMPLETE",
                        "value": 0.4,
                        "user_attrs": {"neural_experiment_id": experiment_id},
                    },
                    {"number": 1, "state": "RUNNING", "value": None, "user_attrs": {}},
                ]
            }
        ),
        encoding="utf-8",
    )
    evidence = evaluation._selection_evidence(snapshot, experiment_id)
    assert evidence["complete_trials"] == 1
    with pytest.raises(RuntimeError, match="maximum validation policy MRR candidate mismatch"):
        evaluation._selection_evidence(snapshot, "nonwinner")
