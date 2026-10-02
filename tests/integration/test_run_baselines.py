from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from chesslens.features.action_encoding import encode_move_to_index
from chesslens.features.position_encoding import normalize_fen, position_id_from_fen
from chesslens.modeling.labels import rating_band, time_control_category
from chesslens.modeling.run_baselines import run_baselines


def _parse_single_json_document(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    assert stripped, "stdout is unexpectedly empty"

    decoder = json.JSONDecoder()
    payload, end_index = decoder.raw_decode(stripped)
    assert stripped[end_index:].strip() == "", "stdout contains trailing non-JSON content"
    assert isinstance(payload, dict), "CLI JSON root is not an object"
    return payload


def _fen_after(moves_uci: list[str]) -> str:
    board = chess.Board()
    for move_uci in moves_uci:
        move = chess.Move.from_uci(move_uci)
        assert move in board.legal_moves
        board.push(move)
    return board.fen(en_passant="legal")


def _policy_row(
    *,
    split: str,
    game_id: str,
    source_month: str,
    fen: str,
    target_move_uci: str,
    value_target_wdl: str,
    mover_rating: int,
    opponent_rating: int,
    time_control_raw: str,
    eco: str,
    opening: str,
    is_player_holdout_game: bool,
) -> dict[str, object]:
    board = chess.Board(fen)
    target_move = chess.Move.from_uci(target_move_uci)
    assert target_move in board.legal_moves

    side_to_move = "w" if board.turn == chess.WHITE else "b"
    tc_category = time_control_category(time_control_raw)
    mover_band = rating_band(mover_rating)

    return {
        "game_id": game_id,
        "ply": 0,
        "temporal_split": split,
        "source_month": source_month,
        "position_id": position_id_from_fen(fen),
        "pre_move_fen": fen,
        "normalized_pre_move_fen": normalize_fen(fen),
        "side_to_move": side_to_move,
        "time_control_raw": time_control_raw,
        "time_control_category": tc_category,
        "eco": eco,
        "opening": opening,
        "mover_rating": mover_rating,
        "opponent_rating": opponent_rating,
        "rating_difference": mover_rating - opponent_rating,
        "mover_rating_band": mover_band,
        "player_disjoint_training_eligible": True,
        "is_player_holdout_game": is_player_holdout_game,
        "policy_target_action_index": encode_move_to_index(target_move, board),
        "value_target_wdl": value_target_wdl,
    }


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_modeling_fixture(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "modeling_dataset"

    train_rows = [
        _policy_row(
            split="train",
            game_id="train_game_a",
            source_month="2013-01",
            fen=_fen_after(["e2e4", "e7e5"]),
            target_move_uci="g1f3",
            value_target_wdl="win",
            mover_rating=1500,
            opponent_rating=1450,
            time_control_raw="300+0",
            eco="C20",
            opening="King Pawn Game",
            is_player_holdout_game=False,
        ),
        _policy_row(
            split="train",
            game_id="train_game_b",
            source_month="2013-01",
            fen=_fen_after(["d2d4", "d7d5"]),
            target_move_uci="c1f4",
            value_target_wdl="draw",
            mover_rating=1650,
            opponent_rating=1675,
            time_control_raw="180+2",
            eco="D00",
            opening="Queen Pawn Game",
            is_player_holdout_game=False,
        ),
        _policy_row(
            split="train",
            game_id="train_game_c",
            source_month="2013-01",
            fen=_fen_after(["c2c4", "e7e6"]),
            target_move_uci="b1c3",
            value_target_wdl="loss",
            mover_rating=1300,
            opponent_rating=1350,
            time_control_raw="600+0",
            eco="A10",
            opening="English Opening",
            is_player_holdout_game=True,
        ),
    ]

    validation_rows = [
        _policy_row(
            split="validation",
            game_id="validation_game_a",
            source_month="2013-01",
            fen=_fen_after(["g1f3", "d7d5", "g2g3"]),
            target_move_uci="c8g4",
            value_target_wdl="loss",
            mover_rating=1800,
            opponent_rating=1775,
            time_control_raw="60+0",
            eco="A04",
            opening="Reti Opening",
            is_player_holdout_game=True,
        )
    ]

    test_rows = [
        _policy_row(
            split="test",
            game_id="test_game_a",
            source_month="2013-01",
            fen=_fen_after(["e2e4", "c7c5", "g1f3"]),
            target_move_uci="d7d6",
            value_target_wdl="draw",
            mover_rating=1900,
            opponent_rating=1850,
            time_control_raw="900+10",
            eco="B20",
            opening="Sicilian Defense",
            is_player_holdout_game=True,
        )
    ]

    novel_rows = [{"position_id": test_rows[0]["position_id"]}]

    train_policy_path = dataset_root / "policy_examples" / "train.parquet"
    validation_policy_path = dataset_root / "policy_examples" / "validation.parquet"
    test_policy_path = dataset_root / "policy_examples" / "test.parquet"
    novel_policy_path = dataset_root / "policy_examples" / "novel_position_test.parquet"

    _write_parquet(train_policy_path, train_rows)
    _write_parquet(validation_policy_path, validation_rows)
    _write_parquet(test_policy_path, test_rows)
    _write_parquet(novel_policy_path, novel_rows)

    train_games: list[dict[str, object]] = [
        {
            "game_id": "train_game_a",
            "temporal_split": "train",
            "white_player_hash": "player_train_a_white",
            "black_player_hash": "player_train_a_black",
        },
        {
            "game_id": "train_game_b",
            "temporal_split": "train",
            "white_player_hash": "player_train_b_white",
            "black_player_hash": "player_train_b_black",
        },
        {
            "game_id": "train_game_c",
            "temporal_split": "train",
            "white_player_hash": "player_train_c_white",
            "black_player_hash": "player_train_c_black",
        },
    ]
    validation_games: list[dict[str, object]] = [
        {
            "game_id": "validation_game_a",
            "temporal_split": "validation",
            "white_player_hash": "player_validation_white",
            "black_player_hash": "player_validation_black",
        }
    ]
    test_games: list[dict[str, object]] = [
        {
            "game_id": "test_game_a",
            "temporal_split": "test",
            "white_player_hash": "player_test_white",
            "black_player_hash": "player_test_black",
        }
    ]

    train_game_path = dataset_root / "game_assignments" / "train.parquet"
    validation_game_path = dataset_root / "game_assignments" / "validation.parquet"
    test_game_path = dataset_root / "game_assignments" / "test.parquet"

    _write_parquet(train_game_path, train_games)
    _write_parquet(validation_game_path, validation_games)
    _write_parquet(test_game_path, test_games)

    manifest = {
        "status": "complete",
        "modeling_dataset_id": "baseline-fixture-modeling-dataset",
        "upstream": {
            "collection_id": "baseline-fixture-collection",
        },
        "versions": {
            "feature_schema_version": "policy_value_features_v1",
        },
        "splits": {
            "definition_version": "temporal_game_split_v1",
        },
        "datasets": {
            "policy_examples": {
                "train": {
                    "relative_files": ["policy_examples/train.parquet"],
                },
                "validation": {
                    "relative_files": ["policy_examples/validation.parquet"],
                },
                "test": {
                    "relative_files": ["policy_examples/test.parquet"],
                },
                "novel_position_test": {
                    "relative_files": ["policy_examples/novel_position_test.parquet"],
                },
            },
            "game_assignments": {
                "train": {
                    "relative_files": ["game_assignments/train.parquet"],
                },
                "validation": {
                    "relative_files": ["game_assignments/validation.parquet"],
                },
                "test": {
                    "relative_files": ["game_assignments/test.parquet"],
                },
            },
        },
    }

    manifest_path = dataset_root / "_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return manifest_path


def _write_baseline_config(path: Path, *, manifest_path: Path, output_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "input:",
                f"  modeling_manifest_path: {manifest_path.as_posix()}",
                "versions:",
                "  baseline_pipeline_version: classical_baselines_v1",
                "  feature_schema_version: policy_value_features_v1",
                "  split_definition_version: temporal_game_split_v1",
                "  rating_band_definition_version: rating_band_v1",
                "  game_phase_definition_version: game_phase_v1",
                "runtime:",
                "  seed: 7",
                "  threads: 1",
                "  early_stopping_rounds: 10",
                "limits:",
                "  max_train_positions: 100",
                "  max_validation_positions: 100",
                "  max_test_positions: 100",
                "  max_negative_candidates_per_train_position: 16",
                "  evaluate_all_legal_candidates_validation: true",
                "  evaluate_all_legal_candidates_test: true",
                "frequency:",
                "  backoff_levels:",
                "    - rating_band_phase",
                "    - phase",
                "    - global",
                "lightgbm:",
                "  n_estimators: 32",
                "  learning_rate: 0.1",
                "  num_leaves: 31",
                "  min_data_in_leaf: 1",
                "  feature_fraction: 1.0",
                "  lambda_l2: 1.0",
                "logistic:",
                "  solver: lbfgs",
                "  c: 1.0",
                "  max_iter: 300",
                "evaluation:",
                "  ranking_cutoffs:",
                "    - 1",
                "    - 3",
                "    - 5",
                "  ece_bins: 5",
                "  bootstrap_iterations: 8",
                "output:",
                f"  output_root: {output_root.as_posix()}",
                "  prediction_artifact_row_limit: 1000",
                "behavior:",
                "  strict: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_cli(config_path: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    args = [
        sys.executable,
        "-m",
        "chesslens.modeling.run_baselines",
        "--config",
        str(config_path),
    ]
    args.extend(extra_args)

    return subprocess.run(
        args,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_baselines_dry_run_and_validate_only(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)
    output_root = tmp_path / "baselines_output"
    config_path = tmp_path / "baseline.yaml"
    _write_baseline_config(config_path, manifest_path=manifest_path, output_root=output_root)

    dry_result = run_baselines(config_path=config_path, dry_run=True)
    assert dry_result.dry_run is True
    assert dry_result.validate_only is False
    assert dry_result.summary is not None
    assert dry_result.summary["split_row_counts_selected"]["train"] == 3
    assert dry_result.output_path.exists() is False

    validate_result = run_baselines(config_path=config_path, validate_only=True)
    assert validate_result.dry_run is False
    assert validate_result.validate_only is True
    assert validate_result.summary is not None
    assert validate_result.summary["leakage_audit"]["label_legality"]["status"] == "pass"
    assert validate_result.output_path.exists() is False


def test_run_baselines_end_to_end_and_reuse(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)
    output_root = tmp_path / "baselines_output"
    config_path = tmp_path / "baseline.yaml"
    _write_baseline_config(config_path, manifest_path=manifest_path, output_root=output_root)

    first = run_baselines(config_path=config_path)

    assert first.reused_existing is False
    assert first.output_path.exists()
    assert first.manifest_path.exists()

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["experiment_id"] == first.experiment_id

    for required in (
        "aggregate_metrics.json",
        "subgroup_metrics.json",
        "bootstrap_confidence_intervals.json",
        "lightgbm_model.txt",
        "logistic_pipeline.pkl",
        "frequency_model.json",
        "leakage_audit.json",
        "leakage_audit.md",
        "baseline_report.md",
    ):
        assert (first.output_path / required).exists()

    second = run_baselines(config_path=config_path)
    assert second.reused_existing is True
    assert second.output_path == first.output_path
    assert second.experiment_id == first.experiment_id


def test_run_baselines_cli_outputs_json_for_dry_run_and_validate_only(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)
    output_root = tmp_path / "baselines_output"
    config_path = tmp_path / "baseline.yaml"
    _write_baseline_config(config_path, manifest_path=manifest_path, output_root=output_root)

    dry = _run_cli(config_path, "--dry-run")
    assert dry.returncode == 0, dry.stderr
    dry_payload = _parse_single_json_document(dry.stdout)
    assert dry_payload["dry_run"] is True
    assert dry_payload["validate_only"] is False
    assert isinstance(dry_payload.get("summary"), dict)

    validate = _run_cli(config_path, "--validate-only")
    assert validate.returncode == 0, validate.stderr
    validate_payload = _parse_single_json_document(validate.stdout)
    assert validate_payload["dry_run"] is False
    assert validate_payload["validate_only"] is True
    assert isinstance(validate_payload.get("summary"), dict)
