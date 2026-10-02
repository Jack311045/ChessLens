from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import chesslens.modeling.run_baselines as run_baselines_module
from chesslens.features.action_encoding import encode_move_to_index
from chesslens.features.position_encoding import normalize_fen, position_id_from_fen
from chesslens.modeling.labels import rating_band, time_control_category
from chesslens.modeling.run_baselines import (
    FrequencyPolicyModel,
    _frequency_score,
    run_baselines,
)


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
    player_disjoint_training_eligible: bool,
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
        "player_disjoint_training_eligible": player_disjoint_training_eligible,
        "is_player_holdout_game": is_player_holdout_game,
        "policy_target_action_index": encode_move_to_index(target_move, board),
        "value_target_wdl": value_target_wdl,
    }


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_modeling_fixture(
    tmp_path: Path,
    *,
    dataset_name: str = "modeling_dataset",
    ineligible_row_variant: str = "default",
    inject_illegal_selected_train_target: bool = False,
) -> Path:
    dataset_root = tmp_path / dataset_name

    if ineligible_row_variant not in {"default", "alternate"}:
        raise ValueError(f"Unsupported ineligible_row_variant={ineligible_row_variant!r}")

    ineligible_target_move = "b1c3"
    ineligible_value = "win"
    ineligible_mover_rating = 1250
    ineligible_opponent_rating = 1525
    if ineligible_row_variant == "alternate":
        ineligible_target_move = "g1f3"
        ineligible_value = "loss"
        ineligible_mover_rating = 2100
        ineligible_opponent_rating = 2000

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
            player_disjoint_training_eligible=True,
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
            player_disjoint_training_eligible=True,
            is_player_holdout_game=False,
        ),
        _policy_row(
            split="train",
            game_id="train_game_c",
            source_month="2013-01",
            fen=_fen_after(["g2g3", "d7d5"]),
            target_move_uci="f1g2",
            value_target_wdl="loss",
            mover_rating=1300,
            opponent_rating=1350,
            time_control_raw="600+0",
            eco="A10",
            opening="English Opening",
            player_disjoint_training_eligible=True,
            is_player_holdout_game=False,
        ),
        _policy_row(
            split="train",
            game_id="train_game_d_holdout",
            source_month="2013-01",
            fen=_fen_after(["c2c4", "e7e5"]),
            target_move_uci=ineligible_target_move,
            value_target_wdl=ineligible_value,
            mover_rating=ineligible_mover_rating,
            opponent_rating=ineligible_opponent_rating,
            time_control_raw="120+1",
            eco="A20",
            opening="English: Reversed Sicilian",
            player_disjoint_training_eligible=False,
            is_player_holdout_game=True,
        ),
    ]

    if inject_illegal_selected_train_target:
        train_rows[0]["policy_target_action_index"] = 9999

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
            player_disjoint_training_eligible=False,
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
            player_disjoint_training_eligible=False,
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
        {
            "game_id": "train_game_d_holdout",
            "temporal_split": "train",
            "white_player_hash": "player_holdout_shared_white",
            "black_player_hash": "player_holdout_shared_black",
        },
    ]
    validation_games: list[dict[str, object]] = [
        {
            "game_id": "validation_game_a",
            "temporal_split": "validation",
            "white_player_hash": "player_holdout_shared_white",
            "black_player_hash": "player_validation_black",
        }
    ]
    test_games: list[dict[str, object]] = [
        {
            "game_id": "test_game_a",
            "temporal_split": "test",
            "white_player_hash": "player_test_white",
            "black_player_hash": "player_holdout_shared_black",
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
    _write_baseline_config_custom(
        path,
        manifest_path=manifest_path,
        output_root=output_root,
    )


def _write_baseline_config_custom(
    path: Path,
    *,
    manifest_path: Path,
    output_root: Path,
    max_train_positions: int = 100,
    max_validation_positions: int = 100,
    max_test_positions: int = 100,
    max_negative_candidates_per_train_position: int = 16,
    evaluate_all_legal_candidates_validation: bool = True,
    evaluate_all_legal_candidates_test: bool = True,
    training_population_mode: str = "temporal_all",
    legality_scope: str = "selected",
    frequency_backoff_levels: tuple[str, ...] = ("rating_band_phase", "phase", "global"),
) -> None:
    validation_flag = str(evaluate_all_legal_candidates_validation).lower()
    test_flag = str(evaluate_all_legal_candidates_test).lower()
    backoff_lines = [f"    - {level}" for level in frequency_backoff_levels]

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
                f"  max_train_positions: {max_train_positions}",
                f"  max_validation_positions: {max_validation_positions}",
                f"  max_test_positions: {max_test_positions}",
                (
                    "  max_negative_candidates_per_train_position: "
                    f"{max_negative_candidates_per_train_position}"
                ),
                f"  evaluate_all_legal_candidates_validation: {validation_flag}",
                f"  evaluate_all_legal_candidates_test: {test_flag}",
                "training_population:",
                f"  mode: {training_population_mode}",
                "preflight:",
                f"  legality_scope: {legality_scope}",
                "frequency:",
                "  backoff_levels:",
                *backoff_lines,
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_run_baselines_dry_run_and_validate_only(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)
    output_root = tmp_path / "baselines_output"
    config_path = tmp_path / "baseline.yaml"
    _write_baseline_config(config_path, manifest_path=manifest_path, output_root=output_root)

    dry_result = run_baselines(config_path=config_path, dry_run=True)
    assert dry_result.dry_run is True
    assert dry_result.validate_only is False
    assert dry_result.summary is not None
    assert dry_result.summary["split_row_counts_selected"]["train"] == 4
    assert dry_result.summary["legality_validation"]["status"] == "not_run_dry_run"
    assert dry_result.summary["training_population"]["mode"] == "temporal_all"
    assert dry_result.output_path.exists() is False

    validate_result = run_baselines(config_path=config_path, validate_only=True)
    assert validate_result.dry_run is False
    assert validate_result.validate_only is True
    assert validate_result.summary is not None
    assert validate_result.summary["leakage_audit"]["label_legality"]["status"] == "pass"
    assert validate_result.summary["leakage_audit"]["label_legality"]["scope"] == "selected"
    assert validate_result.summary["leakage_audit"]["label_legality"]["checked_rows"] == 6
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
    assert "artifact_integrity" in manifest
    assert manifest["counts"]["training_population"]["full_train_row_count"] == 4
    assert manifest["counts"]["training_population"]["eligible_train_row_count"] == 3

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
        assert required in manifest["artifact_integrity"]

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


def test_player_disjoint_mode_excludes_ineligible_example_for_all_models(tmp_path: Path) -> None:
    manifest_a = _write_modeling_fixture(
        tmp_path,
        dataset_name="dataset_a",
        ineligible_row_variant="default",
    )
    manifest_b = _write_modeling_fixture(
        tmp_path,
        dataset_name="dataset_b",
        ineligible_row_variant="alternate",
    )

    config_a = tmp_path / "baseline_player_disjoint_a.yaml"
    config_b = tmp_path / "baseline_player_disjoint_b.yaml"
    _write_baseline_config_custom(
        config_a,
        manifest_path=manifest_a,
        output_root=tmp_path / "out_a",
        training_population_mode="player_disjoint",
        legality_scope="selected",
    )
    _write_baseline_config_custom(
        config_b,
        manifest_path=manifest_b,
        output_root=tmp_path / "out_b",
        training_population_mode="player_disjoint",
        legality_scope="selected",
    )

    first = run_baselines(config_path=config_a)
    second = run_baselines(config_path=config_b)

    first_agg = _read_json(first.output_path / "aggregate_metrics.json")
    second_agg = _read_json(second.output_path / "aggregate_metrics.json")
    assert first_agg == second_agg

    first_frequency = _read_json(first.output_path / "frequency_model.json")
    second_frequency = _read_json(second.output_path / "frequency_model.json")
    assert first_frequency == second_frequency

    first_lightgbm_model = (first.output_path / "lightgbm_model.txt").read_text(encoding="utf-8")
    second_lightgbm_model = (second.output_path / "lightgbm_model.txt").read_text(
        encoding="utf-8"
    )
    assert first_lightgbm_model == second_lightgbm_model


def test_player_holdout_subgroup_interpretation_labels(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)

    temporal_all_config = tmp_path / "baseline_temporal_all.yaml"
    player_disjoint_config = tmp_path / "baseline_player_disjoint.yaml"
    _write_baseline_config_custom(
        temporal_all_config,
        manifest_path=manifest_path,
        output_root=tmp_path / "temporal_all_out",
        training_population_mode="temporal_all",
    )
    _write_baseline_config_custom(
        player_disjoint_config,
        manifest_path=manifest_path,
        output_root=tmp_path / "player_disjoint_out",
        training_population_mode="player_disjoint",
    )

    temporal_all_result = run_baselines(config_path=temporal_all_config)
    player_disjoint_result = run_baselines(config_path=player_disjoint_config)

    temporal_subgroups = _read_json(temporal_all_result.output_path / "subgroup_metrics.json")
    disjoint_subgroups = _read_json(player_disjoint_result.output_path / "subgroup_metrics.json")

    for model_key in ("ranking_frequency_test", "ranking_lightgbm_test", "logistic_wdl_test"):
        assert (
            temporal_subgroups[model_key]["player_holdout_true"]["interpretation"]
            == "descriptive_only_not_player_disjoint"
        )
        assert (
            disjoint_subgroups[model_key]["player_holdout_true"]["interpretation"]
            == "player_disjoint"
        )


def test_legality_scope_selected_vs_full_checked_rows(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)
    selected_config = tmp_path / "baseline_selected.yaml"
    full_config = tmp_path / "baseline_full.yaml"

    _write_baseline_config_custom(
        selected_config,
        manifest_path=manifest_path,
        output_root=tmp_path / "selected_out",
        legality_scope="selected",
        max_train_positions=1,
        max_validation_positions=1,
        max_test_positions=1,
    )
    _write_baseline_config_custom(
        full_config,
        manifest_path=manifest_path,
        output_root=tmp_path / "full_out",
        legality_scope="full",
        max_train_positions=1,
        max_validation_positions=1,
        max_test_positions=1,
    )

    selected = run_baselines(config_path=selected_config, validate_only=True)
    full = run_baselines(config_path=full_config, validate_only=True)
    assert selected.summary is not None
    assert full.summary is not None

    selected_legality = selected.summary["leakage_audit"]["label_legality"]
    full_legality = full.summary["leakage_audit"]["label_legality"]

    assert selected_legality["scope"] == "selected"
    assert selected_legality["checked_rows"] == 3
    assert full_legality["scope"] == "full"
    assert full_legality["checked_rows"] == 6
    assert full_legality["checked_rows_by_split"] == {
        "train": 4,
        "validation": 1,
        "test": 1,
    }


def test_dry_run_skips_legality_board_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)
    config_path = tmp_path / "baseline.yaml"
    _write_baseline_config_custom(
        config_path,
        manifest_path=manifest_path,
        output_root=tmp_path / "dry_run_out",
        legality_scope="full",
    )

    def _should_not_run(*_: object, **__: object) -> object:
        raise AssertionError("legality validation should not run during dry-run")

    monkeypatch.setattr(
        run_baselines_module,
        "_validate_label_legality_streaming",
        _should_not_run,
    )
    monkeypatch.setattr(
        run_baselines_module,
        "_validate_label_legality_selected",
        _should_not_run,
    )

    dry_result = run_baselines(config_path=config_path, dry_run=True)
    assert dry_result.summary is not None
    assert dry_result.summary["legality_validation"]["status"] == "not_run_dry_run"
    assert dry_result.summary["legality_validation"]["checked_rows"] == 0


def test_selected_legality_scope_rejects_illegal_selected_target(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(
        tmp_path,
        inject_illegal_selected_train_target=True,
    )
    config_path = tmp_path / "baseline.yaml"
    _write_baseline_config_custom(
        config_path,
        manifest_path=manifest_path,
        output_root=tmp_path / "illegal_selected_out",
        legality_scope="selected",
    )

    with pytest.raises(RuntimeError, match="scope=selected"):
        run_baselines(config_path=config_path, validate_only=True)


def test_frequency_backoff_order_controls_behavior() -> None:
    model = FrequencyPolicyModel(
        level1_counts={
            ("mid", "opening", 100): 1,
            ("mid", "opening", 200): 2,
        },
        level2_counts={
            ("opening", 100): 10,
            ("opening", 200): 1,
        },
        global_counts={100: 1, 200: 1},
    )

    score_action_100_rb_first = _frequency_score(
        model,
        backoff_levels=("rating_band_phase", "phase", "global"),
        rating_band_value="mid",
        phase="opening",
        action=100,
    ).score
    score_action_200_rb_first = _frequency_score(
        model,
        backoff_levels=("rating_band_phase", "phase", "global"),
        rating_band_value="mid",
        phase="opening",
        action=200,
    ).score
    assert score_action_200_rb_first > score_action_100_rb_first

    score_action_100_phase_first = _frequency_score(
        model,
        backoff_levels=("phase", "rating_band_phase", "global"),
        rating_band_value="mid",
        phase="opening",
        action=100,
    ).score
    score_action_200_phase_first = _frequency_score(
        model,
        backoff_levels=("phase", "rating_band_phase", "global"),
        rating_band_value="mid",
        phase="opening",
        action=200,
    ).score
    assert score_action_100_phase_first > score_action_200_phase_first


def test_candidate_flags_false_mark_non_publishable_and_sampled(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)
    config_path = tmp_path / "baseline_sampled_eval.yaml"
    _write_baseline_config_custom(
        config_path,
        manifest_path=manifest_path,
        output_root=tmp_path / "sampled_eval_out",
        evaluate_all_legal_candidates_validation=False,
        evaluate_all_legal_candidates_test=False,
        max_negative_candidates_per_train_position=1,
    )

    result = run_baselines(config_path=config_path)
    aggregate = _read_json(result.output_path / "aggregate_metrics.json")
    manifest = _read_json(result.output_path / "_manifest.json")

    assert aggregate["evaluation_candidate_mode"]["mode"] == "non_publishable_sampled_candidates"
    assert aggregate["evaluation_candidate_mode"]["publishable"] is False
    assert manifest["evaluation_candidate_mode"]["mode"] == "non_publishable_sampled_candidates"
    assert manifest["counts"]["candidate_rows"]["validation"] <= 2
    assert manifest["counts"]["candidate_rows"]["test"] <= 2


def test_dry_run_candidate_estimate_respects_negative_cap(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)
    config_path = tmp_path / "baseline_cap_estimate.yaml"
    _write_baseline_config_custom(
        config_path,
        manifest_path=manifest_path,
        output_root=tmp_path / "cap_estimate_out",
        training_population_mode="player_disjoint",
        max_negative_candidates_per_train_position=1,
    )

    dry_result = run_baselines(config_path=config_path, dry_run=True)
    assert dry_result.summary is not None
    assert dry_result.summary["training_population"]["training_row_count_used"] == 3
    assert dry_result.summary["estimated_candidate_rows"]["train"] == 6


def test_reuse_detects_tampered_artifact(tmp_path: Path) -> None:
    manifest_path = _write_modeling_fixture(tmp_path)
    config_path = tmp_path / "baseline_tamper.yaml"
    _write_baseline_config_custom(
        config_path,
        manifest_path=manifest_path,
        output_root=tmp_path / "tamper_out",
    )

    first = run_baselines(config_path=config_path)
    (first.output_path / "aggregate_metrics.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="integrity mismatch"):
        run_baselines(config_path=config_path)
