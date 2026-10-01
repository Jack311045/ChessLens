from __future__ import annotations

import json
from pathlib import Path

import chess
import duckdb
import pytest

from chesslens.features.position_encoding import normalize_fen, position_id_from_fen
from chesslens.modeling.build_dataset import run_modeling_dataset_build


def _write_collection_manifest(collection_root: Path) -> None:
    collection_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "collection_id": "collection-fixture-001",
        "status": "complete",
        "counts": {
            "accepted_games": 3,
            "emitted_moves": 5,
        },
    }
    (collection_root / "_collection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _move_rows_for_game(
    *,
    game_id: str,
    source_month: str,
    white_rating: int,
    black_rating: int,
    time_control_raw: str,
    eco: str,
    opening: str,
    result: str,
    termination: str,
    moves: list[str],
) -> list[tuple[object, ...]]:
    board = chess.Board()
    rows: list[tuple[object, ...]] = []
    for ply, move_uci in enumerate(moves):
        pre_move_fen = board.fen(en_passant="legal")
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise AssertionError(f"Illegal move in test fixture: {move_uci}")

        rows.append(
            (
                game_id,
                ply,
                source_month,
                position_id_from_fen(pre_move_fen),
                pre_move_fen,
                normalize_fen(pre_move_fen),
                "w" if board.turn == chess.WHITE else "b",
                move_uci,
                white_rating,
                black_rating,
                time_control_raw,
                eco,
                opening,
                result,
                termination,
            )
        )
        board.push(move)

    return rows


def _create_modeling_source_tables(db_path: Path, *, bad_move: bool = False) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute(
            """
            CREATE TABLE main.stg_games (
                game_id VARCHAR,
                source_month VARCHAR,
                played_date VARCHAR,
                result VARCHAR,
                white_player_hash VARCHAR,
                black_player_hash VARCHAR,
                white_rating INTEGER,
                black_rating INTEGER,
                time_control_raw VARCHAR,
                eco VARCHAR,
                opening VARCHAR,
                ply_count INTEGER
            )
            """
        )

        games = [
            (
                "g_train",
                "2013-01",
                "2013.01.05",
                "1-0",
                "white_a",
                "black_a",
                1500,
                1400,
                "300+0",
                "C20",
                "King Pawn Game",
                2,
            ),
            (
                "g_validation",
                "2013-01",
                "2013.01.24",
                "0-1",
                "white_b",
                "black_b",
                1700,
                1600,
                "180+2",
                "D00",
                "Queen Pawn Game",
                1,
            ),
            (
                "g_test",
                "2013-01",
                "2013.01.29",
                "1/2-1/2",
                "white_c",
                "black_c",
                1300,
                1250,
                "600+0",
                "A10",
                "English Opening",
                2,
            ),
        ]
        connection.executemany(
            """
            INSERT INTO main.stg_games (
                game_id, source_month, played_date, result,
                white_player_hash, black_player_hash,
                white_rating, black_rating,
                time_control_raw, eco, opening, ply_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            games,
        )

        move_rows = []
        move_rows.extend(
            _move_rows_for_game(
                game_id="g_train",
                source_month="2013-01",
                white_rating=1500,
                black_rating=1400,
                time_control_raw="300+0",
                eco="C20",
                opening="King Pawn Game",
                result="1-0",
                termination="Normal",
                moves=["e2e4", "e7e5"],
            )
        )
        move_rows.extend(
            _move_rows_for_game(
                game_id="g_validation",
                source_month="2013-01",
                white_rating=1700,
                black_rating=1600,
                time_control_raw="180+2",
                eco="D00",
                opening="Queen Pawn Game",
                result="0-1",
                termination="Normal",
                moves=["d2d4"],
            )
        )
        move_rows.extend(
            _move_rows_for_game(
                game_id="g_test",
                source_month="2013-01",
                white_rating=1300,
                black_rating=1250,
                time_control_raw="600+0",
                eco="A10",
                opening="English Opening",
                result="1/2-1/2",
                termination="Normal",
                moves=["c2c4", "e7e5"],
            )
        )

        if bad_move:
            first = move_rows[0]
            move_rows[0] = first[:7] + ("e7e5",) + first[8:]

        connection.execute(
            """
            CREATE TABLE main.int_move_context (
                game_id VARCHAR,
                ply INTEGER,
                source_month VARCHAR,
                position_id VARCHAR,
                pre_move_fen VARCHAR,
                normalized_pre_move_fen VARCHAR,
                side_to_move VARCHAR,
                played_move_uci VARCHAR,
                white_rating INTEGER,
                black_rating INTEGER,
                time_control_raw VARCHAR,
                eco VARCHAR,
                opening VARCHAR,
                result VARCHAR,
                termination VARCHAR
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO main.int_move_context (
                game_id, ply, source_month, position_id, pre_move_fen,
                normalized_pre_move_fen, side_to_move, played_move_uci,
                white_rating, black_rating, time_control_raw,
                eco, opening, result, termination
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            move_rows,
        )
    finally:
        connection.close()


def _write_modeling_config(
    path: Path, *, collection_root: Path, duckdb_path: Path, output_root: Path
) -> None:
    path.write_text(
        "\n".join(
            [
                "input:",
                f"  collection_root: {collection_root.as_posix()}",
                f"  duckdb_path: {duckdb_path.as_posix()}",
                "  expected_collection_id: collection-fixture-001",
                "  move_context_relation: main.int_move_context",
                "  games_relation: main.stg_games",
                "",
                "versions:",
                "  modeling_pipeline_version: modeling_dataset_v1",
                "  split_definition_version: temporal_game_split_v1",
                "  feature_schema_version: policy_value_features_v1",
                "  label_definition_version: policy_value_labels_v1",
                "  schema_version: 1.1.0",
                "  position_normalization_version: fen4_legal_ep_v1",
                "  board_encoding_version: board18_abs_v1",
                "  action_encoding_version: action8x8x73_v1",
                "",
                "sampling:",
                "  rule_version: sha256_mod_v1",
                "  seed: integration-seed",
                "  hash_modulus: 10000",
                "  hash_threshold: 10000",
                "  requested_rate_percent: 100.0",
                "  max_games: null",
                "",
                "splits:",
                "  train:",
                "    start_date: 2013-01-01",
                "    end_date: 2013-01-20",
                "  validation:",
                "    start_date: 2013-01-21",
                "    end_date: 2013-01-25",
                "  test:",
                "    start_date: 2013-01-26",
                "    end_date: 2013-01-31",
                "  missing_or_invalid_date_policy: assign_train",
                "",
                "player_holdout:",
                "  rule_version: sha256_mod_v1",
                "  seed: holdout-seed",
                "  hash_modulus: 10000",
                "  hash_threshold: 0",
                "  requested_rate_percent: 0.0",
                "  missing_player_hash_policy: exclude_from_player_disjoint_training",
                "",
                "output:",
                f"  output_root: {output_root.as_posix()}",
                "  batch_rows: 2",
                "  max_examples: null",
                "  parquet_compression: zstd",
                "  parquet_row_group_size: 2",
                "",
                "behavior:",
                "  strict: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _setup_fixture_environment(tmp_path: Path, *, bad_move: bool = False) -> tuple[Path, Path]:
    collection_root = tmp_path / "collection"
    duckdb_path = tmp_path / "warehouse.duckdb"
    output_root = tmp_path / "modeling"
    config_path = tmp_path / "modeling.yaml"

    _write_collection_manifest(collection_root)
    _create_modeling_source_tables(duckdb_path, bad_move=bad_move)
    _write_modeling_config(
        config_path,
        collection_root=collection_root,
        duckdb_path=duckdb_path,
        output_root=output_root,
    )
    return config_path, output_root


def test_modeling_build_idempotent_reuse(tmp_path: Path) -> None:
    config_path, _ = _setup_fixture_environment(tmp_path)

    first = run_modeling_dataset_build(config_path=config_path)
    second = run_modeling_dataset_build(config_path=config_path)

    assert first.reused_existing is False
    assert second.reused_existing is True
    assert first.modeling_dataset_id == second.modeling_dataset_id

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["counts"]["selected_games"] == 3
    assert manifest["counts"]["selected_policy_examples"] == 5
    assert manifest["counts"]["novel_position_test_rows"] == 1
    assert (first.dataset_path / "_SUCCESS").exists()


def test_modeling_build_rejects_tampered_manifest(tmp_path: Path) -> None:
    config_path, _ = _setup_fixture_environment(tmp_path)

    first = run_modeling_dataset_build(config_path=config_path)

    manifest_payload = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    manifest_payload["identity_payload_sha256"] = "tampered"
    first.manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="identity mismatch"):
        run_modeling_dataset_build(config_path=config_path)


def test_modeling_build_cleans_staging_on_failure(tmp_path: Path) -> None:
    config_path, output_root = _setup_fixture_environment(tmp_path, bad_move=True)

    with pytest.raises(RuntimeError, match="Policy label validation failed"):
        run_modeling_dataset_build(config_path=config_path)

    staging_root = output_root / "staging"
    if staging_root.exists():
        assert list(staging_root.iterdir()) == []
