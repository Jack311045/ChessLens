from __future__ import annotations

import json
from pathlib import Path

import chess
import pyarrow.parquet as pq

from chesslens.ingestion.pgn_reader import iter_parsed_games, iter_raw_games
from chesslens.validation.schemas import (
    GAME_ARROW_SCHEMA,
    MOVE_ARROW_SCHEMA,
    records_to_arrow_table,
)

FIXTURE_PATH = Path("data/fixtures/lichess_2013_01_first20.pgn.zst")
MANIFEST_PATH = Path("data/manifests/fixture_2013_01_first20.json")


def test_fixture_files_exist() -> None:
    assert FIXTURE_PATH.exists()
    assert MANIFEST_PATH.exists()


def test_streaming_read_expected_number_of_games() -> None:
    games = list(iter_raw_games(FIXTURE_PATH, max_games=100))
    assert len(games) == 20


def test_every_move_is_legal_and_pre_move_fen_is_captured_before_push() -> None:
    for parsed in iter_parsed_games(
        archive_path=FIXTURE_PATH,
        max_games=20,
        strict=True,
        source_archive_sha256="fixture",
    ):
        board = chess.Board()
        assert parsed.game_record.ply_count == len(parsed.move_records)

        for move_record in parsed.move_records:
            assert move_record.pre_move_fen == board.fen(en_passant="legal")
            move = chess.Move.from_uci(move_record.played_move_uci)
            assert move in board.legal_moves
            board.push(move)


def test_schema_conformance_and_arrow_round_trip(tmp_path: Path) -> None:
    parsed_games = list(
        iter_parsed_games(
            archive_path=FIXTURE_PATH,
            max_games=20,
            strict=True,
            source_archive_sha256="fixture",
        )
    )
    game_records = [item.game_record for item in parsed_games]
    move_records = [move for item in parsed_games for move in item.move_records]

    game_table = records_to_arrow_table(game_records, GAME_ARROW_SCHEMA)
    move_table = records_to_arrow_table(move_records, MOVE_ARROW_SCHEMA)

    game_file = tmp_path / "games.parquet"
    move_file = tmp_path / "moves.parquet"
    pq.write_table(game_table, game_file)
    pq.write_table(move_table, move_file)

    loaded_games = pq.read_table(game_file)
    loaded_moves = pq.read_table(move_file)

    assert loaded_games.schema.equals(game_table.schema)
    assert loaded_moves.schema.equals(move_table.schema)
    assert loaded_games.num_rows == len(game_records)
    assert loaded_moves.num_rows == len(move_records)


def test_repeated_runs_produce_same_ids() -> None:
    run_a = list(
        iter_parsed_games(
            archive_path=FIXTURE_PATH,
            max_games=20,
            strict=True,
            source_archive_sha256="fixture",
        )
    )
    run_b = list(
        iter_parsed_games(
            archive_path=FIXTURE_PATH,
            max_games=20,
            strict=True,
            source_archive_sha256="fixture",
        )
    )

    ids_a = [item.game_record.game_id for item in run_a]
    ids_b = [item.game_record.game_id for item in run_b]
    assert ids_a == ids_b

    moves_a = [[move.position_id for move in item.move_records] for item in run_a]
    moves_b = [[move.position_id for move in item.move_records] for item in run_b]
    assert moves_a == moves_b


def test_manifest_declares_expected_fixture_size() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["selected_games"] == 20