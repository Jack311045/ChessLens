from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chesslens.domain.records import GameRecord, IngestionErrorRecord, MoveRecord
from chesslens.validation.dataset_validation import DatasetValidationError, validate_staged_dataset
from chesslens.validation.schemas import (
    GAME_ARROW_SCHEMA,
    INGESTION_ERROR_ARROW_SCHEMA,
    MOVE_ARROW_SCHEMA,
    records_to_arrow_table,
)


def _write_partitioned_table(
    *,
    dataset_root: Path,
    dataset_name: str,
    source_month: str,
    table: pa.Table,
) -> None:
    output_dir = dataset_root / dataset_name / f"source_month={source_month}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_dir / "part-000000.parquet", compression="zstd", row_group_size=10)


def _valid_game(source_month: str) -> GameRecord:
    return GameRecord(
        game_id="game-1",
        source_archive="sample.pgn.zst",
        source_archive_sha256="a" * 64,
        source_month=source_month,
        source_game_index=0,
        event="Event",
        played_date="2026.01.01",
        round="1",
        result="1-0",
        white_rating=1500,
        black_rating=1400,
        time_control_raw="300+0",
        eco="C20",
        opening="King Pawn",
        termination="Normal",
        white_player_hash="player_a",
        black_player_hash="player_b",
        ply_count=2,
    )


def _valid_moves() -> list[MoveRecord]:
    return [
        MoveRecord(
            game_id="game-1",
            ply=0,
            position_id="pos-1",
            pre_move_fen="fen-1",
            normalized_pre_move_fen="nfen-1",
            side_to_move="w",
            played_move_uci="e2e4",
            played_move_san="e4",
            halfmove_clock=0,
            fullmove_number=1,
            clock_annotation_raw=None,
        ),
        MoveRecord(
            game_id="game-1",
            ply=1,
            position_id="pos-2",
            pre_move_fen="fen-2",
            normalized_pre_move_fen="nfen-2",
            side_to_move="b",
            played_move_uci="e7e5",
            played_move_san="e5",
            halfmove_clock=0,
            fullmove_number=1,
            clock_annotation_raw=None,
        ),
    ]


def _empty_errors() -> list[IngestionErrorRecord]:
    return []


def test_validate_staged_dataset_passes_for_consistent_data(tmp_path: Path) -> None:
    source_month = "2026-01"
    dataset_root = tmp_path / "dataset"

    _write_partitioned_table(
        dataset_root=dataset_root,
        dataset_name="games",
        source_month=source_month,
        table=records_to_arrow_table([_valid_game(source_month)], GAME_ARROW_SCHEMA),
    )
    _write_partitioned_table(
        dataset_root=dataset_root,
        dataset_name="moves",
        source_month=source_month,
        table=records_to_arrow_table(_valid_moves(), MOVE_ARROW_SCHEMA),
    )
    _write_partitioned_table(
        dataset_root=dataset_root,
        dataset_name="ingestion_errors",
        source_month=source_month,
        table=records_to_arrow_table(_empty_errors(), INGESTION_ERROR_ARROW_SCHEMA),
    )

    result = validate_staged_dataset(
        dataset_root=dataset_root,
        expected_source_month=source_month,
        expected_games=1,
        expected_moves=2,
        expected_errors=0,
    )

    assert result.games_rows == 1
    assert result.moves_rows == 2
    assert result.ingestion_error_rows == 0


def test_validate_staged_dataset_rejects_non_contiguous_ply(tmp_path: Path) -> None:
    source_month = "2026-01"
    dataset_root = tmp_path / "dataset"

    _write_partitioned_table(
        dataset_root=dataset_root,
        dataset_name="games",
        source_month=source_month,
        table=records_to_arrow_table([_valid_game(source_month)], GAME_ARROW_SCHEMA),
    )

    invalid_moves = [
        MoveRecord(
            game_id="game-1",
            ply=0,
            position_id="pos-1",
            pre_move_fen="fen-1",
            normalized_pre_move_fen="nfen-1",
            side_to_move="w",
            played_move_uci="e2e4",
            played_move_san="e4",
            halfmove_clock=0,
            fullmove_number=1,
            clock_annotation_raw=None,
        ),
        MoveRecord(
            game_id="game-1",
            ply=2,
            position_id="pos-2",
            pre_move_fen="fen-2",
            normalized_pre_move_fen="nfen-2",
            side_to_move="b",
            played_move_uci="e7e5",
            played_move_san="e5",
            halfmove_clock=0,
            fullmove_number=1,
            clock_annotation_raw=None,
        ),
    ]
    _write_partitioned_table(
        dataset_root=dataset_root,
        dataset_name="moves",
        source_month=source_month,
        table=records_to_arrow_table(invalid_moves, MOVE_ARROW_SCHEMA),
    )
    _write_partitioned_table(
        dataset_root=dataset_root,
        dataset_name="ingestion_errors",
        source_month=source_month,
        table=records_to_arrow_table(_empty_errors(), INGESTION_ERROR_ARROW_SCHEMA),
    )

    with pytest.raises(DatasetValidationError, match="non_contiguous_ply_games"):
        validate_staged_dataset(
            dataset_root=dataset_root,
            expected_source_month=source_month,
            expected_games=1,
            expected_moves=2,
            expected_errors=0,
        )
