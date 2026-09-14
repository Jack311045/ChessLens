from __future__ import annotations

import pyarrow as pa

from chesslens.domain.records import GameRecord, MoveRecord
from chesslens.validation.schemas import (
    GAME_ARROW_SCHEMA,
    MOVE_ARROW_SCHEMA,
    records_to_arrow_table,
)


def test_game_schema_fields_present() -> None:
    names = GAME_ARROW_SCHEMA.names
    assert "game_id" in names
    assert "source_archive_sha256" in names
    assert "white_player_hash" in names


def test_move_schema_types() -> None:
    assert MOVE_ARROW_SCHEMA.field("ply").type == pa.int32()
    assert MOVE_ARROW_SCHEMA.field("played_move_uci").type == pa.string()


def test_arrow_table_conversion_for_records() -> None:
    game = GameRecord(
        game_id="g1",
        source_archive="a.zst",
        source_archive_sha256="sha",
        source_month="2013-01",
        source_game_index=0,
        event="Rated",
        played_date="2013.01.01",
        round="?",
        result="1-0",
        white_rating=1500,
        black_rating=1400,
        time_control_raw="300+0",
        eco="C20",
        opening="King's Pawn",
        termination="Normal",
        white_player_hash="w",
        black_player_hash="b",
        ply_count=2,
    )
    move = MoveRecord(
        game_id="g1",
        ply=0,
        position_id="p1",
        pre_move_fen="fen",
        normalized_pre_move_fen="nfen",
        side_to_move="w",
        played_move_uci="e2e4",
        played_move_san="e4",
        halfmove_clock=0,
        fullmove_number=1,
        clock_annotation_raw=None,
    )

    game_table = records_to_arrow_table([game], GAME_ARROW_SCHEMA)
    move_table = records_to_arrow_table([move], MOVE_ARROW_SCHEMA)
    assert game_table.num_rows == 1
    assert move_table.num_rows == 1