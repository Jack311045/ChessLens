from __future__ import annotations

import pytest

from chesslens.modeling.labels import (
    mover_and_opponent_ratings,
    policy_target_action_index,
    rating_band,
    time_control_category,
    value_label_from_result,
)


def test_value_label_from_result_white_and_black() -> None:
    assert value_label_from_result(side_to_move="w", result="1-0") == "win"
    assert value_label_from_result(side_to_move="w", result="0-1") == "loss"
    assert value_label_from_result(side_to_move="b", result="1-0") == "loss"
    assert value_label_from_result(side_to_move="b", result="0-1") == "win"
    assert value_label_from_result(side_to_move="w", result="1/2-1/2") == "draw"


def test_policy_target_action_index_for_legal_move() -> None:
    index = policy_target_action_index(
        pre_move_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        played_move_uci="e2e4",
    )
    assert 0 <= index < 4672


def test_policy_target_action_index_rejects_illegal_move() -> None:
    with pytest.raises(ValueError, match="not legal"):
        policy_target_action_index(
            pre_move_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            played_move_uci="e7e5",
        )


def test_mover_and_opponent_ratings_and_bands() -> None:
    mover, opponent, diff = mover_and_opponent_ratings(
        side_to_move="w",
        white_rating=1700,
        black_rating=1500,
    )
    assert (mover, opponent, diff) == (1700, 1500, 200)
    assert rating_band(mover) == "1600_1999"
    assert rating_band(None) == "unknown"


def test_time_control_category() -> None:
    assert time_control_category("60+0") == "bullet"
    assert time_control_category("180+2") == "blitz"
    assert time_control_category("600+0") == "rapid"
    assert time_control_category("1800+0") == "classical"
    assert time_control_category(None) == "unknown"
