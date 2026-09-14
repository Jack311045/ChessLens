from __future__ import annotations

import chess
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

from chesslens.features.action_encoding import (
    decode_index_to_move,
    encode_move_to_index,
    legal_move_mask,
)
from chesslens.features.position_encoding import encode_board_18x8x8


@st.composite
def legal_boards(draw: st.DrawFn) -> chess.Board:
    board = chess.Board()
    plies = draw(st.integers(min_value=0, max_value=80))
    for _ in range(plies):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = draw(st.sampled_from(legal_moves))
        board.push(move)
    return board


@given(board=legal_boards())
@settings(max_examples=120)
def test_action_round_trip_no_collisions_and_mask_count(board: chess.Board) -> None:
    indices: set[int] = set()
    legal_moves = list(board.legal_moves)

    for move in legal_moves:
        index = encode_move_to_index(move, board)
        decoded = decode_index_to_move(index, board)
        assert decoded == move
        assert index not in indices
        indices.add(index)

    mask = legal_move_mask(board)
    assert int(mask.sum()) == len(legal_moves)


@given(board=legal_boards())
@settings(max_examples=120)
def test_position_encoding_deterministic_and_non_mutating(board: chess.Board) -> None:
    before = board.fen(en_passant="legal")
    encoded_a = encode_board_18x8x8(board)
    encoded_b = encode_board_18x8x8(board)
    after = board.fen(en_passant="legal")

    assert before == after
    assert np.array_equal(encoded_a, encoded_b)