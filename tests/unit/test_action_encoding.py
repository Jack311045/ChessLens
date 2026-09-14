from __future__ import annotations

import chess
import numpy as np

from chesslens.features.action_encoding import (
    ACTION_PLANES,
    ACTION_SPACE_SIZE,
    decode_index_to_move,
    encode_move_to_index,
    legal_move_mask,
)


def _idx(move_uci: str, board: chess.Board) -> int:
    return encode_move_to_index(chess.Move.from_uci(move_uci), board)


def test_action_space_size_and_plane_count() -> None:
    assert ACTION_PLANES == 73
    assert ACTION_SPACE_SIZE == 4672


def test_sliding_planes_all_directions() -> None:
    board = chess.Board("4k3/8/8/8/3Q4/8/8/4K3 w - - 0 1")
    from_square = chess.D4

    expected = {
        "d4d5": 0,
        "d4e5": 7,
        "d4e4": 14,
        "d4e3": 21,
        "d4d3": 28,
        "d4c3": 35,
        "d4c4": 42,
        "d4c5": 49,
        "d4d8": 3,
        "d4h4": 17,
    }
    for move_uci, expected_plane in expected.items():
        index = _idx(move_uci, board)
        assert index == from_square * ACTION_PLANES + expected_plane


def test_knight_planes() -> None:
    board = chess.Board("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1")
    expected = {
        "d4e6": 56,
        "d4f5": 57,
        "d4f3": 58,
        "d4e2": 59,
        "d4c2": 60,
        "d4b3": 61,
        "d4b5": 62,
        "d4c6": 63,
    }
    for move_uci, expected_plane in expected.items():
        index = _idx(move_uci, board)
        assert index == chess.D4 * ACTION_PLANES + expected_plane


def test_castling_encoded_as_two_square_king_move() -> None:
    white_board = chess.Board("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    assert _idx("e1g1", white_board) == chess.E1 * ACTION_PLANES + 15
    assert _idx("e1c1", white_board) == chess.E1 * ACTION_PLANES + 43

    black_board = chess.Board("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1")
    assert _idx("e8g8", black_board) == chess.E8 * ACTION_PLANES + 15
    assert _idx("e8c8", black_board) == chess.E8 * ACTION_PLANES + 43


def test_en_passant_encoded_as_pawn_diagonal() -> None:
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    index = _idx("e5d6", board)
    assert index == chess.E5 * ACTION_PLANES + 49


def test_queen_promotion_uses_sliding_plane_and_decodes() -> None:
    board = chess.Board("1k6/P7/8/8/8/8/8/7K w - - 0 1")
    move = chess.Move.from_uci("a7a8q")
    index = encode_move_to_index(move, board)
    decoded = decode_index_to_move(index, board)
    assert decoded == move


def test_underpromotions_and_capture_promotions() -> None:
    board = chess.Board("1k6/P7/8/8/8/8/8/7K w - - 0 1")
    assert _idx("a7a8n", board) == chess.A7 * ACTION_PLANES + 67
    assert _idx("a7a8b", board) == chess.A7 * ACTION_PLANES + 68
    assert _idx("a7a8r", board) == chess.A7 * ACTION_PLANES + 69

    white_capture_board = chess.Board("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    assert _idx("a7b8n", white_capture_board) == chess.A7 * ACTION_PLANES + 70
    assert _idx("a7b8b", white_capture_board) == chess.A7 * ACTION_PLANES + 71
    assert _idx("a7b8r", white_capture_board) == chess.A7 * ACTION_PLANES + 72

    black_capture_board = chess.Board("4k3/8/8/8/8/8/7p/6NK b - - 0 1")
    assert _idx("h2g1n", black_capture_board) == chess.H2 * ACTION_PLANES + 64
    assert _idx("h2g1b", black_capture_board) == chess.H2 * ACTION_PLANES + 65
    assert _idx("h2g1r", black_capture_board) == chess.H2 * ACTION_PLANES + 66


def test_encode_decode_round_trip_for_legal_moves() -> None:
    board = chess.Board("r2qk2r/ppp2ppp/2npbn2/3Np3/2B1P3/8/PPP2PPP/R1BQ1RK1 w kq - 0 10")
    for move in board.legal_moves:
        index = encode_move_to_index(move, board)
        decoded = decode_index_to_move(index, board)
        assert decoded == move


def test_legal_mask_matches_legal_move_count_and_no_collisions() -> None:
    board = chess.Board()
    mask = legal_move_mask(board)
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask.dtype == np.bool_
    assert int(mask.sum()) == board.legal_moves.count()