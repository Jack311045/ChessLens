from __future__ import annotations

import chess
import numpy as np

from chesslens.features.position_encoding import (
    BOARD_SHAPE,
    encode_board_18x8x8,
    normalize_fen,
    position_id_from_fen,
    position_id_from_normalized_fen,
)


def test_normalize_fen_uses_first_four_fields() -> None:
    fen = "8/8/8/8/8/8/8/8 w - - 37 81"
    assert normalize_fen(fen) == "8/8/8/8/8/8/8/8 w - -"


def test_normalized_fen_ignores_halfmove_and_fullmove() -> None:
    fen_a = "8/8/8/8/8/8/8/8 w - - 0 1"
    fen_b = "8/8/8/8/8/8/8/8 w - - 95 200"
    assert normalize_fen(fen_a) == normalize_fen(fen_b)


def test_position_id_stable() -> None:
    normalized = "8/8/8/8/8/8/8/8 w - -"
    assert position_id_from_normalized_fen(normalized) == position_id_from_normalized_fen(
        normalized
    )


def test_position_id_matches_fen_normalization() -> None:
    fen = chess.STARTING_FEN
    assert position_id_from_fen(fen) == position_id_from_normalized_fen(normalize_fen(fen))


def test_board_encoding_shape_and_dtype_and_determinism() -> None:
    board = chess.Board()
    original_fen = board.fen(en_passant="legal")
    encoded_a = encode_board_18x8x8(board)
    encoded_b = encode_board_18x8x8(board)

    assert encoded_a.shape == BOARD_SHAPE
    assert encoded_a.dtype == np.float32
    assert np.array_equal(encoded_a, encoded_b)
    assert board.fen(en_passant="legal") == original_fen


def test_side_to_move_plane_white_and_black() -> None:
    board = chess.Board()
    white_encoded = encode_board_18x8x8(board)
    assert np.all(white_encoded[12] == 1.0)

    board.push(chess.Move.from_uci("e2e4"))
    black_encoded = encode_board_18x8x8(board)
    assert np.all(black_encoded[12] == 0.0)


def test_castling_right_planes() -> None:
    board = chess.Board()
    encoded = encode_board_18x8x8(board)
    assert np.all(encoded[13] == 1.0)
    assert np.all(encoded[14] == 1.0)
    assert np.all(encoded[15] == 1.0)
    assert np.all(encoded[16] == 1.0)

    board_no_castling = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    encoded_no_castling = encode_board_18x8x8(board_no_castling)
    assert np.all(encoded_no_castling[13] == 0.0)
    assert np.all(encoded_no_castling[14] == 0.0)
    assert np.all(encoded_no_castling[15] == 0.0)
    assert np.all(encoded_no_castling[16] == 0.0)


def test_en_passant_plane_absent_and_present() -> None:
    board_absent = chess.Board()
    encoded_absent = encode_board_18x8x8(board_absent)
    assert np.count_nonzero(encoded_absent[17]) == 0

    board_present = chess.Board("8/8/8/8/3pP3/8/8/4K2k b - e3 0 1")
    encoded_present = encode_board_18x8x8(board_present)
    assert np.count_nonzero(encoded_present[17]) == 1
    assert encoded_present[17, 2, 4] == 1.0  # e3 -> file e (4), rank 3 (index 2)


def test_sparse_position_piece_planes() -> None:
    board = chess.Board("8/8/8/3Q4/8/8/8/4k2K w - - 0 1")
    encoded = encode_board_18x8x8(board)

    assert encoded[4, 4, 3] == 1.0  # white queen on d5
    assert encoded[11, 0, 4] == 1.0  # black king on e1
    assert encoded[5, 0, 7] == 1.0  # white king on h1