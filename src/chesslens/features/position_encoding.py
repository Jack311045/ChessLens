"""FEN normalization, stable position IDs, and 18x8x8 board encoding."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import chess
import numpy as np
import numpy.typing as npt

POSITION_NORMALIZATION_VERSION = "fen4_legal_ep_v1"
BOARD_ENCODING_VERSION = "board18_abs_v1"

BOARD_SHAPE = (18, 8, 8)

# Channel order:
# 0..5   white pawn, knight, bishop, rook, queen, king
# 6..11  black pawn, knight, bishop, rook, queen, king
# 12     side to move (all ones when white to move, otherwise zeros)
# 13..16 castling rights planes: white king-side, white queen-side,
#        black king-side, black queen-side
# 17     legal en-passant target square (single hot square)
_PIECE_TYPE_TO_OFFSET = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


@dataclass(frozen=True)
class FenMetadata:
    normalized_fen: str
    side_to_move: Literal["w", "b"]
    castling_rights: str
    en_passant_square: str | None
    halfmove_clock: int
    fullmove_number: int


def _piece_plane_index(piece: chess.Piece) -> int:
    base = 0 if piece.color == chess.WHITE else 6
    return base + _PIECE_TYPE_TO_OFFSET[piece.piece_type]


def _legal_fen_parts(fen: str) -> tuple[str, str, str, str, int, int]:
    board = chess.Board(fen)
    legal_ep_fen = board.fen(en_passant="legal")
    parts = legal_ep_fen.split()
    if len(parts) != 6:
        raise ValueError(f"Expected 6 FEN parts, received {len(parts)}: {legal_ep_fen}")

    piece_placement, side_to_move, castling, en_passant, halfmove, fullmove = parts
    return (
        piece_placement,
        side_to_move,
        castling,
        en_passant,
        int(halfmove),
        int(fullmove),
    )


def normalize_fen(
    fen: str,
    normalization_version: str = POSITION_NORMALIZATION_VERSION,
) -> str:
    """Normalize to first four FEN fields using legal en-passant semantics."""
    if normalization_version != POSITION_NORMALIZATION_VERSION:
        raise ValueError(f"Unsupported normalization version: {normalization_version}")

    piece_placement, side_to_move, castling, en_passant, _, _ = _legal_fen_parts(fen)
    return f"{piece_placement} {side_to_move} {castling} {en_passant}"


def extract_fen_metadata(fen: str) -> FenMetadata:
    """Return normalized FEN and counters from a legal-en-passant FEN canonicalization."""
    piece_placement, side_to_move, castling, en_passant, halfmove, fullmove = _legal_fen_parts(fen)
    if side_to_move not in {"w", "b"}:
        raise ValueError(f"Unexpected side-to-move token in FEN: {side_to_move}")
    side_literal: Literal["w", "b"] = "w" if side_to_move == "w" else "b"

    return FenMetadata(
        normalized_fen=f"{piece_placement} {side_to_move} {castling} {en_passant}",
        side_to_move=side_literal,
        castling_rights=castling,
        en_passant_square=None if en_passant == "-" else en_passant,
        halfmove_clock=halfmove,
        fullmove_number=fullmove,
    )


def position_id_from_normalized_fen(
    normalized_fen: str,
    normalization_version: str = POSITION_NORMALIZATION_VERSION,
) -> str:
    payload = f"{normalization_version}|{normalized_fen}".encode()
    return hashlib.sha256(payload).hexdigest()


def position_id_from_fen(
    fen: str,
    normalization_version: str = POSITION_NORMALIZATION_VERSION,
) -> str:
    return position_id_from_normalized_fen(
        normalize_fen(fen, normalization_version=normalization_version),
        normalization_version=normalization_version,
    )


def encode_board_18x8x8(
    board: chess.Board,
    dtype: npt.DTypeLike = np.float32,
) -> np.ndarray:
    """
    Encode board state into channel-first tensor shape (18, 8, 8).

    Orientation is absolute:
    - file index 0 is file a
    - rank index 0 is rank 1
    - tensor indexing is [channel, rank, file]
    """
    encoded = np.zeros(BOARD_SHAPE, dtype=dtype)

    for square, piece in board.piece_map().items():
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        encoded[_piece_plane_index(piece), rank_index, file_index] = 1.0

    if board.turn == chess.WHITE:
        encoded[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        encoded[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        encoded[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        encoded[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        encoded[16, :, :] = 1.0

    if board.has_legal_en_passant() and board.ep_square is not None:
        file_index = chess.square_file(board.ep_square)
        rank_index = chess.square_rank(board.ep_square)
        encoded[17, rank_index, file_index] = 1.0

    return encoded


def encode_fen_18x8x8(
    fen: str,
    dtype: npt.DTypeLike = np.float32,
) -> np.ndarray:
    return encode_board_18x8x8(chess.Board(fen), dtype=dtype)