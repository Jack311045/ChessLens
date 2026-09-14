"""Fixed 8x8x73 action encoding (4672 actions) with legal move masking."""

from __future__ import annotations

import chess
import numpy as np

ACTION_ENCODING_VERSION = "action8x8x73_v1"

ACTION_PLANES = 73
BOARD_SQUARES = 64
ACTION_SPACE_SIZE = ACTION_PLANES * BOARD_SQUARES

# Sliding planes: 8 directions * 7 distances = 56 planes.
SLIDING_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)

# Knight planes: 8 offsets.
KNIGHT_OFFSETS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

# Underpromotion planes: 3 relative directions * 3 promoted pieces.
UNDERPROMOTION_FILE_DELTAS: tuple[int, ...] = (-1, 0, 1)
UNDERPROMOTION_PIECES: tuple[int, ...] = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

_DIRECTION_TO_INDEX = {direction: index for index, direction in enumerate(SLIDING_DIRECTIONS)}
_KNIGHT_TO_INDEX = {offset: index for index, offset in enumerate(KNIGHT_OFFSETS)}
_UNDERPROMOTION_PIECE_TO_INDEX = {
    piece: index for index, piece in enumerate(UNDERPROMOTION_PIECES)
}


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _square_coords(square: int) -> tuple[int, int]:
    return chess.square_file(square), chess.square_rank(square)


def _coords_to_square(file_index: int, rank_index: int) -> int | None:
    if 0 <= file_index < 8 and 0 <= rank_index < 8:
        return chess.square(file_index, rank_index)
    return None


def _sliding_plane(delta_file: int, delta_rank: int) -> int | None:
    if delta_file == 0 and delta_rank == 0:
        return None

    if not (delta_file == 0 or delta_rank == 0 or abs(delta_file) == abs(delta_rank)):
        return None

    distance = max(abs(delta_file), abs(delta_rank))
    if distance < 1 or distance > 7:
        return None

    direction = (_sign(delta_file), _sign(delta_rank))
    direction_index = _DIRECTION_TO_INDEX.get(direction)
    if direction_index is None:
        return None

    return direction_index * 7 + (distance - 1)


def _knight_plane(delta_file: int, delta_rank: int) -> int | None:
    index = _KNIGHT_TO_INDEX.get((delta_file, delta_rank))
    if index is None:
        return None
    return 56 + index


def _underpromotion_plane(move: chess.Move, board: chess.Board) -> int | None:
    if move.promotion not in _UNDERPROMOTION_PIECE_TO_INDEX:
        return None

    from_file, from_rank = _square_coords(move.from_square)
    to_file, to_rank = _square_coords(move.to_square)
    delta_file = to_file - from_file
    delta_rank = to_rank - from_rank

    expected_forward = 1 if board.turn == chess.WHITE else -1
    if delta_rank != expected_forward:
        raise ValueError("Underpromotion move has invalid forward direction for side to move")
    if delta_file not in UNDERPROMOTION_FILE_DELTAS:
        raise ValueError("Underpromotion move must use relative delta file in {-1, 0, 1}")

    dir_index = UNDERPROMOTION_FILE_DELTAS.index(delta_file)
    piece_index = _UNDERPROMOTION_PIECE_TO_INDEX[move.promotion]
    return 64 + dir_index * 3 + piece_index


def _promote_to_queen_if_needed(board: chess.Board, from_square: int, to_square: int) -> int | None:
    piece = board.piece_at(from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return None

    to_rank = chess.square_rank(to_square)
    if piece.color == chess.WHITE and to_rank == 7:
        return chess.QUEEN
    if piece.color == chess.BLACK and to_rank == 0:
        return chess.QUEEN
    return None


def encode_move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Map a chess move to a fixed action index in [0, 4671]."""
    underpromotion = _underpromotion_plane(move, board)
    if underpromotion is not None:
        return move.from_square * ACTION_PLANES + underpromotion

    from_file, from_rank = _square_coords(move.from_square)
    to_file, to_rank = _square_coords(move.to_square)
    delta_file = to_file - from_file
    delta_rank = to_rank - from_rank

    plane = _knight_plane(delta_file, delta_rank)
    if plane is None:
        plane = _sliding_plane(delta_file, delta_rank)
    if plane is None:
        raise ValueError(f"Move cannot be represented by 8x8x73 encoding: {move.uci()}")

    return move.from_square * ACTION_PLANES + plane


def decode_index_to_move(
    action_index: int,
    board: chess.Board,
    *,
    require_legal: bool = True,
) -> chess.Move:
    """Decode an action index to a move, using board context for promotions."""
    if action_index < 0 or action_index >= ACTION_SPACE_SIZE:
        raise ValueError(f"Action index out of range: {action_index}")

    from_square = action_index // ACTION_PLANES
    plane = action_index % ACTION_PLANES

    from_file, from_rank = _square_coords(from_square)
    piece = board.piece_at(from_square)
    if piece is None:
        raise ValueError(f"No piece at from-square for action index {action_index}")

    to_square: int | None
    promotion: int | None = None

    if plane < 56:
        direction_index = plane // 7
        distance = (plane % 7) + 1
        dir_file, dir_rank = SLIDING_DIRECTIONS[direction_index]
        to_square = _coords_to_square(
            from_file + dir_file * distance,
            from_rank + dir_rank * distance,
        )
        if to_square is not None:
            promotion = _promote_to_queen_if_needed(board, from_square, to_square)
    elif plane < 64:
        knight_index = plane - 56
        delta_file, delta_rank = KNIGHT_OFFSETS[knight_index]
        to_square = _coords_to_square(from_file + delta_file, from_rank + delta_rank)
    else:
        relative_index = plane - 64
        delta_file = UNDERPROMOTION_FILE_DELTAS[relative_index // 3]
        promotion = UNDERPROMOTION_PIECES[relative_index % 3]
        delta_rank = 1 if board.turn == chess.WHITE else -1
        to_square = _coords_to_square(from_file + delta_file, from_rank + delta_rank)

    if to_square is None:
        raise ValueError(f"Decoded destination is outside board for action index {action_index}")

    move = chess.Move(from_square=from_square, to_square=to_square, promotion=promotion)
    if require_legal and move not in board.legal_moves:
        raise ValueError(f"Decoded move is not legal in provided position: {move.uci()}")

    return move


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return boolean legal-action mask with shape (4672,)."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for move in board.legal_moves:
        action_index = encode_move_to_index(move, board)
        if mask[action_index]:
            raise ValueError(f"Action collision among legal moves at index {action_index}")
        mask[action_index] = True
    return mask


def legal_move_indices(board: chess.Board) -> list[int]:
    return [encode_move_to_index(move, board) for move in board.legal_moves]