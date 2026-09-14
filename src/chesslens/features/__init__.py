"""Feature encodings for chess positions and actions."""

from chesslens.features.action_encoding import (
    ACTION_ENCODING_VERSION,
    ACTION_SPACE_SIZE,
    decode_index_to_move,
    encode_move_to_index,
    legal_move_mask,
)
from chesslens.features.position_encoding import (
    BOARD_ENCODING_VERSION,
    POSITION_NORMALIZATION_VERSION,
    encode_board_18x8x8,
    normalize_fen,
    position_id_from_normalized_fen,
)

__all__ = [
    "POSITION_NORMALIZATION_VERSION",
    "BOARD_ENCODING_VERSION",
    "ACTION_ENCODING_VERSION",
    "ACTION_SPACE_SIZE",
    "normalize_fen",
    "position_id_from_normalized_fen",
    "encode_board_18x8x8",
    "encode_move_to_index",
    "decode_index_to_move",
    "legal_move_mask",
]