"""Supervised policy/value label builders for modeling examples."""

from __future__ import annotations

import chess

from chesslens.features.action_encoding import decode_index_to_move, encode_move_to_index


def value_label_from_result(*, side_to_move: str, result: str | None) -> str:
    if result not in {"1-0", "0-1", "1/2-1/2"}:
        raise ValueError(f"Unsupported or missing final result for value label: {result!r}")

    if side_to_move == "w":
        if result == "1-0":
            return "win"
        if result == "0-1":
            return "loss"
        return "draw"

    if side_to_move == "b":
        if result == "1-0":
            return "loss"
        if result == "0-1":
            return "win"
        return "draw"

    raise ValueError(f"Unsupported side_to_move token: {side_to_move!r}")


def policy_target_action_index(*, pre_move_fen: str, played_move_uci: str) -> int:
    try:
        board = chess.Board(pre_move_fen)
    except ValueError as exc:
        raise ValueError(f"Invalid pre_move_fen: {pre_move_fen!r}") from exc

    try:
        move = chess.Move.from_uci(played_move_uci)
    except ValueError as exc:
        raise ValueError(f"Invalid played_move_uci: {played_move_uci!r}") from exc

    if move not in board.legal_moves:
        raise ValueError(
            "Played move is not legal in pre-move position: "
            f"{played_move_uci!r} on {pre_move_fen!r}"
        )

    action_index = encode_move_to_index(move, board)
    if action_index < 0 or action_index >= 4672:
        raise ValueError(f"Action index out of expected range [0, 4671]: {action_index}")

    decoded = decode_index_to_move(action_index, board, require_legal=True)
    if decoded != move:
        raise ValueError(
            "Encode/decode mismatch for policy target move: "
            f"move={played_move_uci!r}, decoded={decoded.uci()!r}"
        )

    return action_index


def mover_and_opponent_ratings(
    *,
    side_to_move: str,
    white_rating: int | None,
    black_rating: int | None,
) -> tuple[int | None, int | None, int | None]:
    if side_to_move == "w":
        mover = white_rating
        opponent = black_rating
    elif side_to_move == "b":
        mover = black_rating
        opponent = white_rating
    else:
        raise ValueError(f"Unsupported side_to_move token: {side_to_move!r}")

    rating_diff: int | None
    if mover is None or opponent is None:
        rating_diff = None
    else:
        rating_diff = mover - opponent

    return mover, opponent, rating_diff


def rating_band(rating: int | None) -> str:
    if rating is None:
        return "unknown"
    if rating < 1200:
        return "lt_1200"
    if rating < 1600:
        return "1200_1599"
    if rating < 2000:
        return "1600_1999"
    return "gte_2000"


def time_control_category(raw: str | None) -> str:
    if raw is None:
        return "unknown"
    text = raw.strip()
    if not text or text == "-":
        return "unknown"

    if "+" not in text:
        return "unknown"

    base_raw = text.split("+", maxsplit=1)[0]
    try:
        base_seconds = int(base_raw)
    except ValueError:
        return "unknown"

    if base_seconds < 180:
        return "bullet"
    if base_seconds < 480:
        return "blitz"
    if base_seconds < 1500:
        return "rapid"
    return "classical"
