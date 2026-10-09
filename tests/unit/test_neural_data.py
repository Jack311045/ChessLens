from __future__ import annotations

import chess
import numpy as np

from chesslens.features.action_encoding import encode_move_to_index
from chesslens.features.position_encoding import normalize_fen, position_id_from_fen
from chesslens.modeling.neural_data import (
    UNKNOWN_CATEGORY_TOKEN,
    encode_examples,
    fit_preprocessor,
)
from chesslens.modeling.run_baselines import PositionExample


def _example(
    *,
    game_id: str,
    fen: str,
    move_uci: str,
    mover_rating: int,
    opponent_rating: int,
    mover_rating_band: str,
    time_control_category: str,
    eco: str,
    opening: str,
) -> PositionExample:
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    assert move in board.legal_moves
    return PositionExample(
        split="train",
        game_id=game_id,
        ply=0,
        source_month="2013-01",
        position_id=position_id_from_fen(fen),
        pre_move_fen=fen,
        normalized_pre_move_fen=normalize_fen(fen),
        side_to_move="w" if board.turn == chess.WHITE else "b",
        time_control_raw="300+0",
        time_control_category=time_control_category,
        eco=eco,
        opening=opening,
        mover_rating=mover_rating,
        opponent_rating=opponent_rating,
        rating_difference=mover_rating - opponent_rating,
        mover_rating_band=mover_rating_band,
        player_disjoint_training_eligible=True,
        is_player_holdout_game=False,
        policy_target_action_index=encode_move_to_index(move, board),
        value_target_wdl="win",
        novel_position_test=False,
    )


def test_preprocessor_fits_on_train_only_and_unknown_categories_fallback() -> None:
    train_a = _example(
        game_id="g1",
        fen=chess.STARTING_FEN,
        move_uci="e2e4",
        mover_rating=1000,
        opponent_rating=1200,
        mover_rating_band="1000-1199",
        time_control_category="blitz",
        eco="C20",
        opening="King Pawn Game",
    )
    train_b = _example(
        game_id="g2",
        fen=chess.STARTING_FEN,
        move_uci="d2d4",
        mover_rating=1400,
        opponent_rating=1300,
        mover_rating_band="1400-1599",
        time_control_category="rapid",
        eco="D00",
        opening="Queen Pawn Game",
    )

    preprocessor = fit_preprocessor([train_a, train_b])

    assert preprocessor.continuous_stats["mover_rating"].mean == 1200.0
    assert preprocessor.rating_band_to_index[UNKNOWN_CATEGORY_TOKEN] == 0
    assert preprocessor.time_control_to_index[UNKNOWN_CATEGORY_TOKEN] == 0

    validation = _example(
        game_id="g3",
        fen=chess.STARTING_FEN,
        move_uci="g1f3",
        mover_rating=2000,
        opponent_rating=2000,
        mover_rating_band="2000-2199",
        time_control_category="classical",
        eco="A04",
        opening="Reti",
    )
    encoded = encode_examples([validation], split="validation", preprocessor=preprocessor)[0]

    assert encoded.rating_band_index == 0
    assert encoded.time_control_index == 0
    assert np.isclose(encoded.continuous_context[0], (2000.0 - 1200.0) / 200.0)


def test_opening_metadata_changes_do_not_change_model_inputs() -> None:
    base = _example(
        game_id="g1",
        fen=chess.STARTING_FEN,
        move_uci="e2e4",
        mover_rating=1500,
        opponent_rating=1500,
        mover_rating_band="1400-1599",
        time_control_category="blitz",
        eco="C20",
        opening="King Pawn Game",
    )
    variant = _example(
        game_id="g1",
        fen=chess.STARTING_FEN,
        move_uci="e2e4",
        mover_rating=1500,
        opponent_rating=1500,
        mover_rating_band="1400-1599",
        time_control_category="blitz",
        eco="Z99",
        opening="Completely Different Opening Name",
    )

    preprocessor = fit_preprocessor([base])
    enc_base = encode_examples([base], split="train", preprocessor=preprocessor)[0]
    enc_variant = encode_examples([variant], split="train", preprocessor=preprocessor)[0]

    assert np.array_equal(enc_base.board_tensor, enc_variant.board_tensor)
    assert np.array_equal(enc_base.legal_mask, enc_variant.legal_mask)
    assert enc_base.policy_target_action_index == enc_variant.policy_target_action_index
    assert enc_base.value_target_index == enc_variant.value_target_index
    assert np.array_equal(enc_base.continuous_context, enc_variant.continuous_context)
    assert enc_base.rating_band_index == enc_variant.rating_band_index
    assert enc_base.time_control_index == enc_variant.time_control_index
