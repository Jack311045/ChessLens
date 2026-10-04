"""Versioned modeling-data contracts for Phase 2.1."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from chesslens.domain.records import (
    ACTION_ENCODING_VERSION,
    BOARD_ENCODING_VERSION,
    POSITION_NORMALIZATION_VERSION,
    SCHEMA_VERSION,
)

MODELING_PIPELINE_VERSION = "modeling_dataset_v2"
SPLIT_DEFINITION_VERSION = "temporal_game_split_v1"
FEATURE_SCHEMA_VERSION = "policy_value_features_v1"
LABEL_DEFINITION_VERSION = "policy_value_labels_v1"
SAMPLING_RULE_VERSION = "sha256_mod_v1"
PLAYER_HOLDOUT_RULE_VERSION = "sha256_mod_v1"
VALUE_LABEL_VERSION = "wdl_side_to_move_v1"

TEMPORAL_SPLITS: tuple[str, str, str] = ("train", "validation", "test")
NOVEL_POSITION_TEST_SPLIT = "novel_position_test"

UPSTREAM_REQUIRED_MOVE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "ply",
    "source_month",
    "position_id",
    "pre_move_fen",
    "normalized_pre_move_fen",
    "side_to_move",
    "played_move_uci",
    "white_rating",
    "black_rating",
    "time_control_raw",
    "eco",
    "opening",
    "result",
    "termination",
)

UPSTREAM_REQUIRED_GAME_COLUMNS: tuple[str, ...] = (
    "game_id",
    "source_month",
    "played_date",
    "result",
    "white_player_hash",
    "black_player_hash",
    "white_rating",
    "black_rating",
    "time_control_raw",
    "eco",
    "opening",
    "ply_count",
)

GAME_ASSIGNMENT_ARROW_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("source_month", pa.string(), nullable=True),
        pa.field("played_date_raw", pa.string(), nullable=True),
        pa.field("played_date_iso", pa.string(), nullable=True),
        pa.field("played_date_source", pa.string(), nullable=False),
        pa.field("temporal_split", pa.string(), nullable=False),
        pa.field("temporal_split_reason", pa.string(), nullable=False),
        pa.field("sample_score_u64", pa.uint64(), nullable=False),
        pa.field("white_player_hash", pa.string(), nullable=True),
        pa.field("black_player_hash", pa.string(), nullable=True),
        pa.field("white_player_is_holdout", pa.bool_(), nullable=False),
        pa.field("black_player_is_holdout", pa.bool_(), nullable=False),
        pa.field("is_player_holdout_game", pa.bool_(), nullable=False),
        pa.field("player_disjoint_training_eligible", pa.bool_(), nullable=False),
        pa.field("result", pa.string(), nullable=True),
        pa.field("white_rating", pa.int32(), nullable=True),
        pa.field("black_rating", pa.int32(), nullable=True),
        pa.field("time_control_raw", pa.string(), nullable=True),
        pa.field("eco", pa.string(), nullable=True),
        pa.field("opening", pa.string(), nullable=True),
        pa.field("ply_count", pa.int32(), nullable=True),
    ]
)

POLICY_EXAMPLE_ARROW_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("ply", pa.int32(), nullable=False),
        pa.field("temporal_split", pa.string(), nullable=False),
        pa.field("source_month", pa.string(), nullable=True),
        pa.field("played_date_iso", pa.string(), nullable=True),
        pa.field("played_date_source", pa.string(), nullable=False),
        pa.field("position_id", pa.string(), nullable=False),
        pa.field("pre_move_fen", pa.string(), nullable=False),
        pa.field("normalized_pre_move_fen", pa.string(), nullable=False),
        pa.field("side_to_move", pa.string(), nullable=False),
        pa.field("time_control_raw", pa.string(), nullable=True),
        pa.field("time_control_category", pa.string(), nullable=False),
        pa.field("eco", pa.string(), nullable=True),
        pa.field("opening", pa.string(), nullable=True),
        pa.field("white_player_hash", pa.string(), nullable=True),
        pa.field("black_player_hash", pa.string(), nullable=True),
        pa.field("mover_rating", pa.int32(), nullable=True),
        pa.field("opponent_rating", pa.int32(), nullable=True),
        pa.field("rating_difference", pa.int32(), nullable=True),
        pa.field("mover_rating_band", pa.string(), nullable=False),
        pa.field("player_disjoint_training_eligible", pa.bool_(), nullable=False),
        pa.field("is_player_holdout_game", pa.bool_(), nullable=False),
        pa.field("played_move_uci_target", pa.string(), nullable=False),
        pa.field("policy_target_action_index", pa.int32(), nullable=False),
        pa.field("value_target_wdl", pa.string(), nullable=False),
        pa.field("final_result_target", pa.string(), nullable=True),
        pa.field("termination_target", pa.string(), nullable=True),
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("position_normalization_version", pa.string(), nullable=False),
        pa.field("board_encoding_version", pa.string(), nullable=False),
        pa.field("action_encoding_version", pa.string(), nullable=False),
        pa.field("feature_schema_version", pa.string(), nullable=False),
        pa.field("label_definition_version", pa.string(), nullable=False),
        pa.field("value_label_version", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class FeatureLabelContract:
    identifier_columns: tuple[str, ...]
    pre_move_feature_columns: tuple[str, ...]
    context_feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    audit_columns: tuple[str, ...]
    forbidden_feature_columns: tuple[str, ...]


def default_feature_label_contract() -> FeatureLabelContract:
    return FeatureLabelContract(
        identifier_columns=("game_id", "ply", "position_id"),
        pre_move_feature_columns=(
            "pre_move_fen",
            "normalized_pre_move_fen",
            "side_to_move",
            "mover_rating",
            "opponent_rating",
            "rating_difference",
            "mover_rating_band",
            "time_control_raw",
            "time_control_category",
            "eco",
            "opening",
            "source_month",
            "played_date_iso",
            "played_date_source",
        ),
        context_feature_columns=(
            "player_disjoint_training_eligible",
            "is_player_holdout_game",
        ),
        target_columns=(
            "played_move_uci_target",
            "policy_target_action_index",
            "value_target_wdl",
            "final_result_target",
            "termination_target",
        ),
        audit_columns=(
            "schema_version",
            "position_normalization_version",
            "board_encoding_version",
            "action_encoding_version",
            "feature_schema_version",
            "label_definition_version",
            "value_label_version",
        ),
        forbidden_feature_columns=(
            "played_move_uci_target",
            "policy_target_action_index",
            "final_result_target",
            "termination_target",
            "result",
            "played_move_uci",
            "post_move_fen",
            "future_clock_annotation",
            "engine_centipawn_score",
            "engine_mate_score",
            "blunder_label",
            "value_target_wdl",
        ),
    )


def validate_feature_label_contract(contract: FeatureLabelContract) -> None:
    feature_columns = set(contract.pre_move_feature_columns) | set(contract.context_feature_columns)
    forbidden = set(contract.forbidden_feature_columns)
    overlap = sorted(feature_columns & forbidden)
    if overlap:
        raise ValueError("Forbidden columns appear in declared feature set: " + ", ".join(overlap))


def supported_version_payload() -> dict[str, str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "position_normalization_version": POSITION_NORMALIZATION_VERSION,
        "board_encoding_version": BOARD_ENCODING_VERSION,
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "modeling_pipeline_version": MODELING_PIPELINE_VERSION,
        "split_definition_version": SPLIT_DEFINITION_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_definition_version": LABEL_DEFINITION_VERSION,
        "value_label_version": VALUE_LABEL_VERSION,
        "sampling_rule_version": SAMPLING_RULE_VERSION,
        "player_holdout_rule_version": PLAYER_HOLDOUT_RULE_VERSION,
    }
