"""Data preparation for Phase 3.1 neural policy/value training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess
import numpy as np
import numpy.typing as npt

from chesslens.features.action_encoding import ACTION_SPACE_SIZE, legal_move_mask
from chesslens.features.position_encoding import encode_fen_18x8x8
from chesslens.modeling.run_baselines import PositionExample
from chesslens.modeling.validation import canonical_json, sha256_text

WDL_CLASS_ORDER: tuple[str, str, str] = ("win", "draw", "loss")
CONTINUOUS_CONTEXT_FIELDS: tuple[str, ...] = (
    "mover_rating",
    "opponent_rating",
    "rating_difference",
    "ply",
)
UNKNOWN_CATEGORY_TOKEN = "__UNK__"


@dataclass(frozen=True)
class ContinuousStat:
    mean: float
    std: float
    observed_count: int
    missing_count: int


@dataclass(frozen=True)
class NeuralPreprocessor:
    continuous_stats: dict[str, ContinuousStat]
    rating_band_to_index: dict[str, int]
    time_control_to_index: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "continuous_stats": {
                field: {
                    "mean": stat.mean,
                    "std": stat.std,
                    "observed_count": stat.observed_count,
                    "missing_count": stat.missing_count,
                }
                for field, stat in sorted(self.continuous_stats.items())
            },
            "rating_band_to_index": dict(sorted(self.rating_band_to_index.items())),
            "time_control_to_index": dict(sorted(self.time_control_to_index.items())),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> NeuralPreprocessor:
        raw_stats = payload.get("continuous_stats")
        if not isinstance(raw_stats, dict):
            raise ValueError("continuous_stats must be an object")

        stats: dict[str, ContinuousStat] = {}
        for field_name, field_payload in raw_stats.items():
            if not isinstance(field_payload, dict):
                raise ValueError(f"continuous_stats.{field_name} must be an object")
            stats[str(field_name)] = ContinuousStat(
                mean=float(field_payload["mean"]),
                std=float(field_payload["std"]),
                observed_count=int(field_payload["observed_count"]),
                missing_count=int(field_payload["missing_count"]),
            )

        rating_band = payload.get("rating_band_to_index")
        if not isinstance(rating_band, dict):
            raise ValueError("rating_band_to_index must be an object")

        time_control = payload.get("time_control_to_index")
        if not isinstance(time_control, dict):
            raise ValueError("time_control_to_index must be an object")

        return NeuralPreprocessor(
            continuous_stats=stats,
            rating_band_to_index={str(key): int(value) for key, value in rating_band.items()},
            time_control_to_index={str(key): int(value) for key, value in time_control.items()},
        )


@dataclass(frozen=True)
class EncodedExample:
    split: str
    game_id: str
    ply: int
    position_id: str
    board_tensor: npt.NDArray[np.float32]
    legal_mask: npt.NDArray[np.bool_]
    policy_target_action_index: int
    value_target_index: int
    continuous_context: npt.NDArray[np.float32]
    rating_band_index: int
    time_control_index: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SelectionFingerprints:
    split_key_hashes: dict[str, str]
    train_population_key_hash: str


@dataclass(frozen=True)
class CompatibilityReport:
    compatible: bool
    reason: str


def _opening_family(eco: str | None) -> str:
    if eco is None:
        return "unknown"
    text = eco.strip().upper()
    if not text:
        return "unknown"
    return text[0]


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_numeric_field(example: PositionExample, field_name: str) -> float | None:
    if field_name == "ply":
        return float(example.ply)
    return _to_optional_float(getattr(example, field_name))


def _build_category_map(values: list[str]) -> dict[str, int]:
    categories = sorted({value for value in values if value})
    index_map = {UNKNOWN_CATEGORY_TOKEN: 0}
    for category in categories:
        if category == UNKNOWN_CATEGORY_TOKEN:
            continue
        index_map[category] = len(index_map)
    return index_map


def fit_preprocessor(train_examples: list[PositionExample]) -> NeuralPreprocessor:
    if not train_examples:
        raise RuntimeError("Cannot fit preprocessor on empty training set")

    continuous_stats: dict[str, ContinuousStat] = {}
    for field_name in CONTINUOUS_CONTEXT_FIELDS:
        values: list[float] = []
        missing = 0
        for example in train_examples:
            value = _extract_numeric_field(example, field_name)
            if value is None:
                missing += 1
            else:
                values.append(value)

        if values:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))
            if std <= 0.0:
                std = 1.0
        else:
            mean = 0.0
            std = 1.0

        continuous_stats[field_name] = ContinuousStat(
            mean=mean,
            std=std,
            observed_count=len(values),
            missing_count=missing,
        )

    rating_band_to_index = _build_category_map(
        [
            (example.mover_rating_band if example.mover_rating_band else "unknown")
            for example in train_examples
        ]
    )
    time_control_to_index = _build_category_map(
        [
            (example.time_control_category if example.time_control_category else "unknown")
            for example in train_examples
        ]
    )

    return NeuralPreprocessor(
        continuous_stats=continuous_stats,
        rating_band_to_index=rating_band_to_index,
        time_control_to_index=time_control_to_index,
    )


def value_target_to_index(value_target_wdl: str) -> int:
    mapping = {label: idx for idx, label in enumerate(WDL_CLASS_ORDER)}
    if value_target_wdl not in mapping:
        raise RuntimeError(f"Unexpected value_target_wdl label: {value_target_wdl!r}")
    return mapping[value_target_wdl]


def _continuous_context_vector(
    example: PositionExample,
    preprocessor: NeuralPreprocessor,
) -> npt.NDArray[np.float32]:
    values: list[float] = []

    for field_name in CONTINUOUS_CONTEXT_FIELDS:
        stat = preprocessor.continuous_stats[field_name]
        raw_value = _extract_numeric_field(example, field_name)
        if raw_value is None:
            values.append(0.0)
            values.append(1.0)
        else:
            values.append((raw_value - stat.mean) / stat.std)
            values.append(0.0)

    return np.asarray(values, dtype=np.float32)


def encode_examples(
    examples: list[PositionExample],
    *,
    split: str,
    preprocessor: NeuralPreprocessor,
) -> list[EncodedExample]:
    encoded: list[EncodedExample] = []

    for example in examples:
        board_tensor = encode_fen_18x8x8(example.pre_move_fen, dtype=np.float32)
        board = chess.Board(example.pre_move_fen)
        legal_mask = legal_move_mask(board)

        if legal_mask.shape != (ACTION_SPACE_SIZE,):
            raise RuntimeError("legal_move_mask returned unexpected shape")

        policy_target = int(example.policy_target_action_index)
        if policy_target < 0 or policy_target >= ACTION_SPACE_SIZE:
            raise RuntimeError(
                "Policy target action index is out of range: "
                f"split={split}, game_id={example.game_id}, ply={example.ply}, "
                f"target={policy_target}"
            )
        if not bool(legal_mask[policy_target]):
            raise RuntimeError(
                "Policy target action index is not legal in pre-move position: "
                f"split={split}, game_id={example.game_id}, ply={example.ply}, "
                f"target={policy_target}"
            )
        if not bool(np.any(legal_mask)):
            raise RuntimeError(
                "No legal actions available for selected row: "
                f"split={split}, game_id={example.game_id}, ply={example.ply}"
            )

        rating_token = example.mover_rating_band if example.mover_rating_band else "unknown"
        time_control_token = (
            example.time_control_category if example.time_control_category else "unknown"
        )

        encoded.append(
            EncodedExample(
                split=split,
                game_id=example.game_id,
                ply=example.ply,
                position_id=example.position_id,
                board_tensor=board_tensor,
                legal_mask=legal_mask,
                policy_target_action_index=policy_target,
                value_target_index=value_target_to_index(example.value_target_wdl),
                continuous_context=_continuous_context_vector(example, preprocessor),
                rating_band_index=preprocessor.rating_band_to_index.get(
                    rating_token,
                    preprocessor.rating_band_to_index[UNKNOWN_CATEGORY_TOKEN],
                ),
                time_control_index=preprocessor.time_control_to_index.get(
                    time_control_token,
                    preprocessor.time_control_to_index[UNKNOWN_CATEGORY_TOKEN],
                ),
                metadata={
                    "split": split,
                    "game_id": example.game_id,
                    "ply": example.ply,
                    "position_id": example.position_id,
                    "source_month": example.source_month,
                    "is_player_holdout_game": example.is_player_holdout_game,
                    "novel_position_test": example.novel_position_test,
                    "mover_rating_band": rating_token,
                    "time_control_category": time_control_token,
                    "opening_family": _opening_family(example.eco),
                },
            )
        )

    return encoded


def selected_key_hash(examples: list[PositionExample]) -> str:
    payload = [[item.game_id, int(item.ply)] for item in examples]
    return sha256_text(canonical_json(payload))


def selection_fingerprints(
    *,
    train_selected: list[PositionExample],
    validation_selected: list[PositionExample],
    test_selected: list[PositionExample],
    train_population_used: list[PositionExample],
) -> SelectionFingerprints:
    return SelectionFingerprints(
        split_key_hashes={
            "train": selected_key_hash(train_selected),
            "validation": selected_key_hash(validation_selected),
            "test": selected_key_hash(test_selected),
        },
        train_population_key_hash=selected_key_hash(train_population_used),
    )


def compare_selection_fingerprints(
    *,
    expected_split_key_hashes: dict[str, str],
    expected_train_population_key_hash: str,
    actual: SelectionFingerprints,
) -> CompatibilityReport:
    for split_name in ("train", "validation", "test"):
        expected = expected_split_key_hashes.get(split_name)
        actual_hash = actual.split_key_hashes.get(split_name)
        if expected is None or actual_hash is None:
            return CompatibilityReport(
                compatible=False,
                reason=(
                    "Missing split key hash for compatibility check: "
                    f"{split_name}"
                ),
            )
        if expected != actual_hash:
            return CompatibilityReport(
                compatible=False,
                reason=(
                    "Selection key mismatch for split "
                    f"{split_name}: expected={expected} actual={actual_hash}"
                ),
            )

    if expected_train_population_key_hash != actual.train_population_key_hash:
        return CompatibilityReport(
            compatible=False,
            reason=(
                "Training population key mismatch: "
                f"expected={expected_train_population_key_hash} "
                f"actual={actual.train_population_key_hash}"
            ),
        )

    return CompatibilityReport(compatible=True, reason="selection_fingerprints_match")
