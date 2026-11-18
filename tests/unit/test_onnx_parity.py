from __future__ import annotations

from types import SimpleNamespace

import chess
import numpy as np
import pytest

from chesslens.features.action_encoding import ACTION_SPACE_SIZE, legal_move_mask
from chesslens.modeling.neural_data import ContinuousStat, NeuralPreprocessor
from chesslens.modeling.onnx_parity import (
    ATOL,
    compare_arrays,
    compare_behavior,
    rank_legal,
    softmax_wdl,
    validate_inputs,
    validate_session,
)


def test_numeric_comparison_uses_reference_tolerance_and_reports_errors() -> None:
    ref = np.asarray([[0.0, 1.0, 100.0]], dtype=np.float32)
    assert compare_arrays(ref, ref.copy())["passed"]
    actual = ref.copy()
    actual[0, 0] = ATOL * 2
    result = compare_arrays(ref, actual)
    assert not result["passed"]
    assert result["over_tolerance_count"] == 1
    assert result["failure_examples"][0]["column"] == 0
    assert not compare_arrays(ref, ref[:, :2])["passed"]
    assert not compare_arrays(ref, np.full_like(ref, np.nan))["passed"]


def test_tie_break_and_explained_vs_unexplained_behavior() -> None:
    board = chess.Board()
    mask = legal_move_mask(board)
    legal = np.flatnonzero(mask)
    policy = np.zeros((1, ACTION_SPACE_SIZE), dtype=np.float32)
    value = np.asarray([[1.0, 0.0, -1.0]], dtype=np.float32)
    assert rank_legal(policy[0], mask, board.fen()) == legal[:5].tolist()
    changed = policy.copy()
    changed[0, legal[1]] = ATOL / 2
    result = compare_behavior(
        (policy, value),
        (changed, value),
        mask[None],
        [board.fen()],
        [("game", 0)],
        numerical_passed=True,
    )
    assert result["passed"]
    assert result["near_tie_cases"] == 1
    assert result["legal_top1_matches"] == 0
    assert result["mismatches"][0]["gaps"][0]["within_bound"]

    reference = policy.copy()
    reference[0, legal[0]] = 1.0
    changed[0, legal[1]] = 2.0
    result = compare_behavior(
        (reference, value),
        (changed, value),
        mask[None],
        [board.fen()],
        [("game", 0)],
        numerical_passed=False,
    )
    assert not result["passed"]
    assert result["unexplained_mismatch_cases"] == 1
    corrupted = mask.copy()
    corrupted[0] = not corrupted[0]
    with pytest.raises(ValueError, match="python-chess"):
        rank_legal(policy[0], corrupted, board.fen())


def test_wdl_softmax_and_temperature_contract() -> None:
    logits = np.asarray([[2000.0, 1999.0, -2000.0]], dtype=np.float32)
    raw = softmax_wdl(logits)
    assert np.isfinite(raw).all()
    np.testing.assert_allclose(raw.sum(1), 1.0)
    assert raw.argmax(1).tolist() == softmax_wdl(logits, 1.042827698).argmax(1).tolist()
    with pytest.raises(ValueError, match="positive"):
        softmax_wdl(logits, 0.0)


def test_session_contract_rejects_fixed_batch() -> None:
    from chesslens.modeling.onnx_parity import INPUT_SPEC, OUTPUT_SPEC

    nodes = {
        family: [
            SimpleNamespace(
                name=name,
                type="tensor(float)" if dtype == "float32" else "tensor(int64)",
                shape=["batch", *shape],
            )
            for name, (dtype, shape) in spec.items()
        ]
        for family, spec in (("inputs", INPUT_SPEC), ("outputs", OUTPUT_SPEC))
    }
    session = SimpleNamespace(
        get_providers=lambda: ["CPUExecutionProvider"],
        get_inputs=lambda: nodes["inputs"],
        get_outputs=lambda: nodes["outputs"],
    )
    validate_session(session)
    nodes["inputs"][0].shape[0] = 1
    with pytest.raises(ValueError, match="dtype/shape"):
        validate_session(session)


@pytest.mark.parametrize("failure", ["dtype", "shape", "nan", "category", "names"])
def test_input_contract_rejects_bad_inputs(failure: str) -> None:
    preprocessor = NeuralPreprocessor(
        continuous_stats={"ply": ContinuousStat(0.0, 1.0, 1, 0)},
        rating_band_to_index={"__UNK__": 0},
        time_control_to_index={"__UNK__": 0},
    )
    inputs: dict[str, np.ndarray] = {
        "board_tensor": np.zeros((1, 18, 8, 8), dtype=np.float32),
        "continuous_context": np.zeros((1, 8), dtype=np.float32),
        "rating_band_index": np.zeros(1, dtype=np.int64),
        "time_control_index": np.zeros(1, dtype=np.int64),
    }
    validate_inputs(inputs, preprocessor)
    if failure == "dtype":
        inputs["board_tensor"] = inputs["board_tensor"].astype(np.float64)
    elif failure == "shape":
        inputs["continuous_context"] = np.zeros((2, 8), dtype=np.float32)
    elif failure == "nan":
        inputs["continuous_context"][0, 0] = np.nan
    elif failure == "category":
        inputs["rating_band_index"][0] = 1
    else:
        inputs["unexpected"] = np.zeros(1)
    with pytest.raises(ValueError):
        validate_inputs(inputs, preprocessor)
