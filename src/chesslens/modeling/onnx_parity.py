"""CPU FP32 inference contract and tolerance-aware chess behavior comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import chess
import numpy as np

from chesslens.features.action_encoding import (
    ACTION_SPACE_SIZE,
    decode_index_to_move,
    encode_move_to_index,
    legal_move_mask,
)
from chesslens.features.position_encoding import BOARD_SHAPE
from chesslens.modeling.neural_data import CONTINUOUS_CONTEXT_FIELDS, NeuralPreprocessor

if TYPE_CHECKING:
    import onnxruntime as ort

RTOL = 1e-4
ATOL = 1e-5
INPUT_SPEC: dict[str, tuple[str, tuple[int, ...]]] = {
    "board_tensor": ("float32", BOARD_SHAPE),
    "continuous_context": ("float32", (2 * len(CONTINUOUS_CONTEXT_FIELDS),)),
    "rating_band_index": ("int64", ()),
    "time_control_index": ("int64", ()),
}
OUTPUT_SPEC: dict[str, tuple[str, tuple[int, ...]]] = {
    "policy_logits": ("float32", (ACTION_SPACE_SIZE,)),
    "value_logits": ("float32", (3,)),
}


def io_contract() -> dict[str, Any]:
    return {
        family: [
            {"name": name, "dtype": dtype, "shape": ["batch", *shape]}
            for name, (dtype, shape) in spec.items()
        ]
        for family, spec in (("inputs", INPUT_SPEC), ("outputs", OUTPUT_SPEC))
    }


def validate_session(session: Any) -> None:
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("ONNX session must exclusively use CPUExecutionProvider")
    for nodes, spec in ((session.get_inputs(), INPUT_SPEC), (session.get_outputs(), OUTPUT_SPEC)):
        if [node.name for node in nodes] != list(spec):
            raise ValueError("ONNX session I/O name/order mismatch")
        for node in nodes:
            dtype, shape = spec[node.name]
            expected_type = "tensor(float)" if dtype == "float32" else "tensor(int64)"
            if node.type != expected_type or node.shape != ["batch", *shape]:
                raise ValueError(f"ONNX session dtype/shape mismatch for {node.name}: {node}")


def create_cpu_session(path: Path, threads: int) -> ort.InferenceSession:
    import onnxruntime as ort

    if threads < 1:
        raise ValueError("Inference threads must be positive")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    session.disable_fallback()
    validate_session(session)
    return session


def validate_inputs(inputs: dict[str, np.ndarray], preprocessor: NeuralPreprocessor) -> None:
    if set(inputs) != set(INPUT_SPEC):
        raise ValueError("Input names must exactly match the inference contract")
    batch = inputs["board_tensor"].shape[0] if inputs["board_tensor"].ndim else 0
    if batch <= 0:
        raise ValueError("Inference batch must be nonempty")
    for name, (dtype, shape) in INPUT_SPEC.items():
        array = inputs[name]
        if array.dtype != np.dtype(dtype) or array.shape != (batch, *shape):
            raise ValueError(f"Invalid dtype/shape for {name}: {array.dtype}, {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite input: {name}")
    for name, mapping in (
        ("rating_band_index", preprocessor.rating_band_to_index),
        ("time_control_index", preprocessor.time_control_to_index),
    ):
        if np.any(inputs[name] < 0) or np.any(inputs[name] >= len(mapping)):
            raise ValueError(f"Out-of-vocabulary embedding index: {name}")


def softmax_wdl(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if logits.ndim != 2 or logits.shape[1] != 3 or not np.isfinite(logits).all():
        raise ValueError("WDL logits must be finite [B,3]")
    scores = logits.astype(np.float64) / temperature
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return np.asarray(probabilities / probabilities.sum(axis=1, keepdims=True), dtype=np.float64)


def compare_arrays(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    if reference.shape != actual.shape:
        return {
            "passed": False,
            "reason": "shape mismatch",
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
        }
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        return {"passed": False, "reason": "nonfinite values"}
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    bounds = ATOL + RTOL * np.abs(reference.astype(np.float64))
    failed = error > bounds
    return {
        "passed": bool(np.allclose(actual, reference, rtol=RTOL, atol=ATOL)),
        "shape": list(reference.shape),
        "max_absolute_error": float(error.max()),
        "p99_absolute_error": float(np.percentile(error, 99)),
        "over_tolerance_count": int(failed.sum()),
        "failure_examples": [
            {
                "row": int(row),
                "column": int(col),
                "reference": float(reference[row, col]),
                "actual": float(actual[row, col]),
                "absolute_error": float(error[row, col]),
                "tolerance": float(bounds[row, col]),
            }
            for row, col in np.argwhere(failed)[:20]
        ],
    }


def rank_legal(logits: np.ndarray, mask: np.ndarray, fen: str) -> list[int]:
    return rank_legal_actions(logits, mask, fen)[:5]


def rank_legal_actions(logits: np.ndarray, mask: np.ndarray, fen: str) -> list[int]:
    board = chess.Board(fen)
    if mask.dtype != np.bool_ or mask.shape != (ACTION_SPACE_SIZE,):
        raise ValueError("Legal mask dtype/shape mismatch")
    if not np.array_equal(mask, legal_move_mask(board)):
        raise ValueError("Legal mask does not match python-chess")
    if logits.shape != (ACTION_SPACE_SIZE,) or not np.isfinite(logits).all():
        raise ValueError("Policy logits must be finite [4672]")
    ranked = sorted(np.flatnonzero(mask).tolist(), key=lambda idx: (-float(logits[idx]), idx))
    for idx in ranked[:5]:
        move = decode_index_to_move(idx, board, require_legal=True)
        if encode_move_to_index(move, board) != idx:
            raise ValueError("Action encoder/decoder round-trip mismatch")
    return ranked


def _gap(
    reference: np.ndarray,
    actual: np.ndarray,
    left: int,
    right: int,
) -> dict[str, Any]:
    ref_gap = float(abs(float(reference[left]) - float(reference[right])))
    bound = float(2 * ATOL + RTOL * (abs(float(reference[left])) + abs(float(reference[right]))))
    return {
        "left": left,
        "right": right,
        "reference_gap": ref_gap,
        "actual_gap": float(abs(float(actual[left]) - float(actual[right]))),
        "summed_tolerance_bound": bound,
        "summed_observed_error": float(
            abs(float(actual[left]) - float(reference[left]))
            + abs(float(actual[right]) - float(reference[right]))
        ),
        "within_bound": ref_gap <= bound,
    }


def compare_behavior(
    reference: tuple[np.ndarray, np.ndarray],
    actual: tuple[np.ndarray, np.ndarray],
    masks: np.ndarray,
    fens: list[str],
    keys: list[tuple[str, int]],
    *,
    numerical_passed: bool,
) -> dict[str, Any]:
    top1_matches = top5_matches = wdl_matches = near_ties = unexplained = 0
    mismatches: list[dict[str, Any]] = []
    for row, (fen, key) in enumerate(zip(fens, keys, strict=True)):
        ref_top = rank_legal(reference[0][row], masks[row], fen)
        ort_top = rank_legal(actual[0][row], masks[row], fen)
        ref_wdl, ort_wdl = int(reference[1][row].argmax()), int(actual[1][row].argmax())
        same_top1 = ref_top[:1] == ort_top[:1]
        same_top5 = ref_top == ort_top
        top1_matches += int(same_top1)
        top5_matches += int(same_top5)
        wdl_matches += int(ref_wdl == ort_wdl)
        if same_top5 and ref_wdl == ort_wdl:
            continue
        gaps = [
            _gap(reference[0][row], actual[0][row], left, right)
            for left, right in zip(ref_top, ort_top, strict=True)
            if left != right
        ]
        if ref_wdl != ort_wdl:
            gaps.append(_gap(reference[1][row], actual[1][row], ref_wdl, ort_wdl))
        near = numerical_passed and all(gap["within_bound"] for gap in gaps)
        near_ties += int(near)
        unexplained += int(not near)
        board = chess.Board(fen)
        mismatches.append(
            {
                "row": row,
                "game_id": key[0],
                "ply": key[1],
                "reference_top5": ref_top,
                "actual_top5": ort_top,
                "reference_moves": [decode_index_to_move(idx, board).uci() for idx in ref_top],
                "actual_moves": [decode_index_to_move(idx, board).uci() for idx in ort_top],
                "reference_wdl_argmax": ref_wdl,
                "actual_wdl_argmax": ort_wdl,
                "top1_match": same_top1,
                "gaps": gaps,
                "numerical_near_tie": near,
            }
        )
    return {
        "passed": unexplained == 0,
        "rows": len(keys),
        "legal_top1_matches": top1_matches,
        "ordered_legal_top5_matches": top5_matches,
        "wdl_argmax_matches": wdl_matches,
        "near_tie_cases": near_ties,
        "unexplained_mismatch_cases": unexplained,
        "illegal_output_count": 0,
        "mismatches": mismatches,
    }
