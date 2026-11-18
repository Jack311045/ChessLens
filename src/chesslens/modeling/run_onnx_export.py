"""Export frozen CPU FP32 policy/value checkpoints and verify chess inference parity."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import chess
import numpy as np
import onnx
import onnxruntime as ort
import psutil
import torch

from chesslens.features import action_encoding, position_encoding
from chesslens.modeling.neural_checkpoint import (
    _require_equal,
    load_frozen_model,
    validate_checkpoint_config,
)
from chesslens.modeling.neural_config import NeuralConfig
from chesslens.modeling.neural_data import (
    CONTINUOUS_CONTEXT_FIELDS,
    WDL_CLASS_ORDER,
    NeuralPreprocessor,
    encode_examples,
    selected_key_hash,
)
from chesslens.modeling.neural_network import PolicyValueResidualCNN
from chesslens.modeling.neural_training import _batch_tensors, _file_sha256, _SingleWriterLock
from chesslens.modeling.onnx_parity import (
    ATOL,
    INPUT_SPEC,
    OUTPUT_SPEC,
    RTOL,
    compare_arrays,
    compare_behavior,
    create_cpu_session,
    io_contract,
    softmax_wdl,
    validate_inputs,
)
from chesslens.modeling.run_baselines import (
    PositionExample,
    _load_position_examples,
    _required_partition_paths,
)
from chesslens.modeling.validation import canonical_json, read_json_object, sha256_text

OPSET = 17
SOURCE_FILES = {
    "source_neural_manifest.json": "_manifest.json",
    "model_state_best.pt": "model_state_best.pt",
    "feature_preprocessing.json": "feature_preprocessing.json",
    "effective_config.json": "effective_config.json",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _integrity(path: Path) -> dict[str, Any]:
    return {"sha256": _file_sha256(path), "size_bytes": path.stat().st_size}


def _session(path: Path, threads: int) -> ort.InferenceSession:
    return create_cpu_session(path, threads)


def load_bundle(
    root: Path,
    *,
    threads: int = 2,
    allow_fixture: bool = False,
) -> tuple[ort.InferenceSession, NeuralPreprocessor, dict[str, Any]]:
    manifest = read_json_object(root / "_manifest.json")
    if threads < 1:
        raise ValueError("Bundle inference threads must be positive")
    if manifest["status"] != "complete" or not (root / "_SUCCESS").is_file():
        raise RuntimeError("ONNX bundle is not complete")
    required = {
        *SOURCE_FILES,
        "model.onnx",
        "metadata.json",
        "parity.json",
        "parity_arrays.npz",
        "report.md",
    }
    if not required.issubset(manifest["artifact_integrity"]):
        raise RuntimeError("ONNX bundle missing required artifact integrity entries")
    for name, record in manifest["artifact_integrity"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid bundle-relative artifact path: {name}")
        _require_equal(_integrity(root / relative), record, f"bundle {name} integrity")
    metadata = read_json_object(root / "metadata.json")
    if not metadata["ready_for_handoff"] and not allow_fixture:
        raise RuntimeError("Fixture-only bundle is not a validated real-data handoff")
    _require_equal(metadata["io"], io_contract(), "bundle I/O contract")
    preprocessor = NeuralPreprocessor.from_dict(
        read_json_object(root / "feature_preprocessing.json")
    )
    return _session(root / "model.onnx", threads), preprocessor, metadata


def fixture_positions(preprocessor: NeuralPreprocessor) -> list[PositionExample]:
    black_start = chess.Board()
    black_start.push_uci("e2e4")
    fens = [
        chess.STARTING_FEN,
        black_start.fen(),
        "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
        "r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "1k6/P7/8/8/8/8/8/7K w - - 0 1",
        "4k3/8/8/8/8/8/7p/6NK b - - 0 1",
        "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
    ]
    examples = []
    for idx, fen in enumerate(fens):
        board = chess.Board(fen)
        missing = idx % 2 == 1
        examples.append(
            PositionExample(
                split="validation",
                game_id=f"onnx_fixture_{idx}",
                ply=board.ply(),
                source_month=None,
                position_id=position_encoding.position_id_from_fen(fen),
                pre_move_fen=fen,
                normalized_pre_move_fen=position_encoding.normalize_fen(fen),
                side_to_move="w" if board.turn else "b",
                time_control_raw=None if missing else "300+0",
                time_control_category="unseen_time_control"
                if missing
                else next(
                    (key for key in preprocessor.time_control_to_index if key != "__UNK__"),
                    "__UNK__",
                ),
                eco=None,
                opening=None,
                mover_rating=None if missing else 1500,
                opponent_rating=None if missing else 1600,
                rating_difference=None if missing else -100,
                mover_rating_band="unseen_rating_band"
                if missing
                else next(
                    (key for key in preprocessor.rating_band_to_index if key != "__UNK__"),
                    "__UNK__",
                ),
                player_disjoint_training_eligible=False,
                is_player_holdout_game=False,
                policy_target_action_index=action_encoding.encode_move_to_index(
                    next(iter(board.legal_moves)),
                    board,
                ),
                value_target_wdl="draw",
                novel_position_test=False,
            )
        )
    return examples


def _validation_positions(
    path: Path,
    manifest: dict[str, Any],
    config: NeuralConfig,
    max_positions: int,
    seed: int,
) -> tuple[list[PositionExample], dict[str, Any]]:
    modeling = read_json_object(path)
    identity = manifest["identity_payload"]
    _require_equal(modeling["status"], "complete", "modeling status")
    if not (path.parent / "_SUCCESS").is_file():
        raise RuntimeError("Modeling dataset missing _SUCCESS")
    _require_equal(modeling["modeling_dataset_id"], identity["modeling_dataset_id"], "dataset ID")
    _require_equal(
        sha256_text(canonical_json(modeling)),
        identity["modeling_manifest_sha256"],
        "modeling manifest SHA256",
    )
    paths = _required_partition_paths(
        modeling,
        dataset_name="policy_examples",
        partition="validation",
        dataset_path=path.parent,
    )
    examples = _load_position_examples(
        paths,
        split="validation",
        limit=config.selection.max_validation_positions,
        seed=config.runtime.seed,
        novel_position_ids=set(),
    )
    fingerprint = selected_key_hash(examples)
    _require_equal(
        fingerprint,
        manifest["selection_fingerprints"]["split_key_hashes"]["validation"],
        "validation selection fingerprint",
    )
    examples = sorted(
        examples,
        key=lambda row: (
            sha256_text(canonical_json(["onnx_parity_v1", seed, row.game_id, row.ply])),
            row.game_id,
            row.ply,
        ),
    )[:max_positions]
    return examples, {
        "kind": "validation",
        "modeling_manifest": _integrity(path),
        "validation_selection_fingerprint": fingerprint,
        "validation_files": [{"name": p.name, **_integrity(p)} for p in paths],
        "sampling": "sha256_onnx_parity_v1_seed_game_ply_v1",
        "test_split_access": False,
    }


def _encode(
    examples: list[PositionExample],
    preprocessor: NeuralPreprocessor,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    encoded = encode_examples(examples, split="validation", preprocessor=preprocessor)
    tensors = _batch_tensors(encoded, np.arange(len(encoded)), device=torch.device("cpu"))
    inputs = {
        name: np.ascontiguousarray(tensors[source].numpy())
        for name, source in zip(
            INPUT_SPEC, ("board", "continuous", "rating_band", "time_control"), strict=True
        )
    }
    validate_inputs(inputs, preprocessor)
    return inputs, np.stack([row.legal_mask for row in encoded])


def _forward(
    model: PolicyValueResidualCNN,
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
    batch_size: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], list[int]]:
    pytorch: list[list[np.ndarray]] = [[], []]
    runtime: list[list[np.ndarray]] = [[], []]
    actual_sizes = []
    with torch.inference_mode():
        for start in range(0, len(inputs["board_tensor"]), batch_size):
            batch = {name: value[start : start + batch_size] for name, value in inputs.items()}
            actual_sizes.append(len(batch["board_tensor"]))
            outputs = model(*(torch.from_numpy(value) for value in batch.values()))
            ort_outputs = session.run(list(OUTPUT_SPEC), batch)
            for head, (tensor, array) in enumerate(zip(outputs, ort_outputs, strict=True)):
                ref = tensor.numpy().copy()
                if ref.dtype != np.float32 or array.dtype != np.float32:
                    raise ValueError("Both raw-logit heads must remain FP32")
                pytorch[head].append(ref)
                runtime[head].append(array)
    return (
        (np.concatenate(pytorch[0]), np.concatenate(pytorch[1])),
        (np.concatenate(runtime[0]), np.concatenate(runtime[1])),
        actual_sizes,
    )


def evaluate_parity(
    *,
    model: PolicyValueResidualCNN,
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
    masks: np.ndarray,
    examples: list[PositionExample],
    batch_sizes: tuple[int, ...],
    exploratory_temperature: float | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    comparisons: list[dict[str, Any]] = []
    arrays = dict(inputs)
    arrays.update(
        {
            "legal_mask": masks,
            "game_id": np.asarray([row.game_id for row in examples]),
            "ply": np.asarray([row.ply for row in examples], dtype=np.int64),
            "fen": np.asarray([row.pre_move_fen for row in examples]),
        }
    )
    single: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None
    executions = []
    for size in batch_sizes:
        ref, actual, actual_sizes = _forward(model, session, inputs, size)
        executions.append({"requested_batch_size": size, "actual_batch_sizes": actual_sizes})
        if size == 1:
            single = ref, actual
        pairs = [(f"pytorch_vs_ort_batch_{size}", ref, actual)]
        if size != 1 and single is not None:
            pairs += [
                (f"pytorch_batch_{size}_vs_single", single[0], ref),
                (f"ort_batch_{size}_vs_single", single[1], actual),
            ]
        for name, left, right in pairs:
            numeric = {
                "policy_logits": compare_arrays(left[0], right[0]),
                "value_logits": compare_arrays(left[1], right[1]),
            }
            valid_values = all(
                output.shape == (len(examples), 3) and np.isfinite(output).all()
                for output in (left[1], right[1])
            )
            numeric["wdl_probabilities_t1"] = (
                compare_arrays(softmax_wdl(left[1]), softmax_wdl(right[1]))
                if valid_values
                else {"passed": False, "reason": "invalid WDL logits"}
            )
            if exploratory_temperature is not None and valid_values:
                numeric["exploratory_wdl_probabilities"] = compare_arrays(
                    softmax_wdl(left[1], exploratory_temperature),
                    softmax_wdl(right[1], exploratory_temperature),
                )
            numerical_passed = all(item["passed"] for item in numeric.values())
            valid_policy = all(
                output.shape == (len(examples), action_encoding.ACTION_SPACE_SIZE)
                and np.isfinite(output).all()
                for output in (left[0], right[0])
            )
            if valid_policy and valid_values:
                behavior = compare_behavior(
                    left,
                    right,
                    masks,
                    [row.pre_move_fen for row in examples],
                    [(row.game_id, row.ply) for row in examples],
                    numerical_passed=numerical_passed,
                )
            else:
                behavior = {
                    "passed": False,
                    "skipped": "invalid logits retained in parity_arrays.npz",
                    "legal_top1_matches": 0,
                    "ordered_legal_top5_matches": 0,
                    "wdl_argmax_matches": 0,
                    "near_tie_cases": 0,
                    "unexplained_mismatch_cases": len(examples),
                }
            for metric in numeric.values():
                for failure in metric.get("failure_examples", []):
                    row = examples[failure["row"]]
                    failure.update({"game_id": row.game_id, "ply": row.ply})
            comparisons.append(
                {
                    "name": name,
                    "numerical": numeric,
                    "behavior": behavior,
                    "passed": numerical_passed and behavior["passed"],
                }
            )
        for runtime, outputs in (("pytorch", ref), ("ort", actual)):
            for head, output_array in zip(OUTPUT_SPEC, outputs, strict=True):
                arrays[f"{runtime}_{head}_batch_{size}"] = output_array
    return {
        "passed": all(item["passed"] for item in comparisons),
        "rtol": RTOL,
        "atol": ATOL,
        "rows": len(examples),
        "key_fingerprint": selected_key_hash(examples),
        "executions": executions,
        "comparisons": comparisons,
        "tie_break": "descending legal logit, then ascending action index (baseline convention)",
        "near_tie_policy": "All numeric checks must pass; every affected gap <= summed tolerances",
    }, arrays


def benchmark_forward(
    model: PolicyValueResidualCNN,
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
    *,
    small_batch: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    results = []
    with torch.inference_mode():
        for size in (1, small_batch):
            batch = {name: value[:size] for name, value in inputs.items()}
            tensors = tuple(torch.from_numpy(value) for value in batch.values())
            for _ in range(warmup):
                model(*tensors)
                session.run(list(OUTPUT_SPEC), batch)
            measurements: dict[str, list[float]] = {"pytorch_eager": [], "onnxruntime": []}
            for idx in range(iterations):
                order = (
                    ("pytorch_eager", "onnxruntime")
                    if idx % 2 == 0
                    else (
                        "onnxruntime",
                        "pytorch_eager",
                    )
                )
                for runtime in order:
                    started = time.perf_counter_ns()
                    if runtime == "pytorch_eager":
                        model(*tensors)
                    else:
                        session.run(list(OUTPUT_SPEC), batch)
                    measurements[runtime].append((time.perf_counter_ns() - started) / 1e6)
            for runtime, values in measurements.items():
                results.append(
                    {
                        "runtime": runtime,
                        "batch_size": size,
                        "p50_ms": float(np.percentile(values, 50)),
                        "p95_ms": float(np.percentile(values, 95)),
                        "iterations": iterations,
                        "warmup": warmup,
                    }
                )
    return {
        "scope": "forward-only; excludes load/export/encoding/conversion/postprocessing",
        "order": "alternating runtimes per measured iteration; output allocation included",
        "results": results,
    }


def _markdown(report: dict[str, Any], metadata: dict[str, Any]) -> str:
    lines = [
        "# CPU FP32 ONNX export and parity",
        "",
        f"- Experiment: `{metadata['experiment_id']}`",
        f"- Data: `{metadata['modeling_dataset_id']}`",
        f"- Parity passed: **{report['parity']['passed']}**",
        f"- Ready for real-data handoff: **{metadata['ready_for_handoff']}**",
        f"- Positions: {report['parity']['rows']}; rtol={RTOL}, atol={ATOL}",
        "- CPUExecutionProvider; dynamic batch; raw policy/WDL logits; production WDL T=1.",
        "- Masking, action decoding/ranking, preprocessing and softmax remain outside the graph.",
        "",
        "| Comparison | Policy max abs | WDL max abs | Top-1 / top-5 / WDL matches | Near ties |",
        "|---|---:|---:|---|---:|",
    ]
    for item in report["parity"]["comparisons"]:
        behavior = item["behavior"]
        counts = " / ".join(
            str(behavior[k])
            for k in (
                "legal_top1_matches",
                "ordered_legal_top5_matches",
                "wdl_argmax_matches",
            )
        )
        lines.append(
            f"| {item['name']} | {item['numerical']['policy_logits'].get('max_absolute_error')} "
            f"| {item['numerical']['value_logits'].get('max_absolute_error')} "
            f"| {counts} | {behavior['near_tie_cases']} |"
        )
    lines += [
        "",
        "## Forward-only CPU benchmark",
        "",
        "| Runtime | Batch | p50 ms | p95 ms |",
        "|---|---:|---:|---:|",
    ]
    for row in report.get("benchmark", {}).get("results", []):
        lines.append(
            f"| {row['runtime']} | {row['batch_size']} "
            f"| {row['p50_ms']:.4f} | {row['p95_ms']:.4f} |"
        )
    lines += [
        "",
        "All numerical/behavior failures and near-tie cases are in parity.json. "
        "Encoded inputs, keys, FENs, masks and both heads for every batching strategy are "
        "in parity_arrays.npz (allow_pickle=False). This is runtime equivalence, not a "
        "generalization or calibration acceptance claim. Fixture dummy labels are never scored.",
        "",
        "metadata.json defines I/O, encoding versions, continuous/missing-flag order, "
        "category mappings, provenance and relative artifact paths. The original source manifest "
        "and effective config are archival; historical input paths are not runtime dependencies.",
        "",
    ]
    return "\n".join(lines)


def run_onnx_export(
    *,
    neural_manifest: Path,
    output: Path,
    modeling_manifest: Path | None = None,
    evaluation_report: Path | None = None,
    fixture: bool = False,
    max_positions: int = 256,
    seed: int = 20240131,
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    threads: int = 2,
    warmup: int = 10,
    iterations: int = 50,
    benchmark_batch_size: int = 8,
    exploratory_temperature: float | None = None,
) -> dict[str, Any]:
    if not 1 <= max_positions <= 256 or threads < 1 or seed < 0:
        raise ValueError("Require 1..256 positions, positive threads and nonnegative seed")
    if (
        not batch_sizes
        or batch_sizes[0] != 1
        or len(set(batch_sizes)) != len(batch_sizes)
        or any(size < 1 or size > 256 for size in batch_sizes)
    ):
        raise ValueError("Batch sizes must be unique, start with 1 and lie within 1..256")
    if not 1 <= warmup <= 1000 or not 1 <= iterations <= 1000 or benchmark_batch_size < 2:
        raise ValueError("Require 1..1000 warmups/iterations and benchmark batch >= 2")
    if exploratory_temperature is not None:
        softmax_wdl(np.zeros((1, 3)), exploratory_temperature)
    if output.exists():
        raise RuntimeError(
            "Output already exists; choose a fresh directory to rerun parity/benchmark"
        )
    manifest, config = validate_checkpoint_config(neural_manifest)
    root = neural_manifest.parent
    if evaluation_report is not None:
        evaluated = read_json_object(evaluation_report)
        _require_equal(
            evaluated["experiment_id"], manifest["experiment_id"], "evaluated experiment"
        )
        _require_equal(
            evaluated["modeling_dataset_id"],
            manifest["identity_payload"]["modeling_dataset_id"],
            "evaluated dataset",
        )
        for name in SOURCE_FILES.values():
            record = evaluated["sources"]["neural_manifest" if name == "_manifest.json" else name]
            _require_equal(
                _integrity(root / name),
                {
                    "sha256": record["sha256"],
                    "size_bytes": record["size_bytes"],
                },
                f"evaluation source {name}",
            )
        if exploratory_temperature is not None:
            _require_equal(
                exploratory_temperature, evaluated["temperature"], "exploratory T source"
            )
    preprocessor, model = load_frozen_model(root, config)
    if fixture:
        examples, lineage = (
            fixture_positions(preprocessor)[:max_positions],
            {"kind": "fixture", "test_split_access": False},
        )
    else:
        examples, lineage = _validation_positions(
            modeling_manifest or config.input.modeling_manifest_path,
            manifest,
            config,
            max_positions,
            seed,
        )
    if len(examples) < benchmark_batch_size:
        raise ValueError("Not enough parity positions for the requested benchmark batch")
    model.to(device="cpu", dtype=torch.float32)
    inputs, masks = _encode(examples, preprocessor)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + "__" + uuid.uuid4().hex)
    previous_threads = torch.get_num_threads()
    with _SingleWriterLock(output.with_name(output.name + ".lock")):
        if output.exists():
            raise RuntimeError("Output was created by another writer")
        staging.mkdir()
        try:
            torch.set_num_threads(threads)
            for destination, source in SOURCE_FILES.items():
                shutil.copy2(root / source, staging / destination)
            if evaluation_report is not None:
                shutil.copy2(evaluation_report, staging / "source_evaluation_report.json")
            onnx_path = staging / "model.onnx"
            sample = tuple(torch.from_numpy(array[:1]) for array in inputs.values())
            with torch.inference_mode():
                torch.onnx.export(
                    model,
                    sample,
                    str(onnx_path),
                    input_names=list(INPUT_SPEC),
                    output_names=list(OUTPUT_SPEC),
                    opset_version=OPSET,
                    dynamo=False,
                    dynamic_axes={name: {0: "batch"} for name in (*INPUT_SPEC, *OUTPUT_SPEC)},
                    external_data=False,
                )
            onnx.checker.check_model(str(onnx_path), full_check=True)
            session = _session(onnx_path, threads)
            parity, arrays = evaluate_parity(
                model=model,
                session=session,
                inputs=inputs,
                masks=masks,
                examples=examples,
                batch_sizes=batch_sizes,
                exploratory_temperature=exploratory_temperature,
            )
            benchmark = (
                benchmark_forward(
                    model,
                    session,
                    inputs,
                    small_batch=benchmark_batch_size,
                    warmup=warmup,
                    iterations=iterations,
                )
                if parity["passed"]
                else {"skipped": "parity failed", "results": []}
            )
            repository = Path(__file__).resolve().parents[3]
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                text=True,
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=repository,
                    text=True,
                ).strip()
            )
            metadata = {
                "schema_version": "cpu_fp32_onnx_bundle_v1",
                "ready_for_handoff": parity["passed"] and not fixture,
                "experiment_id": manifest["experiment_id"],
                "modeling_dataset_id": manifest["identity_payload"]["modeling_dataset_id"],
                "modeling_manifest_canonical_sha256": manifest["identity_payload"][
                    "modeling_manifest_sha256"
                ],
                "source_artifacts": {
                    destination: _integrity(staging / destination) for destination in SOURCE_FILES
                },
                "evaluation_report": None
                if evaluation_report is None
                else {
                    "name": "source_evaluation_report.json",
                    **_integrity(staging / "source_evaluation_report.json"),
                },
                "exporter": {
                    "name": "torch.onnx.export",
                    "dynamo": False,
                    "opset": OPSET,
                    "external_data": False,
                    "dtype": "float32",
                    "device": "cpu",
                },
                "io": io_contract(),
                "versions": asdict(config.versions),
                "wdl_class_order": list(WDL_CLASS_ORDER),
                "context": {
                    "continuous_order": [
                        f"{field}_{suffix}"
                        for field in CONTINUOUS_CONTEXT_FIELDS
                        for suffix in ("zscore", "missing_flag")
                    ],
                    "missing_numeric": "zscore=0, missing_flag=1; observed missing_flag=0",
                    "rating_band_mapping": "feature_preprocessing.json#/rating_band_to_index",
                    "time_control_mapping": "feature_preprocessing.json#/time_control_to_index",
                    "unknown_categories": "__UNK__ index 0 from saved preprocessor",
                },
                "postprocessing": {
                    "legal_mask": "python-chess legal_move_mask; never rank unmasked actions",
                    "action_index": "from_square * 73 + plane; absolute board orientation",
                    "ranking": parity["tie_break"],
                    "wdl": "float64 stable softmax(value_logits / T), mover perspective",
                    "production_temperature": 1.0,
                    "exploratory_temperature": exploratory_temperature,
                    "exploratory_status": "not approved for production; graph outputs raw logits",
                },
                "files": {
                    "onnx": "model.onnx",
                    "preprocessor": "feature_preprocessing.json",
                    "parity": "parity.json",
                    "arrays": "parity_arrays.npz",
                },
                "code": {
                    "git_sha": git_sha,
                    "worktree_dirty": dirty,
                    "source_hashes": {
                        str(path.relative_to(repository)): _integrity(path)
                        for path in [
                            Path(__file__),
                            Path(__file__).with_name("neural_checkpoint.py"),
                            Path(__file__).with_name("onnx_parity.py"),
                            Path(__file__).with_name("neural_network.py"),
                            Path(__file__).with_name("neural_data.py"),
                            Path(action_encoding.__file__),
                            Path(position_encoding.__file__),
                            repository / "pyproject.toml",
                            repository / "uv.lock",
                        ]
                    },
                },
            }
            report = {
                "schema_version": "cpu_fp32_onnx_parity_v1",
                "parity": parity,
                "lineage": {
                    **lineage,
                    "seed": seed,
                    "selected_key_fingerprint": selected_key_hash(examples),
                },
                "runtime": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "onnx": onnx.__version__,
                    "onnxruntime": ort.__version__,
                    "numpy": np.__version__,
                    "provider": session.get_providers(),
                    "torch_intra_op_threads": torch.get_num_threads(),
                    "torch_inter_op_threads": torch.get_num_interop_threads(),
                    "ort_intra_op_threads": threads,
                    "ort_inter_op_threads": 1,
                    "ort_execution_mode": "ORT_SEQUENTIAL",
                    "ort_graph_optimization": "ORT_ENABLE_ALL",
                    "cpu": platform.processor(),
                    "machine": platform.machine(),
                    "physical_cores": psutil.cpu_count(logical=False),
                    "logical_cores": psutil.cpu_count(),
                    "os": platform.platform(),
                },
                "model_sizes": {
                    "pytorch_state_bytes": (staging / "model_state_best.pt").stat().st_size,
                    "onnx_total_bytes": onnx_path.stat().st_size,
                    "onnx_files": [{"name": "model.onnx", **_integrity(onnx_path)}],
                },
                "benchmark": benchmark,
            }
            _write_json(staging / "metadata.json", metadata)
            _write_json(staging / "parity.json", report)
            np.savez_compressed(staging / "parity_arrays.npz", allow_pickle=False, **arrays)
            (staging / "report.md").write_text(_markdown(report, metadata), encoding="utf-8")
            _write_json(
                staging / "_manifest.json",
                {
                    "status": "complete" if parity["passed"] else "failed",
                    "artifact_integrity": {
                        path.name: _integrity(path)
                        for path in sorted(staging.iterdir())
                        if path.is_file()
                    },
                },
            )
            if parity["passed"]:
                (staging / "_SUCCESS").write_text("ok\n", encoding="utf-8")
            os.replace(staging, output)
            return report
        except (RuntimeError, ValueError, onnx.checker.ValidationError) as exc:
            if staging.exists():
                _write_json(staging / "failure.json", {"status": "failed", "error": str(exc)})
                os.replace(staging, output)
            raise
        finally:
            torch.set_num_threads(previous_threads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modeling-manifest", type=Path)
    parser.add_argument("--evaluation-report", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--max-positions", default=256, type=int)
    parser.add_argument("--seed", default=20240131, type=int)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--iterations", default=50, type=int)
    parser.add_argument("--benchmark-batch-size", default=8, type=int)
    parser.add_argument("--exploratory-temperature", type=float)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        report = run_onnx_export(
            neural_manifest=args.neural_manifest,
            output=args.output,
            modeling_manifest=args.modeling_manifest,
            evaluation_report=args.evaluation_report,
            fixture=args.fixture,
            max_positions=args.max_positions,
            seed=args.seed,
            batch_sizes=tuple(args.batch_sizes),
            threads=args.threads,
            warmup=args.warmup,
            iterations=args.iterations,
            benchmark_batch_size=args.benchmark_batch_size,
            exploratory_temperature=args.exploratory_temperature,
        )
    except (OSError, ValueError, RuntimeError, KeyError, onnx.checker.ValidationError) as exc:
        print(f"ONNX export/parity failed: {exc}; inspect {args.output}", file=sys.stderr)
        return 1
    print(
        json.dumps(report, indent=2, sort_keys=True)
        if args.json
        else (f"parity_passed={report['parity']['passed']} bundle={args.output}")
    )
    if not report["parity"]["passed"]:
        print(f"Parity failed; original differences retained in {args.output}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
