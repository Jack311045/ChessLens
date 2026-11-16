"""Reproduce validation-row encoding/HTTP parity and a bounded localhost latency sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import chess
import httpx
import numpy as np
import psutil

from chesslens.features.action_encoding import encode_move_to_index
from chesslens.modeling.neural_checkpoint import validate_checkpoint_config
from chesslens.modeling.onnx_parity import INPUT_SPEC, softmax_wdl
from chesslens.modeling.run_onnx_export import _validation_positions
from chesslens.modeling.validation import read_json_object
from chesslens.serving.app import RankRequest, encode_request
from chesslens.serving.bundle import load_serving_bundle


def verify(
    bundle_path: Path,
    neural_manifest: Path,
    output: Path,
    url: str,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"Evidence output must be new: {output}")
    bundle = load_serving_bundle(bundle_path)
    manifest = read_json_object(bundle_path / "_manifest.json")
    for name in ("parity.json", "parity_arrays.npz"):
        path = bundle_path / name
        record = manifest["artifact_integrity"][name]
        if (
            path.stat().st_size != record["size_bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"Accepted parity artifact integrity mismatch: {name}")
    parity = read_json_object(bundle_path / "parity.json")
    if not parity["parity"]["passed"]:
        raise ValueError("Reference bundle parity did not pass")
    source, config = validate_checkpoint_config(neural_manifest)
    if source["experiment_id"] != bundle.metadata.experiment_id:
        raise ValueError("Neural source and served experiment identity differ")
    examples, lineage = _validation_positions(
        config.input.modeling_manifest_path,
        source,
        config,
        parity["parity"]["rows"],
        20240131,
    )
    max_policy_error = max_wdl_error = 0.0
    first_request: dict[str, Any] = {}
    first_response: dict[str, Any] = {}
    with httpx.Client(base_url=url, timeout=30, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json() == {"status": "ready"}
        info_response = client.get("/v1/model-info")
        info_response.raise_for_status()
        info = info_response.json()
        assert info["model_version"] == bundle.metadata.experiment_id
        assert info["model_sha256"] == bundle.model_sha256
        assert info["preprocessing_sha256"] == bundle.preprocessing_sha256
        assert info["temperature"] == 1 and info["runtime_provider"] == "CPUExecutionProvider"
        with np.load(bundle_path / "parity_arrays.npz", allow_pickle=False) as arrays:
            assert len(examples) == len(arrays["fen"])
            for idx, example in enumerate(examples):
                assert (example.game_id, example.ply, example.pre_move_fen) == (
                    str(arrays["game_id"][idx]),
                    int(arrays["ply"][idx]),
                    str(arrays["fen"][idx]),
                )
                board = chess.Board(example.pre_move_fen)
                assert board.ply() == example.ply
                body = {
                    "fen": example.pre_move_fen,
                    "context": {
                        "mover_rating": example.mover_rating,
                        "opponent_rating": example.opponent_rating,
                        "time_control": example.time_control_raw,
                    },
                    "top_k": 5,
                }
                inputs = encode_request(RankRequest.model_validate(body), bundle)
                for name in INPUT_SPEC:
                    np.testing.assert_array_equal(inputs[name][0], arrays[name][idx])
                response = client.post("/v1/rank-moves", json=body)
                response.raise_for_status()
                result = response.json()
                assert result["status"] == "ok"
                assert result["model_version"] == info["model_version"]
                assert result["model_sha256"] == info["model_sha256"]
                assert result["side_to_move"] == ("white" if board.turn else "black")
                logits = arrays["ort_policy_logits_batch_1"][idx]
                legal = [encode_move_to_index(move, board) for move in board.legal_moves]
                ranked = sorted(legal, key=lambda action: (-float(logits[action]), action))
                scores = logits[ranked].astype(np.float64)
                probabilities = np.exp(scores - scores.max())
                probabilities /= probabilities.sum()
                assert [move["action_index"] for move in result["moves"]] == ranked[:5]
                for move, action in zip(result["moves"], ranked[:5], strict=True):
                    assert encode_move_to_index(chess.Move.from_uci(move["uci"]), board) == action
                    assert chess.Move.from_uci(move["uci"]) in board.legal_moves
                actual_policy = [move["probability"] for move in result["moves"]]
                actual_wdl = [result["wdl"][label] for label in ("win", "draw", "loss")]
                expected_wdl = softmax_wdl(arrays["ort_value_logits_batch_1"][idx : idx + 1])[0]
                np.testing.assert_allclose(actual_policy, probabilities[:5], atol=1e-8, rtol=1e-6)
                np.testing.assert_allclose(actual_wdl, expected_wdl, atol=1e-8, rtol=1e-6)
                max_policy_error = max(
                    max_policy_error, float(np.max(np.abs(actual_policy - probabilities[:5])))
                )
                max_wdl_error = max(max_wdl_error, float(np.max(np.abs(actual_wdl - expected_wdl))))
                if idx == 0:
                    first_request, first_response = body, result
        omitted = client.post("/v1/rank-moves", json={"fen": first_request["fen"]})
        omitted.raise_for_status()
        assert omitted.json()["status"] == "ok" and omitted.json()["moves"]
        durations = []
        for idx in range(35):
            start = perf_counter()
            measured = client.post("/v1/rank-moves", json=first_request)
            measured.raise_for_status()
            if idx >= 5:
                durations.append((perf_counter() - start) * 1000)
    report = {
        "status": "passed",
        "base_url": url,
        "model_info": info,
        "validation_rows": len(examples),
        "exact_encoded_rows": len(examples),
        "legal_top5_matches": len(examples),
        "max_policy_probability_absolute_error": max_policy_error,
        "max_wdl_probability_absolute_error": max_wdl_error,
        "context_source": "original validation rows, saved training selection and parity key order",
        "validation_lineage": lineage,
        "omitted_context_response": omitted.json(),
        "http_latency": {
            "warmup": 5,
            "samples": len(durations),
            "p50_ms": float(np.percentile(durations, 50)),
            "p95_ms": float(np.percentile(durations, 95)),
            "samples_ms": durations,
            "scope": "persistent localhost HTTP POST through body receipt, excluding JSON decoding",
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "python": platform.python_version(),
            "packages": {
                name: version(name) for name in ("fastapi", "uvicorn", "onnxruntime", "httpx")
            },
        },
    }
    output.mkdir(parents=True)
    for name, payload in (
        ("report.json", report),
        ("known_validation.request.json", first_request),
        ("known_validation.response.json", first_response),
        ("health.json", health.json()),
        ("model-info.json", info),
    ):
        (output / name).write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--neural-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    report = verify(args.bundle, args.neural_manifest, args.output, args.url)
    print(json.dumps({key: value for key, value in report.items() if key != "validation_lineage"}))


if __name__ == "__main__":
    main()
