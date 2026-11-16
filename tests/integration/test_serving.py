from __future__ import annotations

# ruff: noqa: E402
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from fastapi.testclient import TestClient

from chesslens.features.action_encoding import encode_move_to_index
from chesslens.modeling.neural_data import encode_examples
from chesslens.modeling.onnx_parity import INPUT_SPEC, OUTPUT_SPEC, io_contract
from chesslens.serving.app import RankRequest, create_app, encode_request
from chesslens.serving.bundle import SUPPORTED_VERSIONS, load_serving_bundle
from tests.unit.test_neural_data import _example


def _integrity(path: Path) -> dict[str, Any]:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _refresh(root: Path, name: str) -> None:
    path = root / "_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifact_integrity"][name] = _integrity(root / name)
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    from onnx import TensorProto, helper, numpy_helper

    root = tmp_path / "bundle"
    root.mkdir()
    inputs = [
        helper.make_tensor_value_info(
            name, TensorProto.FLOAT if dtype == "float32" else TensorProto.INT64, ["batch", *shape]
        )
        for name, (dtype, shape) in INPUT_SPEC.items()
    ]
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", *shape])
        for name, (_, shape) in OUTPUT_SPEC.items()
    ]
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["board_tensor"], ["mean"], axes=[1, 2, 3], keepdims=0),
            helper.make_node("Unsqueeze", ["mean", "axes"], ["offset"]),
            helper.make_node("Add", ["offset", "policy"], ["policy_logits"]),
            helper.make_node("Add", ["offset", "value"], ["value_logits"]),
        ],
        "serving-fixture",
        inputs,
        outputs,
        initializer=[
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axes"),
            numpy_helper.from_array(np.zeros((1, 4672), dtype=np.float32), "policy"),
            numpy_helper.from_array(np.asarray([[2, 0, -1]], dtype=np.float32), "value"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, root / "model.onnx")
    preprocessing = {
        "continuous_stats": {
            name: {"mean": 1000, "std": 200, "observed_count": 4, "missing_count": 0}
            for name in ("mover_rating", "opponent_rating", "rating_difference", "ply")
        },
        "rating_band_to_index": {"__UNK__": 0, "1200_1599": 1, "lt_1200": 2},
        "time_control_to_index": {"__UNK__": 0, "blitz": 1, "unknown": 2},
    }
    (root / "feature_preprocessing.json").write_text(json.dumps(preprocessing), encoding="utf-8")
    metadata = {
        "schema_version": "cpu_fp32_onnx_bundle_v1",
        "ready_for_handoff": True,
        "experiment_id": "a" * 64,
        "modeling_dataset_id": "b" * 64,
        "files": {"onnx": "model.onnx", "preprocessor": "feature_preprocessing.json"},
        "io": io_contract(),
        "versions": SUPPORTED_VERSIONS,
        "wdl_class_order": ["win", "draw", "loss"],
        "postprocessing": {"production_temperature": 1.0},
        "source_artifacts": {
            "feature_preprocessing.json": _integrity(root / "feature_preprocessing.json")
        },
        "context": {
            "continuous_order": [
                f"{name}_{suffix}"
                for name in ("mover_rating", "opponent_rating", "rating_difference", "ply")
                for suffix in ("zscore", "missing_flag")
            ]
        },
        "exporter": {"dtype": "float32", "device": "cpu", "external_data": False},
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_integrity": {
                    name: _integrity(root / name)
                    for name in ("model.onnx", "feature_preprocessing.json", "metadata.json")
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return root


def test_lifespan_loads_once_and_openapi(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chesslens.serving import app as serving

    original = serving.load_serving_bundle
    calls = []

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(serving, "load_serving_bundle", counted)
    application = create_app(bundle, threads=1)
    with TestClient(application) as client:
        assert client.get("/health").json() == {"status": "ready"}
        info = client.get("/v1/model-info").json()
        assert info["model_version"] == "a" * 64
        assert info["model_sha256"] == _integrity(bundle / "model.onnx")["sha256"]
        assert info["runtime_provider"] == "CPUExecutionProvider"
        assert info["wdl_class_order"] == ["win", "draw", "loss"] and info["temperature"] == 1
        for _ in range(2):
            assert (
                client.post("/v1/rank-moves", json={"fen": chess.STARTING_FEN}).status_code == 200
            )
        schema = client.get("/openapi.json").json()
        assert "RankRequest" in schema["components"]["schemas"]
        assert set(schema["paths"]) == {"/health", "/v1/model-info", "/v1/rank-moves"}
    assert len(calls) == 1
    assert TestClient(application).get("/health").status_code == 503


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        chess.Board(chess.STARTING_FEN.replace(" w ", " b ")).fen(),
        "7k/P7/8/8/8/8/8/7K w - - 0 1",
        "7k/8/8/8/8/8/p7/7K b - - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
    ],
)
def test_legal_ranking_probability_semantics_and_perspective(bundle: Path, fen: str) -> None:
    board = chess.Board(fen)
    expected = sorted(
        ((encode_move_to_index(move, board), move.uci()) for move in board.legal_moves),
        key=lambda item: item[0],
    )
    with TestClient(create_app(bundle, threads=1)) as client:
        response = client.post("/v1/rank-moves", json={"fen": fen, "top_k": 218})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["status"] == "ok" and result["terminal_reason"] is None
        assert result["side_to_move"] == ("white" if board.turn else "black")
        assert result["legal_move_count"] == len(expected)
        assert [(move["action_index"], move["uci"]) for move in result["moves"]] == expected
        assert [move["rank"] for move in result["moves"]] == list(range(1, len(expected) + 1))
        assert sum(move["probability"] for move in result["moves"]) == pytest.approx(1)
        one = client.post("/v1/rank-moves", json={"fen": fen, "top_k": 1}).json()
        assert one["moves"][0]["probability"] == pytest.approx(1 / len(expected))
        assert sum(result["wdl"].values()) == pytest.approx(1)
        assert result["wdl"]["win"] > result["wdl"]["draw"] > result["wdl"]["loss"]
        assert result["processing_time_ms"] >= 0
        if any(move.promotion for move in board.legal_moves):
            assert any(move["uci"].endswith("n") for move in result["moves"])


def test_label_free_encoder_matches_training_and_missing_is_not_zero(bundle: Path) -> None:
    loaded = load_serving_bundle(bundle, threads=1)
    for black in (False, True):
        board = chess.Board()
        if black:
            board.push_uci("e2e4")
        request = RankRequest.model_validate(
            {
                "fen": board.fen(),
                "context": {
                    "mover_rating": 1500,
                    "opponent_rating": 1600,
                    "time_control": "300+0",
                },
            }
        )
        example = _example(
            game_id="shared",
            fen=board.fen(),
            move_uci=next(iter(board.legal_moves)).uci(),
            mover_rating=1500,
            opponent_rating=1600,
            mover_rating_band="1200_1599",
            time_control_category="blitz",
            eco="A00",
            opening="ignored",
        )
        encoded = encode_examples(
            [replace(example, ply=board.ply())],
            split="validation",
            preprocessor=loaded.preprocessor,
        )[0]
        inputs = encode_request(request, loaded)
        np.testing.assert_array_equal(inputs["board_tensor"][0], encoded.board_tensor)
        np.testing.assert_array_equal(inputs["continuous_context"][0], encoded.continuous_context)
        assert inputs["rating_band_index"][0] == encoded.rating_band_index
        assert inputs["time_control_index"][0] == encoded.time_control_index
    missing = encode_request(RankRequest(fen=chess.STARTING_FEN), loaded)
    zero = encode_request(
        RankRequest.model_validate(
            {"fen": chess.STARTING_FEN, "context": {"mover_rating": 0, "opponent_rating": 0}}
        ),
        loaded,
    )
    np.testing.assert_array_equal(missing["continuous_context"][0, [1, 3, 5, 7]], [1, 1, 1, 0])
    np.testing.assert_array_equal(zero["continuous_context"][0, [1, 3, 5, 7]], [0, 0, 0, 0])
    assert missing["rating_band_index"][0] == 0
    assert missing["time_control_index"][0] == loaded.preprocessor.time_control_to_index["unknown"]
    unknown = encode_request(
        RankRequest.model_validate(
            {"fen": chess.STARTING_FEN, "context": {"time_control": "not-a-time-control"}}
        ),
        loaded,
    )
    np.testing.assert_array_equal(unknown["time_control_index"], missing["time_control_index"])


@pytest.mark.parametrize(
    "fen,reason",
    [
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "checkmate"),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "stalemate"),
        ("7k/8/8/8/8/8/8/K7 w - - 0 1", "insufficient_material"),
    ],
)
def test_terminal_positions(bundle: Path, fen: str, reason: str) -> None:
    with TestClient(create_app(bundle, threads=1)) as client:
        response = client.post("/v1/rank-moves", json={"fen": fen})
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "terminal" and result["terminal_reason"] == reason
        assert result["moves"] == [] and result["wdl"] is None


@pytest.mark.parametrize(
    "patch",
    [
        {"fen": "not a FEN"},
        {"fen": "8/8/8/8/8/8/8/8 w - - 0 1"},
        {"fen": " ".join(chess.STARTING_FEN.split()[:4])},
        {"fen": chess.STARTING_FEN[:-1] + "0"},
        {"fen": chess.STARTING_FEN[:-1] + "2147483648"},
        {"top_k": 0},
        {"top_k": 219},
        {"top_k": True},
        {"top_k": "5"},
        {"context": {"mover_rating": -1}},
        {"context": {"mover_rating": 10001}},
        {"context": {"mover_rating": True}},
        {"context": {"mover_rating": 1500.5}},
        {"context": {"mover_rating": "1500"}},
        {"context": {"unknown": 1}},
        {"unexpected": True},
    ],
)
def test_request_validation(bundle: Path, patch: dict[str, Any]) -> None:
    with TestClient(create_app(bundle, threads=1)) as client:
        response = client.post("/v1/rank-moves", json={"fen": chess.STARTING_FEN, **patch})
        assert response.status_code == 422, response.text


@pytest.mark.parametrize(
    "damage",
    [
        "missing_model",
        "corrupt_preprocessing",
        "missing_marker",
        "version",
        "io",
        "temperature",
        "context_order",
        "fixture",
        "path",
        "statistics",
        "graph_shape",
        "graph_nan",
    ],
)
def test_startup_rejects_bad_inference_bundles(bundle: Path, damage: str) -> None:
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if damage == "missing_model":
        (bundle / "model.onnx").unlink()
    elif damage == "corrupt_preprocessing":
        (bundle / "feature_preprocessing.json").write_text("{}")
    elif damage == "missing_marker":
        (bundle / "_SUCCESS").unlink()
    elif damage in {"graph_shape", "graph_nan"}:
        model = onnx.load(bundle / "model.onnx")
        if damage == "graph_shape":
            model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 19
        else:
            from onnx import numpy_helper

            model.graph.initializer[1].CopyFrom(
                numpy_helper.from_array(np.full((1, 4672), np.nan, dtype=np.float32), "policy")
            )
        onnx.save(model, bundle / "model.onnx")
        _refresh(bundle, "model.onnx")
    elif damage == "statistics":
        path = bundle / "feature_preprocessing.json"
        payload = json.loads(path.read_text())
        payload["continuous_stats"]["ply"]["std"] = 0
        path.write_text(json.dumps(payload))
        _refresh(bundle, path.name)
        metadata["source_artifacts"][path.name] = _integrity(path)
    else:
        if damage == "version":
            metadata["versions"]["board_encoding_version"] = "unsupported"
        elif damage == "io":
            metadata["io"]["outputs"][1]["shape"] = ["batch", 4]
        elif damage == "temperature":
            metadata["postprocessing"]["production_temperature"] = 1.04
        elif damage == "context_order":
            metadata["context"]["continuous_order"].reverse()
        elif damage == "fixture":
            metadata["ready_for_handoff"] = False
        elif damage == "path":
            metadata["files"]["onnx"] = "..\\outside.onnx"
    metadata_path.write_text(json.dumps(metadata))
    _refresh(bundle, "metadata.json")
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        with TestClient(create_app(bundle, threads=1)):
            pytest.fail("Invalid inference bundle became ready")


def test_relocated_minimal_bundle_and_environment_config(
    bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    moved = tmp_path / "relocated"
    shutil.copytree(bundle, moved)
    manifest = json.loads((moved / "_manifest.json").read_text())
    manifest["artifact_integrity"]["parity_arrays.npz"] = {"sha256": "0" * 64, "size_bytes": 1}
    (moved / "_manifest.json").write_text(json.dumps(manifest))
    shutil.rmtree(bundle)
    monkeypatch.setenv("CHESSLENS_BUNDLE_DIR", str(moved))
    with TestClient(create_app(threads=1)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/rank-moves", json={"fen": chess.STARTING_FEN}).status_code == 200
    monkeypatch.delenv("CHESSLENS_BUNDLE_DIR")
    with pytest.raises(RuntimeError, match="CHESSLENS_BUNDLE_DIR"):
        with TestClient(create_app()):
            pytest.fail("Missing configuration became ready")


def test_serving_does_not_import_training_stack(bundle: Path) -> None:
    code = (
        "import sys; from pathlib import Path; "
        "from chesslens.serving.bundle import load_serving_bundle; "
        f"load_serving_bundle(Path({str(bundle)!r}), threads=1); "
        "assert not {'torch', 'onnx', 'duckdb', "
        "'chesslens.modeling.run_baselines'} & sys.modules.keys()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")},
    )
    assert result.returncode == 0, result.stderr
