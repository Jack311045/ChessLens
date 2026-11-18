from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.integration.test_checkpoint_evaluation import _fixture

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None or importlib.util.find_spec("onnxruntime") is None,
    reason="Install the optional onnx extra to run export integration tests",
)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    return _fixture(tmp_path_factory.mktemp("onnx_checkpoint"))


@pytest.fixture(scope="module")
def fixture_bundle(
    frozen: tuple[Path, Path, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    manifest, _, _ = frozen
    output = tmp_path_factory.mktemp("onnx_bundle") / "fixture"
    command = [
        sys.executable,
        "-m",
        "chesslens.modeling.run_onnx_export",
        "--neural-manifest",
        str(manifest),
        "--output",
        str(output),
        "--fixture",
        "--batch-sizes",
        "1",
        "8",
        "32",
        "--threads",
        "1",
        "--warmup",
        "1",
        "--iterations",
        "2",
        "--json",
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["parity"]["passed"]
    return output


def test_export_dynamic_batch_and_portable_bundle(fixture_bundle: Path, tmp_path: Path) -> None:
    from chesslens.modeling.run_onnx_export import load_bundle

    moved = tmp_path / "moved"
    shutil.copytree(fixture_bundle, moved)
    with pytest.raises(RuntimeError, match="Fixture-only"):
        load_bundle(moved)
    session, _, metadata = load_bundle(moved, threads=1, allow_fixture=True)
    assert metadata["io"]["outputs"][0]["shape"] == ["batch", 4672]
    assert not metadata["ready_for_handoff"]
    assert metadata["postprocessing"]["production_temperature"] == 1.0
    with np.load(moved / "parity_arrays.npz", allow_pickle=False) as arrays:
        for size in (1, 8):
            inputs = {node.name: arrays[node.name][:size] for node in session.get_inputs()}
            policy, value = session.run(None, inputs)
            assert policy.shape == (size, 4672) and value.shape == (size, 3)
            np.testing.assert_allclose(
                policy,
                arrays["pytorch_policy_logits_batch_1"][:size],
                rtol=1e-4,
                atol=1e-5,
            )
        assert set(arrays["board_tensor"][:, 12, 0, 0]) == {0.0, 1.0}
        assert np.any(arrays["continuous_context"][:, 1::2] == 1.0)
        assert np.any(arrays["continuous_context"][:, 1::2] == 0.0)
        assert 0 in arrays["rating_band_index"]
        assert np.any(arrays["rating_band_index"] > 0)
    report = json.loads((moved / "parity.json").read_text(encoding="utf-8"))
    assert all(item["passed"] for item in report["parity"]["comparisons"])
    assert {row["batch_size"] for row in report["benchmark"]["results"]} == {1, 8}
    (moved / "feature_preprocessing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        load_bundle(moved, allow_fixture=True)


def test_validation_selection_no_test_reads_and_both_heads(
    frozen: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chesslens.modeling.run_onnx_export as export

    manifest, modeling, _ = frozen
    original = export._load_position_examples

    def validation_only(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["split"] == "validation"
        assert all("test" not in path.stem for path in args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(export, "_load_position_examples", validation_only)
    report = export.run_onnx_export(
        neural_manifest=manifest,
        modeling_manifest=modeling,
        output=tmp_path / "validation",
        max_positions=24,
        batch_sizes=(1, 8, 16),
        threads=1,
        warmup=1,
        iterations=2,
    )
    assert report["parity"]["passed"]
    assert report["parity"]["rows"] == 24
    assert not report["lineage"]["test_split_access"]
    for item in report["parity"]["comparisons"]:
        assert item["numerical"]["policy_logits"]["over_tolerance_count"] == 0
        assert item["numerical"]["value_logits"]["over_tolerance_count"] == 0
    _, _, metadata = export.load_bundle(tmp_path / "validation")
    assert metadata["ready_for_handoff"]


def test_provenance_mismatch(
    frozen: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    from chesslens.modeling.run_onnx_export import run_onnx_export

    manifest, modeling, _ = frozen
    wrong = json.loads(modeling.read_text(encoding="utf-8"))
    wrong["modeling_dataset_id"] = "not-the-trained-dataset"
    relocated = tmp_path / "_manifest.json"
    relocated.write_text(json.dumps(wrong), encoding="utf-8")
    (tmp_path / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dataset ID mismatch"):
        run_onnx_export(
            neural_manifest=manifest,
            modeling_manifest=relocated,
            output=tmp_path / "bad",
        )
    source = json.loads(manifest.read_text(encoding="utf-8"))
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "experiment_id": source["experiment_id"],
                "modeling_dataset_id": source["identity_payload"]["modeling_dataset_id"],
                "sources": {"neural_manifest": {"sha256": "wrong", "size_bytes": 1}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="evaluation source _manifest.json mismatch"):
        run_onnx_export(
            neural_manifest=manifest,
            evaluation_report=evaluation,
            output=tmp_path / "bad-source",
            fixture=True,
        )


@pytest.mark.parametrize("nonfinite", [False, True])
def test_numerical_failure_retains_original_differences_and_skips_benchmark(
    frozen: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nonfinite: bool,
) -> None:
    import chesslens.modeling.run_onnx_export as export

    manifest, _, _ = frozen
    original = export._forward

    def inaccurate(*args: Any, **kwargs: Any) -> Any:
        reference, actual, sizes = original(*args, **kwargs)
        changed = actual[0].copy()
        changed[:, :] += np.float32(0.1)
        if nonfinite:
            changed[0, 0] = np.nan
        return reference, (changed, actual[1]), sizes

    monkeypatch.setattr(export, "_forward", inaccurate)
    output = tmp_path / "failed"
    report = export.run_onnx_export(
        neural_manifest=manifest,
        output=output,
        fixture=True,
        batch_sizes=(1, 8),
        threads=1,
        warmup=1,
        iterations=2,
    )
    assert not report["parity"]["passed"]
    assert report["benchmark"]["skipped"] == "parity failed"
    assert (output / "parity_arrays.npz").is_file()
    assert not (output / "_SUCCESS").exists()
    with pytest.raises(RuntimeError, match="not complete"):
        export.load_bundle(output, allow_fixture=True)
