"""Inference-only loading of integrity-checked CPU FP32 ONNX bundles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, ConfigDict, Field

from chesslens.features.action_encoding import ACTION_ENCODING_VERSION
from chesslens.features.position_encoding import (
    BOARD_ENCODING_VERSION,
    POSITION_NORMALIZATION_VERSION,
)
from chesslens.modeling.neural_data import (
    CONTINUOUS_CONTEXT_FIELDS,
    UNKNOWN_CATEGORY_TOKEN,
    WDL_CLASS_ORDER,
    NeuralPreprocessor,
)
from chesslens.modeling.onnx_parity import (
    INPUT_SPEC,
    OUTPUT_SPEC,
    create_cpu_session,
    io_contract,
)
from chesslens.modeling.validation import read_json_object

SUPPORTED_VERSIONS = {
    "action_encoding_version": ACTION_ENCODING_VERSION,
    "board_encoding_version": BOARD_ENCODING_VERSION,
    "feature_schema_version": "policy_value_features_v1",
    "neural_pipeline_version": "neural_supervised_v1",
    "position_normalization_version": POSITION_NORMALIZATION_VERSION,
    "rating_band_definition_version": "rating_band_v1",
    "split_definition_version": "temporal_game_split_v1",
    "time_control_category_definition_version": "time_control_category_v1",
}
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)]


class ArtifactIntegrity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sha256: Sha256
    size_bytes: Annotated[int, Field(ge=0, strict=True)]


class BundleManifest(BaseModel):
    status: Literal["complete"]
    artifact_integrity: dict[str, ArtifactIntegrity]


class BundleMetadata(BaseModel):
    schema_version: Literal["cpu_fp32_onnx_bundle_v1"]
    ready_for_handoff: Literal[True]
    experiment_id: Sha256
    modeling_dataset_id: Sha256
    files: dict[str, str]
    io: dict[str, Any]
    versions: dict[str, str]
    wdl_class_order: tuple[str, str, str]
    postprocessing: dict[str, Any]
    source_artifacts: dict[str, ArtifactIntegrity]
    context: dict[str, Any]
    exporter: dict[str, Any]


class ContinuousStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mean: Annotated[float, Field(strict=True, allow_inf_nan=False)]
    std: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
    observed_count: Annotated[int, Field(strict=True, ge=0)]
    missing_count: Annotated[int, Field(strict=True, ge=0)]


class SavedPreprocessor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    continuous_stats: dict[str, ContinuousStatistics]
    rating_band_to_index: dict[str, Annotated[int, Field(strict=True, ge=0)]]
    time_control_to_index: dict[str, Annotated[int, Field(strict=True, ge=0)]]


@dataclass(frozen=True)
class ServingBundle:
    session: ort.InferenceSession
    preprocessor: NeuralPreprocessor
    metadata: BundleMetadata
    model_sha256: str
    preprocessing_sha256: str


def _artifact(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if (
        Path(name).is_absolute()
        or not path.is_relative_to(root.resolve())
        or path == root.resolve()
    ):
        raise ValueError(f"Bundle artifact must be a contained relative file: {name}")
    return path


def _verify(root: Path, name: str, manifest: BundleManifest) -> ArtifactIntegrity:
    if name not in manifest.artifact_integrity:
        raise ValueError(f"Missing inference artifact integrity entry: {name}")
    path = _artifact(root, name)
    record = manifest.artifact_integrity[name]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if path.stat().st_size != record.size_bytes or digest.hexdigest() != record.sha256:
        raise ValueError(f"Inference artifact integrity mismatch: {name}")
    return record


def _validate_preprocessor(preprocessor: NeuralPreprocessor) -> None:
    if set(preprocessor.continuous_stats) != set(CONTINUOUS_CONTEXT_FIELDS):
        raise ValueError("Preprocessor continuous field contract mismatch")
    for mapping in (preprocessor.rating_band_to_index, preprocessor.time_control_to_index):
        if mapping.get(UNKNOWN_CATEGORY_TOKEN) != 0 or sorted(mapping.values()) != list(
            range(len(mapping))
        ):
            raise ValueError("Preprocessor category indices must be contiguous with __UNK__=0")


def validate_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(outputs) != 2:
        raise RuntimeError("ONNX inference must return exactly two raw-logit heads")
    for output, (_, shape) in zip(outputs, OUTPUT_SPEC.values(), strict=True):
        if (
            output.dtype != np.float32
            or output.shape != (1, *shape)
            or not np.isfinite(output).all()
        ):
            raise RuntimeError("ONNX inference returned invalid FP32 logits")
    return outputs[0], outputs[1]


def load_serving_bundle(root: Path, *, threads: int = 2) -> ServingBundle:
    """Audit artifacts are deliberately not required by the serving contract."""
    manifest = BundleManifest.model_validate(read_json_object(root / "_manifest.json"))
    if not (root / "_SUCCESS").is_file():
        raise ValueError("ONNX bundle missing completion marker _SUCCESS")
    _verify(root, "metadata.json", manifest)
    metadata = BundleMetadata.model_validate(read_json_object(root / "metadata.json"))
    if metadata.io != io_contract() or metadata.versions != SUPPORTED_VERSIONS:
        raise ValueError("Unsupported bundle I/O or encoding/version contract")
    expected_order = [
        item
        for name in CONTINUOUS_CONTEXT_FIELDS
        for item in (f"{name}_zscore", f"{name}_missing_flag")
    ]
    if metadata.context.get("continuous_order") != expected_order:
        raise ValueError("Bundle continuous context order mismatch")
    if (
        metadata.wdl_class_order != WDL_CLASS_ORDER
        or metadata.postprocessing.get("production_temperature") != 1.0
        or metadata.exporter.get("dtype") != "float32"
        or metadata.exporter.get("device") != "cpu"
        or metadata.exporter.get("external_data") is not False
    ):
        raise ValueError("Unsupported WDL, precision, device or external-data contract")
    for key in ("onnx", "preprocessor"):
        if key not in metadata.files:
            raise ValueError(f"Missing inference filename: {key}")
    model_name, preprocessing_name = metadata.files["onnx"], metadata.files["preprocessor"]
    model_record = _verify(root, model_name, manifest)
    preprocessing_record = _verify(root, preprocessing_name, manifest)
    if metadata.source_artifacts.get("feature_preprocessing.json") != preprocessing_record:
        raise ValueError("Saved preprocessor source integrity mismatch")
    preprocessing = SavedPreprocessor.model_validate(
        read_json_object(_artifact(root, preprocessing_name))
    )
    preprocessor = NeuralPreprocessor.from_dict(preprocessing.model_dump())
    _validate_preprocessor(preprocessor)
    session = create_cpu_session(_artifact(root, model_name), threads)
    warmup = {
        name: np.zeros((1, *shape), dtype=dtype) for name, (dtype, shape) in INPUT_SPEC.items()
    }
    warmup["rating_band_index"][0] = len(preprocessor.rating_band_to_index) - 1
    warmup["time_control_index"][0] = len(preprocessor.time_control_to_index) - 1
    validate_outputs(session.run(list(OUTPUT_SPEC), warmup))
    return ServingBundle(
        session, preprocessor, metadata, model_record.sha256, preprocessing_record.sha256
    )
