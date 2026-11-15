"""Integrity-checked reconstruction of completed neural experiments."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from chesslens.modeling.neural_config import (
    NeuralConfig,
    load_neural_config,
    neural_config_identity_payload,
)
from chesslens.modeling.neural_data import (
    CONTINUOUS_CONTEXT_FIELDS,
    WDL_CLASS_ORDER,
    NeuralPreprocessor,
)
from chesslens.modeling.neural_network import NeuralModelSizes, PolicyValueResidualCNN
from chesslens.modeling.neural_training import _validate_existing_experiment
from chesslens.modeling.validation import canonical_json, read_json_object, sha256_text


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{field} mismatch: expected={expected!r}, actual={actual!r}")


def _validate_neural_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json_object(path)
    identity = manifest["identity_payload"]
    identity_hash = sha256_text(canonical_json(identity))
    _require_equal(manifest["experiment_id"], identity_hash, "experiment identity")
    _require_equal(
        manifest["selection_fingerprints"],
        identity["selection_fingerprints"],
        "manifest selection fingerprints",
    )
    _validate_existing_experiment(
        experiment_path=path.parent,
        expected_experiment_id=identity_hash,
        expected_identity_hash=identity_hash,
    )
    _require_equal(
        read_json_object(path.parent / "experiment_identity_payload.json"),
        identity,
        "identity artifact",
    )
    return manifest


def validate_checkpoint_config(path: Path) -> tuple[dict[str, Any], NeuralConfig]:
    manifest = _validate_neural_manifest(path)
    config = load_neural_config(path.parent / "effective_config.json")
    effective = neural_config_identity_payload(config)
    effective["input"].clear()
    saved = json.loads(canonical_json(manifest["identity_payload"]["effective_config"]))
    saved["input"].clear()
    _require_equal(effective, saved, "effective config")
    versions = asdict(config.versions)
    for name in (
        "neural_pipeline_version",
        "feature_schema_version",
        "split_definition_version",
        "board_encoding_version",
        "action_encoding_version",
        "position_normalization_version",
    ):
        _require_equal(manifest["versions"].get(name), versions[name], f"version {name}")
    metrics = read_json_object(path.parent / "aggregate_metrics.json")
    _require_equal(
        metrics["value"]["validation"]["class_order"],
        list(WDL_CLASS_ORDER),
        "WDL class order",
    )
    return manifest, config


def load_frozen_model(
    root: Path,
    config: NeuralConfig,
) -> tuple[NeuralPreprocessor, PolicyValueResidualCNN]:
    preprocessor = NeuralPreprocessor.from_dict(
        read_json_object(root / "feature_preprocessing.json")
    )
    model = PolicyValueResidualCNN(
        **asdict(config.model),
        sizes=NeuralModelSizes(
            continuous_features=2 * len(CONTINUOUS_CONTEXT_FIELDS),
            rating_band_vocab_size=len(preprocessor.rating_band_to_index),
            time_control_vocab_size=len(preprocessor.time_control_to_index),
        ),
    )
    state = torch.load(root / "model_state_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return preprocessor, model
