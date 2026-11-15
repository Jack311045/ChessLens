"""Reproduce a frozen neural checkpoint's exploratory validation calibration pilot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch

from chesslens.features import action_encoding, position_encoding
from chesslens.modeling.calibration import fit_temperature, temperature_scale
from chesslens.modeling.neural_config import (
    NeuralConfig,
    load_neural_config,
    neural_config_identity_payload,
)
from chesslens.modeling.neural_data import (
    CONTINUOUS_CONTEXT_FIELDS,
    WDL_CLASS_ORDER,
    NeuralPreprocessor,
    encode_examples,
    selected_key_hash,
    value_target_to_index,
)
from chesslens.modeling.neural_network import NeuralModelSizes, PolicyValueResidualCNN
from chesslens.modeling.neural_training import (
    _batch_tensors,
    _file_sha256,
    _neural_preflight,
    _SingleWriterLock,
    _validate_existing_experiment,
)
from chesslens.modeling.run_baselines import (
    PositionExample,
    PreflightSummary,
    _load_position_examples,
    _logistic_metrics,
    _position_selection_query,
)
from chesslens.modeling.validation import canonical_json, read_json_object, sha256_text

FORMAL_PROTOCOL = (
    "Exploratory validation pilot only: validation already informed checkpoint selection, "
    "early stopping and/or HPO; a game-wise re-split does not restore independence. "
    "For formal acceptance, pre-lock the checkpoint, fit T on a previously unused calibration "
    "set excluded from training/HPO/early stopping, then lock T before evaluating an independent "
    "pre-reserved test set. Preserve game-level temporal and player-holdout semantics. "
    "If existing test results influenced model or calibration adoption, collect a new unused "
    "evaluation set. Historical test usage outside this pipeline is unknown. "
    "This CLI never loads test labels or computes test metrics. "
    "WDL temperature scaling does not improve or change policy MRR."
)


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{field} mismatch: expected={expected!r}, actual={actual!r}")


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


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


def _verify_selection(
    preflight: PreflightSummary,
    config: NeuralConfig,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Reproduce all key fingerprints without reading train/test labels or board tensors."""
    split_hashes: dict[str, str] = {}
    population_keys: list[list[Any]] = []
    with duckdb.connect() as connection:
        for split in ("train", "validation", "test"):
            query = _position_selection_query(
                preflight.policy_paths_by_split[split],
                columns="game_id, ply, player_disjoint_training_eligible, is_player_holdout_game",
                limit=getattr(config.selection, f"max_{split}_positions"),
                seed=config.runtime.seed,
            )
            rows = connection.execute(query).fetchall()
            keys = [[str(row[0]), int(row[1])] for row in rows]
            if len({(key[0], key[1]) for key in keys}) != len(keys):
                raise RuntimeError(f"Duplicate (game_id, ply) keys in {split}")
            split_hashes[split] = sha256_text(canonical_json(keys))
            if split == "train":
                population_keys = [
                    [str(row[0]), int(row[1])]
                    for row in rows
                    if config.selection.training_population_mode == "temporal_all" or bool(row[2])
                ]
                if config.selection.training_population_mode == "player_disjoint":
                    if any(bool(row[2]) and bool(row[3]) for row in rows):
                        raise RuntimeError("Training population contains holdout-flagged rows")
    actual = {
        "split_key_hashes": split_hashes,
        "train_population_key_hash": sha256_text(canonical_json(population_keys)),
    }
    _require_equal(actual, expected, "selection fingerprints")
    return actual


def split_validation_games(
    examples: list[PositionExample],
    *,
    seed: int,
    max_positions: int,
) -> tuple[list[PositionExample], list[PositionExample]]:
    selected = sorted(
        examples,
        key=lambda row: (
            sha256_text(canonical_json([seed, row.game_id, row.ply])),
            row.game_id,
            row.ply,
        ),
    )[:max_positions]
    games = sorted(
        {row.game_id for row in selected},
        key=lambda game: (sha256_text(canonical_json([seed, game])), game),
    )
    if len(games) < 2:
        raise RuntimeError("Validation pilot needs at least two distinct games")
    fit_games = set(games[: len(games) // 2])
    return (
        [row for row in selected if row.game_id in fit_games],
        [row for row in selected if row.game_id not in fit_games],
    )


def _support(examples: list[PositionExample]) -> dict[str, Any]:
    return {
        "positions": len(examples),
        "games": len({row.game_id for row in examples}),
        "wdl_support": {
            label: sum(row.value_target_wdl == label for row in examples)
            for label in WDL_CLASS_ORDER
        },
        "key_fingerprint": selected_key_hash(examples),
        "game_fingerprint": sha256_text(canonical_json(sorted({r.game_id for r in examples}))),
    }


def collect_wdl_logits(
    *,
    model: PolicyValueResidualCNN,
    preprocessor: NeuralPreprocessor,
    examples: list[PositionExample],
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    logits: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = encode_examples(
                examples[start : start + batch_size],
                split="validation",
                preprocessor=preprocessor,
            )
            tensors = _batch_tensors(batch, np.arange(len(batch)), device=torch.device("cpu"))
            _, values = model(
                tensors["board"],
                tensors["continuous"],
                tensors["rating_band"],
                tensors["time_control"],
            )
            if values.shape != (len(batch), 3) or not bool(torch.isfinite(values).all()):
                raise RuntimeError("Checkpoint produced invalid WDL logits")
            logits.append(values.detach().to(dtype=torch.float64, device="cpu"))
    return torch.cat(logits)


def _baseline_context(path: Path | None, neural: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return {"directly_comparable": False, "reason": "No baseline manifest supplied"}
    manifest = read_json_object(path)
    identity = manifest["identity_payload"]
    _require_equal(manifest["status"], "complete", "baseline status")
    if not (path.parent / "_SUCCESS").is_file():
        raise RuntimeError("Baseline missing _SUCCESS marker")
    _require_equal(
        sha256_text(canonical_json(identity)),
        manifest["identity_payload_sha256"],
        "baseline identity",
    )
    for name in ("aggregate_metrics.json", "effective_config.json"):
        artifact = _source(path.parent / name)
        record = manifest["artifact_integrity"][name]
        _require_equal(artifact["sha256"], record["sha256"], f"baseline {name} sha256")
        _require_equal(artifact["size_bytes"], record["size_bytes"], f"baseline {name} size")
    same_data = (
        identity["modeling_manifest_sha256"]
        == (neural["identity_payload"]["modeling_manifest_sha256"])
    )
    same_population = (
        manifest["counts"]["training_population"] == (neural["counts"]["training_population"])
    )
    return {
        "source": _source(path),
        "experiment_id": manifest["experiment_id"],
        "same_modeling_manifest": same_data,
        "same_training_population_summary": same_population,
        "directly_comparable": False,
        "reason": (
            "Historical full-validation aggregates are selection context only, NOT pilot-check "
            "comparisons. Baseline selection fingerprints and complete aligned per-row predictions "
            "have not been verified; capped prediction samples are not paired evidence. "
            "No baseline deltas, ranking comparisons or bootstrap CI are claimed."
        ),
        "full_validation_context": _validation_context(path.parent),
    }


def _validation_context(root: Path) -> dict[str, Any]:
    metrics = read_json_object(root / "aggregate_metrics.json")
    return {
        family: payload["validation"]
        for family, payload in metrics.items()
        if isinstance(payload, dict) and "validation" in payload
    }


def _selection_evidence(path: Path | None, experiment_id: str) -> dict[str, Any]:
    if path is None:
        return {"verification": "Operator-supplied rationale; no HPO snapshot supplied"}
    snapshot = read_json_object(path)
    complete = [trial for trial in snapshot["trials"] if trial["state"] == "COMPLETE"]
    if not complete:
        raise RuntimeError("Selection evidence contains no COMPLETE trials")
    for trial in complete:
        if not np.isfinite(float(trial["value"])):
            raise RuntimeError("Selection evidence contains a non-finite objective")
    ranked = sorted(complete, key=lambda trial: (-float(trial["value"]), int(trial["number"])))
    _require_equal(
        ranked[0]["user_attrs"]["neural_experiment_id"],
        experiment_id,
        "pre-test maximum validation policy MRR candidate",
    )
    return {
        "source": _source(path),
        "objective": "maximize validation_policy_mrr",
        "complete_trials": len(complete),
        "trials": [
            {
                "number": trial["number"],
                "state": trial["state"],
                "validation_policy_mrr": trial["value"],
                "experiment_id": trial["user_attrs"]["neural_experiment_id"],
            }
            for trial in ranked
        ],
        "verification": "Maximum COMPLETE validation policy MRR in supplied read-only snapshot",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Checkpoint WDL validation pilot",
        "",
        "**Exploratory; not formal test acceptance.**",
        "",
        f"- Experiment: `{report['experiment_id']}`",
        f"- Selection rationale: {report['selection_rationale']}",
        f"- Dataset: `{report['modeling_dataset_id']}`",
        f"- WDL order: `{report['wdl_class_order']}`",
        f"- Temperature: {report['temperature']:.10g}",
        f"- Fit: {report['fit']['positions']} positions / {report['fit']['games']} games",
        f"- Check: {report['check']['positions']} positions / {report['check']['games']} games",
        f"- Fit WDL support: `{report['fit']['wdl_support']}`",
        f"- Check WDL support: `{report['check']['wdl_support']}`",
        f"- Argmax differences: {report['argmax_difference_count']}",
        "",
        "| Check metric | Raw | Calibrated |",
        "|---|---:|---:|",
    ]
    for metric in ("nll", "multiclass_brier_score", "ece", "accuracy"):
        lines.append(
            f"| {metric} | {report['raw'][metric]:.10g} | {report['calibrated'][metric]:.10g} |"
        )
    lines += ["", "## Limits and acceptance protocol", "", FORMAL_PROTOCOL, ""]
    lines += [f"- {warning}" for warning in report["limitations"]]
    lines += [
        "",
        "## Provenance and ONNX handoff",
        "",
        "The JSON report contains source hashes, versions, split lineage, full-validation context "
        "and classwise reliability bins. predictions.npz preserves aligned (game_id, ply), "
        "fit/check membership, logits, targets and raw/scaled probabilities. "
        "Use the frozen model_state_best.pt, saved preprocessor and WDL order for ONNX parity; "
        "apply this exploratory T only as an explicitly labelled candidate.",
        "",
    ]
    for name, source in report["sources"].items():
        lines.append(f"- {name}: `{source['path']}` (SHA256 `{source['sha256']}`)")
    lines += [
        "",
        "## Baseline context",
        "",
        report["baseline"]["reason"],
        "",
        "No policy MRR improvement or independent-test improvement is inferred from this pilot.",
        "",
    ]
    return "\n".join(lines)


def _publish(
    output: Path,
    report: dict[str, Any],
    examples: list[PositionExample],
    logits: torch.Tensor,
    raw: torch.Tensor,
    scaled: torch.Tensor,
    fit_count: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with _SingleWriterLock(output.with_name(output.name + ".lock")):
        if output.exists():
            if not (output / "_SUCCESS").is_file():
                raise RuntimeError("Output exists without _SUCCESS; choose a new output directory")
            _require_equal(read_json_object(output / "report.json"), report, "existing report")
            integrity = read_json_object(output / "_manifest.json")["artifact_integrity"]
            for name in ("report.json", "report.md", "predictions.npz"):
                actual = _source(output / name)
                _require_equal(
                    {key: actual[key] for key in ("sha256", "size_bytes")},
                    integrity[name],
                    f"existing {name} integrity",
                )
            return
        staging = output.with_name(output.name + "__" + uuid.uuid4().hex)
        staging.mkdir()
        try:
            (staging / "report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            (staging / "report.md").write_text(_markdown(report), encoding="utf-8")
            np.savez_compressed(
                staging / "predictions.npz",
                game_id=np.asarray([row.game_id for row in examples]),
                ply=np.asarray([row.ply for row in examples], dtype=np.int64),
                partition=np.asarray(["fit"] * fit_count + ["check"] * (len(examples) - fit_count)),
                targets=np.asarray(
                    [value_target_to_index(row.value_target_wdl) for row in examples]
                ),
                logits=logits.numpy(),
                raw_probabilities=raw.numpy(),
                calibrated_probabilities=scaled.numpy(),
            )
            integrity = {
                name: {
                    key: value for key, value in _source(staging / name).items() if key != "path"
                }
                for name in ("report.json", "report.md", "predictions.npz")
            }
            (staging / "_manifest.json").write_text(
                json.dumps({"status": "complete", "artifact_integrity": integrity}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            (staging / "_SUCCESS").write_text("ok\n", encoding="utf-8")
            os.replace(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def run_checkpoint_evaluation(
    *,
    neural_manifest: Path,
    output: Path,
    modeling_manifest: Path | None = None,
    baseline_manifest: Path | None = None,
    seed: int = 20240131,
    max_positions: int = 5000,
    batch_size: int = 128,
    threads: int = 2,
    selection_rationale: str,
    selection_evidence: Path | None = None,
) -> dict[str, Any]:
    if min(max_positions, batch_size, threads) <= 0 or seed < 0:
        raise ValueError("max_positions, batch_size, threads must be positive; seed non-negative")
    if not selection_rationale.strip():
        raise ValueError("A pre-test checkpoint selection rationale is required")
    manifest = _validate_neural_manifest(neural_manifest)
    evidence = _selection_evidence(selection_evidence, manifest["experiment_id"])
    root = neural_manifest.parent
    config = load_neural_config(root / "effective_config.json")
    identity = manifest["identity_payload"]
    effective = neural_config_identity_payload(config)
    effective["input"].clear()
    saved_effective = json.loads(canonical_json(identity["effective_config"]))
    saved_effective["input"].clear()
    _require_equal(effective, saved_effective, "effective config")
    version_config = asdict(config.versions)
    for name in (
        "neural_pipeline_version",
        "feature_schema_version",
        "split_definition_version",
        "board_encoding_version",
        "action_encoding_version",
        "position_normalization_version",
    ):
        _require_equal(manifest["versions"].get(name), version_config[name], f"version {name}")
    path = modeling_manifest or config.input.modeling_manifest_path
    config = replace(config, input=replace(config.input, modeling_manifest_path=path))
    preflight = _neural_preflight(config)
    _require_equal(preflight.modeling_dataset_id, identity["modeling_dataset_id"], "dataset ID")
    _require_equal(
        preflight.modeling_manifest_sha256,
        identity["modeling_manifest_sha256"],
        "modeling manifest SHA256",
    )
    fingerprints = _verify_selection(preflight, config, manifest["selection_fingerprints"])
    context = _validation_context(root)
    _require_equal(context["value"]["class_order"], list(WDL_CLASS_ORDER), "WDL class order")
    if selection_evidence is not None:
        _require_equal(
            evidence["trials"][0]["validation_policy_mrr"],
            context["policy"]["mrr"],
            "selection evidence validation policy MRR",
        )
    validation = _load_position_examples(
        preflight.policy_paths_by_split["validation"],
        split="validation",
        limit=config.selection.max_validation_positions,
        seed=config.runtime.seed,
        novel_position_ids=set(),
    )
    _require_equal(
        selected_key_hash(validation),
        fingerprints["split_key_hashes"]["validation"],
        "validation selection fingerprint",
    )
    fit, check = split_validation_games(validation, seed=seed, max_positions=max_positions)
    if min(len(fit), len(check)) < 2:
        raise RuntimeError("Validation pilot needs at least two positions in each fit/check group")

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
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        examples = fit + check
        logits = collect_wdl_logits(
            model=model,
            preprocessor=preprocessor,
            examples=examples,
            batch_size=batch_size,
        )
        fit_targets = torch.tensor(
            [value_target_to_index(row.value_target_wdl) for row in fit],
            dtype=torch.int64,
        )
        temperature = fit_temperature(logits[: len(fit)], fit_targets)
        raw = temperature_scale(logits, 1.0)
        scaled = temperature_scale(logits, temperature)
    finally:
        torch.set_num_threads(previous_threads)
    difference_indices = torch.nonzero(raw.argmax(1) != scaled.argmax(1)).flatten().tolist()
    if difference_indices:
        raise RuntimeError(
            "Positive temperature changed WDL argmax; investigate numerical precision at keys "
            + repr([(examples[i].game_id, examples[i].ply) for i in difference_indices[:10]])
        )
    raw_metrics = _logistic_metrics(
        examples=check,
        probs=raw[len(fit) :].numpy(),
        ece_bins=config.evaluation.ece_bins,
    )
    calibrated_metrics = _logistic_metrics(
        examples=check,
        probs=scaled[len(fit) :].numpy(),
        ece_bins=config.evaluation.ece_bins,
    )
    for metrics in (raw_metrics, calibrated_metrics):
        metrics["nll"] = metrics["log_loss"]
    limitations = [
        "Historical modeling manifests do not pin policy Parquet content hashes. Current "
        "validation-file hashes and reproduced keys bind this pilot, but cannot prove historical "
        "label/feature byte identity. Original training holdout audit is preserved, not repeated.",
    ]
    for name, rows in (("fit", fit), ("check", check)):
        support = _support(rows)
        if support["games"] < 30 or len(rows) < 100:
            limitations.append(f"{name} has low support; this is a smoke/pilot estimate only")
        if any(count == 0 for count in support["wdl_support"].values()):
            limitations.append(f"{name} lacks one or more WDL classes; no classwise conclusion")
    modeling = read_json_object(path)
    report = {
        "schema_version": "checkpoint_validation_pilot_v1",
        "mode": "exploratory_validation_pilot",
        "experiment_id": manifest["experiment_id"],
        "training_git_commit": manifest["git_commit"],
        "modeling_dataset_id": preflight.modeling_dataset_id,
        "modeling_manifest_canonical_sha256": preflight.modeling_manifest_sha256,
        "selection_rationale": selection_rationale,
        "selection_evidence": evidence,
        "sources": {
            "neural_manifest": _source(neural_manifest),
            "modeling_manifest": _source(path),
            **{
                name: _source(root / name)
                for name in (
                    "model_state_best.pt",
                    "feature_preprocessing.json",
                    "effective_config.json",
                )
            },
            **{
                name: _source(Path(__file__).with_name(name))
                for name in (
                    "run_checkpoint_evaluation.py",
                    "calibration.py",
                    "neural_data.py",
                    "neural_network.py",
                    "run_baselines.py",
                    "baselines_metrics.py",
                )
            },
            "position_encoding.py": _source(Path(position_encoding.__file__)),
            "action_encoding.py": _source(Path(action_encoding.__file__)),
        },
        "versions": asdict(config.versions),
        "wdl_class_order": list(WDL_CLASS_ORDER),
        "selection_fingerprints": fingerprints,
        "training_population": manifest["counts"]["training_population"],
        "lineage": {
            "source_split": "validation",
            "splits": modeling["splits"],
            "player_holdout": modeling["player_holdout"],
            "original_evaluation_scope": manifest["evaluation_scope"],
            "test_label_access": False,
            "validation_selected_positions": len(validation),
            "validation_pilot_positions": len(examples),
            "seed": seed,
            "max_positions": max_positions,
            "split_rule": "sha256_seed_row_cap_then_sha256_seed_game_sorted_half_v1",
            "game_overlap": 0,
            "validation_file_sources": [
                _source(p) for p in preflight.policy_paths_by_split["validation"]
            ],
        },
        "runtime": {
            "device": "cpu",
            "threads": threads,
            "batch_size": batch_size,
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
        },
        "temperature": temperature,
        "temperature_fit": {"device": "cpu", "dtype": "float64", "min": 0.05, "max": 20.0},
        "fit": _support(fit),
        "check": _support(check),
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
        "argmax_difference_count": len(difference_indices),
        "full_validation_selection_context": context,
        "baseline": _baseline_context(baseline_manifest, manifest),
        "formal_acceptance_protocol": FORMAL_PROTOCOL,
        "limitations": limitations,
    }
    _publish(output, report, examples, logits, raw, scaled, len(fit))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-manifest", required=True, type=Path)
    parser.add_argument("--modeling-manifest", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection-rationale", required=True)
    parser.add_argument("--selection-evidence", type=Path, help="Read-only trials.json snapshot")
    parser.add_argument("--seed", type=int, default=20240131)
    parser.add_argument("--max-positions", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_checkpoint_evaluation(
            neural_manifest=args.neural_manifest,
            modeling_manifest=args.modeling_manifest,
            baseline_manifest=args.baseline_manifest,
            output=args.output,
            selection_rationale=args.selection_rationale,
            seed=args.seed,
            selection_evidence=args.selection_evidence,
            max_positions=args.max_positions,
            batch_size=args.batch_size,
            threads=args.threads,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Checkpoint evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else _markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
