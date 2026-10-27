"""Prepare a real 100k Phase 3.2 HPO config from pinned Phase 3.1 references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path.as_posix()}")
    return path.read_text(encoding="utf-8")


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(_read_text(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path.as_posix()}")
    return payload


def _non_empty(value: Any, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _resolve_manifest_path(text: str) -> Path:
    raw = Path(text)
    return raw if raw.is_absolute() else (Path.cwd() / raw)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-fast preparation of configs/hpo/real_100k_reference.yaml from "
            "C:/ChessLensWork/phase31_neural/data_scaling_100k pointers"
        )
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("C:/ChessLensWork/phase31_neural/data_scaling_100k"),
        help="Phase 3.1 100k reference directory",
    )
    parser.add_argument(
        "--phase32-output-root",
        type=Path,
        default=Path("C:/ChessLensWork/phase32_experiments"),
        help="Output root for new Phase 3.2 real HPO runs",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("configs/hpo/real_100k_reference.yaml"),
        help="Destination config path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        reference_root = args.reference_root
        neural_config_path = reference_root / "neural_config.json"
        baseline_id_path = reference_root / "baseline_experiment_id.txt"
        neural_id_path = reference_root / "neural_experiment_id.txt"

        neural_config = _load_json_object(neural_config_path)

        baseline_experiment_id = _non_empty(
            _read_text(baseline_id_path),
            name="baseline_experiment_id",
        )
        neural_experiment_id = _non_empty(
            _read_text(neural_id_path),
            name="neural_experiment_id",
        )

        input_payload = neural_config.get("input")
        runtime_payload = neural_config.get("runtime")
        if not isinstance(input_payload, dict):
            raise ValueError("neural_config.json is missing input block")
        if not isinstance(runtime_payload, dict):
            raise ValueError("neural_config.json is missing runtime block")

        modeling_manifest_text = _non_empty(
            input_payload.get("modeling_manifest_path"),
            name="input.modeling_manifest_path",
        )
        modeling_manifest_path = _resolve_manifest_path(modeling_manifest_text)

        baseline_manifest_path = (
            reference_root
            / "baselines"
            / "experiments"
            / baseline_experiment_id
            / "_manifest.json"
        )
        neural_manifest_path = (
            reference_root
            / "neural"
            / "experiments"
            / neural_experiment_id
            / "_manifest.json"
        )

        for required in (
            modeling_manifest_path,
            baseline_manifest_path,
            neural_manifest_path,
        ):
            if not required.exists():
                raise FileNotFoundError(
                    "Required reference manifest path not found: " + required.as_posix()
                )

        runtime_seed = int(runtime_payload.get("seed", 20240131))
        runtime_threads = int(runtime_payload.get("threads", 1))

        config_payload: dict[str, Any] = {
            "reference": {
                "neural_config_path": neural_config_path.as_posix(),
                "modeling_manifest_path_override": modeling_manifest_path.as_posix(),
                "baseline_reference_manifest_path_override": baseline_manifest_path.as_posix(),
            },
            "study": {
                "name": "real_100k_validation_hpo",
                "output_root": args.phase32_output_root.as_posix(),
                "storage_path": (
                    args.phase32_output_root
                    / "studies"
                    / "real_100k_validation_hpo"
                    / "optuna_study.db"
                ).as_posix(),
                "evaluation_scope": "validation_only",
                "objective_metric": "validation.policy.mrr",
            },
            "budget": {
                "max_total_trials": 4,
                "max_new_trials_per_invocation": 1,
            },
            "search_space": {
                "learning_rate": {
                    "low": 1e-4,
                    "high": 2e-3,
                    "log": True,
                },
                "weight_decay": {
                    "low": 1e-5,
                    "high": 1e-2,
                    "log": True,
                },
                "value_weight_choices": [0.2, 0.4, 0.8],
            },
            "runtime": {
                "seed": runtime_seed,
                "threads": runtime_threads,
                "device": "cpu",
            },
            "pruning": {
                "startup_trials": 1,
                "warmup_epochs": 1,
            },
            "behavior": {
                "strict": True,
            },
            "execution": {
                "trial_concurrency": 1,
            },
        }

        args.output_config.parent.mkdir(parents=True, exist_ok=True)
        args.output_config.write_text(
            yaml.safe_dump(config_payload, sort_keys=False),
            encoding="utf-8",
        )

        summary = {
            "output_config": args.output_config.as_posix(),
            "reference_root": reference_root.as_posix(),
            "neural_config_path": neural_config_path.as_posix(),
            "modeling_manifest_path": modeling_manifest_path.as_posix(),
            "baseline_manifest_path": baseline_manifest_path.as_posix(),
            "neural_manifest_path": neural_manifest_path.as_posix(),
            "runtime_threads": runtime_threads,
            "runtime_seed": runtime_seed,
            "phase32_output_root": args.phase32_output_root.as_posix(),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"prepare_real_hpo_config_error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("prepare_real_hpo_config_ok")
        for key, value in summary.items():
            print(f"{key}={value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
