"""CLI for explicit-manifest MLflow imports of baseline/neural experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from chesslens.modeling.mlflow_tracking import (
    import_manifest_to_mlflow,
    mlflow_import_result_to_json,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import completed ChessLens baseline/neural experiment manifests into local MLflow"
        )
    )
    parser.add_argument(
        "--tracking-config",
        type=Path,
        default=Path("configs/mlflow/local_tracking.yaml"),
        help="Path to typed MLflow tracking config YAML",
    )
    parser.add_argument(
        "--kind",
        required=True,
        choices=("baseline", "neural"),
        help="Experiment pipeline kind for all provided manifests",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        action="append",
        help="Path to a completed experiment _manifest.json (repeatable)",
    )
    parser.add_argument(
        "--mode",
        default="historical",
        choices=("historical", "current"),
        help=(
            "Tag imported runs as historical imports or current-code runs. "
            "Use 'historical' for backfills from prior runs."
        ),
    )
    parser.add_argument(
        "--include-model-weights",
        action="store_true",
        help="Opt-in copy of model weights artifact(s) into MLflow",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    kind = args.kind
    if kind not in {"baseline", "neural"}:
        raise AssertionError("argparse choice validation failed for --kind")

    mode = args.mode
    if mode not in {"historical", "current"}:
        raise AssertionError("argparse choice validation failed for --mode")

    payloads: list[dict[str, Any]] = []
    errors: list[str] = []

    for manifest in args.manifest:
        try:
            result = import_manifest_to_mlflow(
                manifest_path=manifest,
                pipeline_kind=kind,
                tracking_config_path=args.tracking_config,
                mode=mode,
                include_model_weights=bool(args.include_model_weights),
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{manifest.as_posix()}: {exc}")
            continue

        payloads.append(mlflow_import_result_to_json(result))

    if args.json:
        print(
            json.dumps(
                {
                    "kind": kind,
                    "mode": mode,
                    "tracking_config": args.tracking_config.as_posix(),
                    "imports": payloads,
                    "errors": errors,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for item in payloads:
            print(
                "import_ok "
                f"kind={item['pipeline_kind']} "
                f"experiment_id={item['experiment_id']} "
                f"mlflow_run_id={item['mlflow_run_id']} "
                f"reused_existing={item['reused_existing']} "
                f"mode={item['mode']}"
            )
        for error_text in errors:
            print(f"import_error {error_text}", file=sys.stderr)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
