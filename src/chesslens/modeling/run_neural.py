"""CLI entrypoint for Phase 3.1 supervised neural training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from chesslens.modeling.neural_training import run_neural, run_neural_result_json

DEFAULT_CONFIG_PATH = Path("configs/neural/fixture_smoke.yaml")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/validate/status for ChessLens Phase 3.1 neural model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to neural config YAML",
    )
    parser.add_argument(
        "--modeling-manifest-path",
        type=str,
        default=None,
        help="Override modeling manifest path",
    )
    parser.add_argument(
        "--baseline-reference-manifest-path",
        type=str,
        default=None,
        help="Optional Phase 2.2 baseline manifest for compatibility checks",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override output root directory",
    )
    parser.add_argument(
        "--max-train-positions",
        type=int,
        default=None,
        help="Override selected train positions",
    )
    parser.add_argument(
        "--max-validation-positions",
        type=int,
        default=None,
        help="Override selected validation positions",
    )
    parser.add_argument(
        "--max-test-positions",
        type=int,
        default=None,
        help="Override selected test positions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cpu", "cuda", "auto"),
        help="Override runtime device",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config and selection only; do not train",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run selection + encoding validation only",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print status for this experiment identity",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint for this experiment identity",
    )
    parser.add_argument(
        "--max-epochs-this-session",
        type=int,
        default=None,
        help="Optional cap for epochs trained in this invocation",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full machine-readable JSON output",
    )
    return parser


def _validate_mode_args(args: argparse.Namespace) -> None:
    mode_flags = [bool(args.dry_run), bool(args.validate_only), bool(args.status)]
    if sum(mode_flags) > 1:
        raise RuntimeError("Use at most one of --dry-run, --validate-only, or --status")
    if args.resume and args.status:
        raise RuntimeError("--resume cannot be combined with --status")


def _summary_lines(payload: dict[str, Any]) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return []

    lines: list[str] = []
    status = summary.get("status")
    if status is not None:
        lines.append(f"status={status}")

    baseline_compat = summary.get("baseline_compatibility")
    if isinstance(baseline_compat, dict):
        compatible = baseline_compat.get("compatible")
        reason = baseline_compat.get("reason")
        lines.append(f"baseline_compatibility={compatible}")
        if reason:
            lines.append(f"baseline_reason={reason}")

    best_epoch = summary.get("best_epoch")
    if best_epoch is not None:
        lines.append(f"best_epoch={best_epoch}")

    best_mrr = summary.get("best_validation_policy_mrr")
    if best_mrr is not None:
        lines.append(f"best_validation_policy_mrr={best_mrr}")

    return lines


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_mode_args(args)
        result = run_neural(
            config_path=args.config,
            modeling_manifest_override=args.modeling_manifest_path,
            baseline_reference_manifest_override=args.baseline_reference_manifest_path,
            output_root_override=args.output_root,
            max_train_positions_override=args.max_train_positions,
            max_validation_positions_override=args.max_validation_positions,
            max_test_positions_override=args.max_test_positions,
            device_override=args.device,
            dry_run=bool(args.dry_run),
            validate_only=bool(args.validate_only),
            resume=bool(args.resume),
            status_only=bool(args.status),
            max_epochs_this_session=args.max_epochs_this_session,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"run_neural_error: {exc}", file=sys.stderr)
        return 1

    payload = run_neural_result_json(result)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"run_id={payload['run_id']}")
        print(f"experiment_id={payload['experiment_id']}")
        print(f"reused_existing={payload['reused_existing']}")
        print(f"completed={payload['completed']}")
        print(f"output_path={payload['output_path']}")
        print(f"manifest_path={payload['manifest_path']}")
        for line in _summary_lines(payload):
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
