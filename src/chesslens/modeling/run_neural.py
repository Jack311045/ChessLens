"""CLI entrypoint for Phase 3.1 supervised neural training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from chesslens.modeling.mlflow_tracking import import_manifest_to_mlflow
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
    parser.add_argument(
        "--evaluation-scope",
        choices=("full", "validation_only"),
        default="full",
        help=(
            "Evaluation scope for completed training runs. "
            "Use validation_only for HPO-style runs that must not evaluate on test split."
        ),
    )
    parser.add_argument(
        "--mlflow-tracking-config",
        type=Path,
        default=None,
        help="Optional MLflow tracking config path to auto-log completed runs",
    )
    parser.add_argument(
        "--mlflow-run-origin",
        choices=("historical", "current"),
        default="current",
        help="Run-origin mode tag used for optional MLflow logging",
    )
    parser.add_argument(
        "--mlflow-include-model-weights",
        action="store_true",
        help="When MLflow logging is enabled, include model weights artifact",
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

    evaluation_scope = summary.get("evaluation_scope")
    if evaluation_scope is not None:
        lines.append(f"evaluation_scope={evaluation_scope}")

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
            evaluation_scope=args.evaluation_scope,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"run_neural_error: {exc}", file=sys.stderr)
        return 1

    payload = run_neural_result_json(result)

    if (
        args.mlflow_tracking_config is not None
        and result.completed
        and not result.reused_existing
        and not result.status_only
        and not result.dry_run
        and not result.validate_only
    ):
        try:
            mlflow_result = import_manifest_to_mlflow(
                manifest_path=result.manifest_path,
                pipeline_kind="neural",
                tracking_config_path=args.mlflow_tracking_config,
                mode=args.mlflow_run_origin,
                include_model_weights=bool(args.mlflow_include_model_weights),
            )
            payload["mlflow"] = {
                "enabled": True,
                "tracking_config": args.mlflow_tracking_config.as_posix(),
                "run_id": mlflow_result.mlflow_run_id,
                "experiment_id": mlflow_result.mlflow_experiment_id,
                "reused_existing": mlflow_result.reused_existing,
                "mode": mlflow_result.mode,
            }
        except Exception as exc:  # noqa: BLE001
            payload["mlflow"] = {
                "enabled": True,
                "tracking_config": args.mlflow_tracking_config.as_posix(),
                "error": str(exc),
            }

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
