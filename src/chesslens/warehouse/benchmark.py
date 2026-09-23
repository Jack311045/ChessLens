"""Generate sanitized benchmark summaries from published ingestion manifests."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil


def _load_manifest(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path.as_posix()}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a JSON object")
    return payload


def _safe_get_nested(payload: dict[str, Any], *keys: str) -> Any:
    node: Any = payload
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def build_benchmark_summary(dataset_root: Path, commit_sha: str | None = None) -> dict[str, Any]:
    manifest = _load_manifest(dataset_root)
    counts = manifest.get("counts", {}) if isinstance(manifest.get("counts"), dict) else {}

    benchmark: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "dataset_id": manifest.get("dataset_id"),
            "status": manifest.get("status"),
            "source_month": _safe_get_nested(manifest, "source", "source_month"),
            "source_archive_filename": _safe_get_nested(manifest, "source", "archive_filename"),
            "source_archive_sha256": _safe_get_nested(manifest, "source", "archive_sha256"),
        },
        "counts": {
            "accepted_games": counts.get("accepted_games"),
            "rejected_games": counts.get("rejected_games"),
            "emitted_moves": counts.get("emitted_moves"),
            "error_records": counts.get("error_records"),
        },
        "timings_seconds": {
            "checksum_duration_seconds": _safe_get_nested(
                manifest, "performance", "checksum_duration_seconds"
            ),
            "processing_and_validation_duration_seconds": _safe_get_nested(
                manifest, "performance", "processing_and_validation_duration_seconds"
            ),
            "total_duration_seconds": _safe_get_nested(
                manifest, "performance", "total_duration_seconds"
            ),
        },
        "performance": {
            "throughput_games_per_second": _safe_get_nested(
                manifest, "performance", "total_games_per_second"
            ),
            "throughput_moves_per_second": _safe_get_nested(
                manifest, "performance", "total_moves_per_second"
            ),
            "processing_games_per_second": _safe_get_nested(
                manifest, "performance", "processing_games_per_second"
            ),
            "processing_moves_per_second": _safe_get_nested(
                manifest, "performance", "processing_moves_per_second"
            ),
            "peak_rss_bytes": _safe_get_nested(manifest, "performance", "peak_rss_bytes"),
        },
        "versions": {
            "ingestion_pipeline": _safe_get_nested(
                manifest, "versions", "ingestion_pipeline_version"
            ),
            "schema": _safe_get_nested(manifest, "versions", "schema_version"),
            "position_normalization": _safe_get_nested(
                manifest, "versions", "position_normalization_version"
            ),
            "board_encoding": _safe_get_nested(manifest, "versions", "board_encoding_version"),
            "action_encoding": _safe_get_nested(manifest, "versions", "action_encoding_version"),
        },
        "runtime": {
            "python": platform.python_version(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gib": round(psutil.virtual_memory().total / (1024**3), 2),
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
        },
    }

    if commit_sha:
        benchmark["git_commit"] = commit_sha

    return benchmark


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate sanitized benchmark report JSON")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Published dataset root containing _manifest.json.",
    )
    parser.add_argument(
        "--output",
        default="reports/benchmarks/ingestion_benchmark.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--git-commit",
        default=None,
        help="Optional git commit SHA to include in report.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    benchmark = build_benchmark_summary(dataset_root, commit_sha=args.git_commit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path.as_posix())


if __name__ == "__main__":
    main()
