"""CLI for streaming, resumable PGN sharding (Phase 1.2c)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chesslens.ingestion.pgn_reader import compute_sha256
from chesslens.ingestion.pgn_sharder import run_sharding
from chesslens.ingestion.shard_estimates import estimate_sizes
from chesslens.ingestion.shard_manifest import (
    SHARD_MANIFEST_FILENAME,
    load_shard_manifest,
    parse_shard_entries,
)
from chesslens.ingestion.sharding_config import load_sharding_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream a parent .pgn.zst into resumable shards")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-new-shards", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _shard_dir(config_path: Path) -> tuple[Path, Path]:
    config = load_sharding_config(config_path)
    shard_dir = config.shard_output_root / config.source_month
    return shard_dir, shard_dir / SHARD_MANIFEST_FILENAME


def _print_status(config_path: Path) -> None:
    _, manifest_path = _shard_dir(config_path)
    if not manifest_path.exists():
        print(json.dumps({"status": "not_started", "shards_complete": 0}, indent=2))
        return
    manifest = load_shard_manifest(manifest_path)
    entries = parse_shard_entries(manifest)
    summary = {
        "status": manifest.get("status"),
        "shards_complete": len(entries),
        "total_emitted_games": manifest.get("total_emitted_games"),
        "expected_total_games": manifest.get("expected_total_games"),
        "last_end_global_index": entries[-1].end_global_index if entries else 0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _print_dry_run(config_path: Path) -> None:
    config = load_sharding_config(config_path)
    if not config.input_path.exists():
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "input_path": config.input_path.as_posix(),
                    "input_present": False,
                    "next_action": "parent archive missing; place it under data/raw/",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    parent_sha = compute_sha256(config.input_path)
    checksum_ok = (
        config.expected_source_sha256 is None or parent_sha == config.expected_source_sha256
    )
    shard_dir = config.shard_output_root / config.source_month
    manifest_path = shard_dir / SHARD_MANIFEST_FILENAME
    completed = 0
    if manifest_path.exists():
        completed = len(parse_shard_entries(load_shard_manifest(manifest_path)))

    expected_games = config.expected_total_games or 0
    expected_shards = (
        -(-expected_games // config.games_per_shard) if expected_games else None
    )
    estimate = estimate_sizes(expected_games or 0, disk_probe_path=config.shard_output_root)

    report = {
        "dry_run": True,
        "input_path": config.input_path.as_posix(),
        "input_present": True,
        "parent_archive_sha256": parent_sha,
        "checksum_ok": checksum_ok,
        "source_month": config.source_month,
        "games_per_shard": config.games_per_shard,
        "expected_total_games": config.expected_total_games,
        "expected_shard_count": expected_shards,
        "shard_output_dir": shard_dir.as_posix(),
        "shards_already_complete": completed,
        "size_estimates": estimate.as_dict(),
        "next_action": (
            "checksum mismatch; verify the parent archive"
            if not checksum_ok
            else "run without --dry-run (add --resume) to create or continue shards"
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    args = _build_parser().parse_args()
    config_path = Path(args.config)

    if args.status:
        _print_status(config_path)
        return
    if args.dry_run:
        _print_dry_run(config_path)
        return

    config = load_sharding_config(config_path)
    result = run_sharding(config, max_new_shards=args.max_new_shards, resume=args.resume)
    print(
        json.dumps(
            {
                "status": result.status,
                "shard_dir": result.shard_dir.as_posix(),
                "manifest_path": result.manifest_path.as_posix(),
                "parent_archive_sha256": result.parent_archive_sha256,
                "total_emitted_games": result.total_emitted_games,
                "total_shard_count": result.total_shard_count,
                "new_shards_written": result.new_shards_written,
                "reached_end_of_archive": result.reached_end_of_archive,
                "peak_rss_bytes": result.peak_rss_bytes,
                "sharding_active_seconds": round(result.sharding_active_seconds, 4),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
