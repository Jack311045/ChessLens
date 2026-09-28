"""Reproducible fixture benchmark for Phase 1.2d shard-level multiprocessing.

This benchmark uses only the committed tiny fixture, never real 2017 data.
It compares workers=1/2/4 on fresh isolated directories to report wall-clock
speedup and efficiency with process-tree RSS metrics.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from chesslens.ingestion.pgn_sharder import run_sharding
from chesslens.ingestion.run_sharded_ingestion import run_sharded_ingestion
from chesslens.ingestion.sharding_config import load_sharding_config

FIXTURE_SHARDING_CONFIG = Path("configs/sharding/fixture.yaml")
FIXTURE_PATH = Path("data/fixtures/lichess_2013_01_first20.pgn.zst")


def _write_orchestrator_config(run_dir: Path) -> Path:
    body = "\n".join(
        [
            f"input_path: {FIXTURE_PATH.resolve().as_posix()}",
            f"output_root: {(run_dir / 'processed').as_posix()}",
            "source_month: 2013-01",
            "strict: false",
            "require_complete_games: true",
            "batch_games: 5",
            "max_buffered_records: 400",
            "parquet_compression: zstd",
            "parquet_row_group_size: 128",
            "player_hash_mode: fixture_placeholder",
            "schema_version: 1.1.0",
            "position_normalization_version: fen4_legal_ep_v1",
            "board_encoding_version: board18_abs_v1",
            "action_encoding_version: action8x8x73_v1",
            f"shard_root: {(run_dir / 'raw_shards').as_posix()}",
            f"collection_root: {(run_dir / 'processed' / 'collections').as_posix()}",
            "games_per_shard: 7",
            "expected_raw_games: 20",
        ]
    )
    config_path = run_dir / "fixture_sharded.yaml"
    config_path.write_text(body + "\n", encoding="utf-8")
    return config_path


def _run_once(run_root: Path, workers: int) -> dict[str, Any]:
    run_dir = run_root / f"workers-{workers}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    sharding_config = load_sharding_config(
        FIXTURE_SHARDING_CONFIG,
        overrides={"shard_output_root": str(run_dir / "raw_shards")},
    )
    sharding_result = run_sharding(sharding_config, resume=False)
    if sharding_result.status != "complete":
        raise RuntimeError(f"Fixture sharding failed for workers={workers}")

    orchestrator_config = _write_orchestrator_config(run_dir)
    result = run_sharded_ingestion(orchestrator_config, resume=True, workers=workers)
    if result["status"] != "complete":
        raise RuntimeError(f"Sharded ingestion did not complete for workers={workers}")
    return result


def _parse_workers(raw: str) -> list[int]:
    workers: list[int] = []
    for token in raw.split(","):
        value = int(token.strip())
        if value < 1:
            raise ValueError("workers list values must be >= 1")
        workers.append(value)
    if not workers:
        raise ValueError("workers list cannot be empty")
    return workers


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fixture parallel shard ingestion")
    parser.add_argument(
        "--workers",
        default="1,2,4",
        help="Comma-separated worker counts to benchmark (default: 1,2,4)",
    )
    parser.add_argument(
        "--run-root",
        default="data/tmp/phase12d_benchmark",
        help="Benchmark scratch root (default: data/tmp/phase12d_benchmark)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path",
    )
    args = parser.parse_args()

    worker_values = _parse_workers(args.workers)
    run_root = Path(args.run_root)
    rows: list[dict[str, Any]] = []

    baseline_wall_clock: float | None = None
    for workers in worker_values:
        result = _run_once(run_root, workers)
        wall_clock = float(result["wall_clock_duration_seconds"])
        if workers == 1:
            baseline_wall_clock = wall_clock

        speedup = None
        efficiency = None
        if baseline_wall_clock is not None and wall_clock > 0:
            speedup = baseline_wall_clock / wall_clock
            efficiency = speedup / workers

        rows.append(
            {
                "workers": workers,
                "accepted_games": int(result["accepted_games"]),
                "emitted_moves": int(result["emitted_moves"]),
                "wall_clock_seconds": wall_clock,
                "games_per_second": float(result["games_per_wall_clock_second"]),
                "moves_per_second": float(result["moves_per_wall_clock_second"]),
                "peak_process_tree_rss_bytes": int(result["peak_process_tree_rss_bytes"]),
                "speedup_vs_workers_1": speedup,
                "parallel_efficiency": efficiency,
            }
        )

    payload = {
        "benchmark": "phase-1-2d-parallel-shard-ingestion-fixture",
        "rows": rows,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
