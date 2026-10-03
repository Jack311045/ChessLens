"""Safely relocate stale path literals in a sampled DuckDB warehouse snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chesslens.warehouse.sampled_snapshot import (
    SampledSnapshotRelocationResult,
    relocate_sampled_snapshot,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a deterministic_sample DuckDB snapshot to a working location and "
            "relocate stale absolute path literals in existing view SQL."
        )
    )
    parser.add_argument("--source-duckdb", required=True)
    parser.add_argument("--target-directory", required=True)
    parser.add_argument("--collection-root", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--sampling-evidence-path", required=True)
    parser.add_argument("--expected-sampled-games", required=True, type=int)
    parser.add_argument("--expected-sampled-moves", required=True, type=int)
    parser.add_argument("--expected-full-games", required=True, type=int)
    parser.add_argument("--expected-full-moves", required=True, type=int)
    return parser


def _result_to_json(result: SampledSnapshotRelocationResult) -> dict[str, object]:
    return {
        "source_duckdb": result.source_duckdb.resolve().as_posix(),
        "target_duckdb": result.target_duckdb.resolve().as_posix(),
        "preserved_filename": result.preserved_filename,
        "replaced_views": list(result.replaced_views),
        "replaced_collection_roots": list(result.replaced_collection_roots),
        "counts": {
            "stg_games": result.stg_games_rows,
            "int_move_context": result.int_move_context_rows,
            "sampled_game_keys": result.sampled_game_keys_rows,
            "sampled_game_indices": result.sampled_game_indices_rows,
            "distinct_game_ids": result.distinct_game_ids,
            "moves_without_game": result.moves_without_game_rows,
        },
        "collection_id": result.collection_id,
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = relocate_sampled_snapshot(
        source_duckdb=Path(args.source_duckdb),
        target_directory=Path(args.target_directory),
        collection_root=Path(args.collection_root),
        collection_id=str(args.collection_id),
        sampling_evidence_path=Path(args.sampling_evidence_path),
        expected_sampled_games=int(args.expected_sampled_games),
        expected_sampled_moves=int(args.expected_sampled_moves),
        expected_full_games=int(args.expected_full_games),
        expected_full_moves=int(args.expected_full_moves),
    )
    print(json.dumps(_result_to_json(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
