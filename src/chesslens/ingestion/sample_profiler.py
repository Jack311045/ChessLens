"""Small bounded sample profiler for validating schema assumptions over real data."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from chesslens.ingestion.config import load_ingestion_config
from chesslens.ingestion.pgn_reader import compute_sha256, iter_parsed_games


def _rating_range(min_rating: int | None, max_rating: int | None) -> dict[str, int | None]:
    return {"min": min_rating, "max": max_rating}


def profile_archive(
    *,
    config_path: Path,
    input_path_override: str | None = None,
    max_games_override: int | None = None,
    strict_override: bool | None = None,
    output_override: str | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if input_path_override is not None:
        overrides["input_path"] = input_path_override
    if max_games_override is not None:
        overrides["max_games"] = max_games_override
    if strict_override is not None:
        overrides["strict"] = strict_override
    if output_override is not None:
        overrides["profile_output_path"] = output_override

    config = load_ingestion_config(config_path, overrides=overrides)

    start = time.perf_counter()
    source_sha256 = compute_sha256(config.input_path)

    errors_by_code: Counter[str] = Counter()
    missing_headers: Counter[str] = Counter()
    result_distribution: Counter[str] = Counter()

    scanned_games = 0
    valid_games = 0
    invalid_games = 0
    emitted_moves = 0
    distinct_positions: set[str] = set()
    rating_min: int | None = None
    rating_max: int | None = None
    time_control_present = 0
    eco_present = 0
    opening_present = 0

    def on_error(error_record: Any) -> None:
        nonlocal invalid_games
        invalid_games += 1
        errors_by_code[str(error_record.error_code)] += 1

    for parsed in iter_parsed_games(
        archive_path=config.input_path,
        max_games=config.max_games,
        strict=config.strict,
        schema_version=config.schema_version,
        source_archive_sha256=source_sha256,
        run_id="sample-profile",
        on_error=on_error,
    ):
        scanned_games += 1
        valid_games += 1
        game = parsed.game_record
        emitted_moves += len(parsed.move_records)
        distinct_positions.update(position.position_id for position in parsed.position_records)

        if game.event is None:
            missing_headers["event"] += 1
        if game.played_date is None:
            missing_headers["played_date"] += 1
        if game.round is None:
            missing_headers["round"] += 1
        if game.result is None:
            missing_headers["result"] += 1
        if game.white_rating is None:
            missing_headers["white_rating"] += 1
        if game.black_rating is None:
            missing_headers["black_rating"] += 1
        if game.time_control_raw is None:
            missing_headers["time_control_raw"] += 1
        if game.eco is None:
            missing_headers["eco"] += 1
        if game.opening is None:
            missing_headers["opening"] += 1
        if game.termination is None:
            missing_headers["termination"] += 1

        result_distribution[game.result or "missing"] += 1

        for rating in (game.white_rating, game.black_rating):
            if rating is None:
                continue
            rating_min = rating if rating_min is None else min(rating_min, rating)
            rating_max = rating if rating_max is None else max(rating_max, rating)

        if game.time_control_raw is not None:
            time_control_present += 1
        if game.eco is not None:
            eco_present += 1
        if game.opening is not None:
            opening_present += 1

    duration_seconds = time.perf_counter() - start
    total_scanned = valid_games + invalid_games

    profile = {
        "profile_kind": "functional_sample_profile",
        "note": (
            "This profile is for bounded functional validation only; "
            "it is not a population-level analysis."
        ),
        "source_archive": config.input_path.name,
        "source_archive_sha256": source_sha256,
        "max_games_requested": config.max_games,
        "strict_mode": config.strict,
        "games_scanned": total_scanned,
        "valid_games": valid_games,
        "invalid_games": invalid_games,
        "moves": emitted_moves,
        "missing_header_counts": dict(missing_headers),
        "result_distribution": dict(result_distribution),
        "rating_range": _rating_range(rating_min, rating_max),
        "time_control_present_games": time_control_present,
        "eco_present_games": eco_present,
        "opening_present_games": opening_present,
        "distinct_normalized_positions": len(distinct_positions),
        "error_counts": dict(errors_by_code),
        "duration_seconds": round(duration_seconds, 4),
    }

    output_path = config.profile_output_path or (
        Path.cwd() / "reports" / "sample_profile.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    return profile


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile a bounded sample from a .pgn.zst archive")
    parser.add_argument("--config", default="configs/ingestion/smoke.yaml")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--tolerant", action="store_true")
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.strict and args.tolerant:
        raise ValueError("Choose either --strict or --tolerant, not both")

    strict_override: bool | None = None
    if args.strict:
        strict_override = True
    if args.tolerant:
        strict_override = False

    profile = profile_archive(
        config_path=Path(args.config),
        input_path_override=args.input_path,
        max_games_override=args.max_games,
        strict_override=strict_override,
        output_override=args.output,
    )
    print(json.dumps(profile, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()