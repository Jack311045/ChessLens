"""Create a small sanitized PGN fixture by streaming a large .pgn.zst source archive."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import chess.pgn
import zstandard

from chesslens.ingestion.config import load_ingestion_config
from chesslens.ingestion.pgn_reader import (
    GameParseError,
    compute_sha256,
    is_complete_game,
    iter_raw_games,
    parse_game_to_records,
)


def _clone_without_comments(game: chess.pgn.Game) -> chess.pgn.Game:
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    pgn_text = game.accept(exporter)
    cloned = chess.pgn.read_game(io.StringIO(pgn_text))
    if cloned is None:
        raise ValueError("Failed to clone PGN game during fixture generation")
    return cloned


def _sanitize_game_headers(game: chess.pgn.Game, fixture_index: int) -> chess.pgn.Game:
    sanitized = _clone_without_comments(game)

    sanitized.headers["White"] = f"white_player_{fixture_index:05d}"
    sanitized.headers["Black"] = f"black_player_{fixture_index:05d}"
    sanitized.headers["Site"] = f"https://example.invalid/game/{fixture_index:05d}"

    for key in [
        "WhiteTitle",
        "BlackTitle",
        "WhiteFideId",
        "BlackFideId",
        "WhiteTeam",
        "BlackTeam",
        "WhiteEloDiff",
        "BlackEloDiff",
    ]:
        if key in sanitized.headers:
            del sanitized.headers[key]

    return sanitized


def _write_fixture_pgn(games: list[chess.pgn.Game], output_pgn_path: Path) -> None:
    output_pgn_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pgn_path.open("w", encoding="utf-8", newline="\n") as stream:
        for game in games:
            print(game, file=stream, end="\n\n")


def _compress_pgn_to_zst(input_pgn_path: Path, output_zst_path: Path) -> None:
    output_zst_path.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstandard.ZstdCompressor(level=10)
    with input_pgn_path.open("rb") as source_stream:
        with output_zst_path.open("wb") as target_stream:
            with compressor.stream_writer(target_stream) as compressed_stream:
                while True:
                    chunk = source_stream.read(1024 * 1024)
                    if not chunk:
                        break
                    compressed_stream.write(chunk)


def create_fixture(
    *,
    config_path: Path,
    max_games_override: int | None = None,
    input_path_override: str | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if max_games_override is not None:
        overrides["max_games"] = max_games_override
    if input_path_override is not None:
        overrides["input_path"] = input_path_override

    config = load_ingestion_config(config_path, overrides=overrides)

    if config.max_games <= 0:
        raise ValueError("Fixture generation requires max_games > 0")
    if config.fixture_output_pgn is None or config.fixture_output_zst is None:
        raise ValueError("Fixture config must include fixture_output_pgn and fixture_output_zst")
    if config.fixture_manifest_path is None:
        raise ValueError("Fixture config must include fixture_manifest_path")

    source_sha256 = compute_sha256(config.input_path)
    source_archive_name = config.input_path.name

    selected_games: list[chess.pgn.Game] = []
    scanned_games = 0
    rejected_games = 0

    for source_game_index, raw_game in iter_raw_games(config.input_path, max_games=None):
        scanned_games += 1
        try:
            parsed = parse_game_to_records(
                game=raw_game,
                source_archive=source_archive_name,
                source_archive_sha256=source_sha256,
                source_game_index=source_game_index,
                schema_version=config.schema_version,
                run_id="fixture-generation",
            )
        except GameParseError:
            rejected_games += 1
            continue

        if not is_complete_game(parsed.game_record):
            rejected_games += 1
            continue

        selected_games.append(_sanitize_game_headers(raw_game, fixture_index=len(selected_games)))
        if len(selected_games) >= config.max_games:
            break

    if len(selected_games) < config.max_games:
        raise RuntimeError(
            f"Requested {config.max_games} complete valid games, found only {len(selected_games)}"
        )

    _write_fixture_pgn(selected_games, config.fixture_output_pgn)
    _compress_pgn_to_zst(config.fixture_output_pgn, config.fixture_output_zst)

    fixture_sha256 = compute_sha256(config.fixture_output_zst)

    manifest = {
        "source_archive": source_archive_name,
        "source_archive_sha256": source_sha256,
        "selection_rule": "first N complete valid games by source_game_index",
        "requested_games": config.max_games,
        "selected_games": len(selected_games),
        "scanned_games": scanned_games,
        "rejected_games": rejected_games,
        "fixture_pgn": str(config.fixture_output_pgn.relative_to(Path.cwd())),
        "fixture_pgn_zst": str(config.fixture_output_zst.relative_to(Path.cwd())),
        "fixture_sha256": fixture_sha256,
        "sanitization": {
            "player_names": "Replaced White/Black with deterministic placeholders",
            "site_urls": "Replaced Site with deterministic placeholder URLs",
            "stripped_headers": [
                "WhiteTitle",
                "BlackTitle",
                "WhiteFideId",
                "BlackFideId",
                "WhiteTeam",
                "BlackTeam",
                "WhiteEloDiff",
                "BlackEloDiff",
            ],
            "comments_and_variations": (
                "Removed while cloning to keep fixture compact and non-identifying"
            ),
        },
    }

    config.fixture_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.fixture_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a tiny sanitized PGN fixture")
    parser.add_argument("--config", default="configs/ingestion/fixture.yaml")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--input-path", default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    manifest = create_fixture(
        config_path=Path(args.config),
        max_games_override=args.max_games,
        input_path_override=args.input_path,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()