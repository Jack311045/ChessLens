from __future__ import annotations

from pathlib import Path

import chess
import pytest
import zstandard

from chesslens.ingestion.pgn_reader import (
    GameParseError,
    hash_player_identifier,
    is_complete_game,
    iter_parsed_games,
    iter_raw_games,
    parse_game_to_records,
    stable_game_id,
)
from tests.conftest import write_zst_text


def _first_game_from_archive(path: Path) -> chess.pgn.Game:
    items = list(iter_raw_games(path, max_games=1))
    assert len(items) == 1
    return items[0][1]


def test_stable_game_id_is_deterministic() -> None:
    gid_a = stable_game_id("abc", 17)
    gid_b = stable_game_id("abc", 17)
    gid_c = stable_game_id("abc", 18)
    assert gid_a == gid_b
    assert gid_a != gid_c


def test_player_hash_deterministic_and_non_empty() -> None:
    assert hash_player_identifier("SomePlayer") == hash_player_identifier("someplayer")
    assert hash_player_identifier("") is None


def test_hmac_hash_is_stable_for_same_key() -> None:
    hash_a = hash_player_identifier("SomePlayer", secret_key="alpha", mode="hmac_sha256")
    hash_b = hash_player_identifier("someplayer", secret_key="alpha", mode="hmac_sha256")
    assert hash_a == hash_b


def test_hmac_hash_changes_with_key() -> None:
    hash_a = hash_player_identifier("SomePlayer", secret_key="alpha", mode="hmac_sha256")
    hash_b = hash_player_identifier("SomePlayer", secret_key="beta", mode="hmac_sha256")
    assert hash_a != hash_b


def test_hmac_mode_requires_secret_key() -> None:
    with pytest.raises(ValueError, match="secret key"):
        hash_player_identifier("SomePlayer", mode="hmac_sha256")


def test_empty_pgn_returns_no_games(tmp_path: Path) -> None:
    archive = write_zst_text(tmp_path, "")
    assert list(iter_raw_games(archive, max_games=10)) == []


def test_malformed_pgn_reports_parse_error(tmp_path: Path) -> None:
    game = chess.pgn.Game()
    game.errors.append(ValueError("malformed pgn token"))
    with pytest.raises(GameParseError, match="malformed"):
        parse_game_to_records(
            game=game,
            source_archive="bad.pgn.zst",
            source_archive_sha256="sha",
            source_game_index=0,
        )


def test_truncated_zstd_raises_error(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.pgn.zst"
    truncated.write_bytes(b"not-a-valid-zstd-frame")

    with pytest.raises(zstandard.ZstdError):
        list(iter_raw_games(truncated, max_games=10))


def test_missing_optional_headers_is_allowed(tmp_path: Path) -> None:
    pgn = "[Event \"Minimal\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0\n"
    archive = write_zst_text(tmp_path, pgn)
    game = _first_game_from_archive(archive)
    parsed = parse_game_to_records(
        game=game,
        source_archive="minimal.pgn.zst",
        source_archive_sha256="sha",
        source_game_index=0,
    )
    assert parsed.game_record.white_rating is None
    assert parsed.game_record.time_control_raw is None


def test_invalid_fen_rejected(tmp_path: Path) -> None:
    pgn = (
        "[Event \"InvalidFen\"]\n"
        "[Result \"1-0\"]\n"
        "[SetUp \"1\"]\n"
        "[FEN \"not-a-fen\"]\n\n"
        "1. e4 1-0\n"
    )
    archive = write_zst_text(tmp_path, pgn)
    with pytest.raises((GameParseError, ValueError), match="invalid|expected 8 rows"):
        game = _first_game_from_archive(archive)
        parse_game_to_records(
            game=game,
            source_archive="invalid-fen.pgn.zst",
            source_archive_sha256="sha",
            source_game_index=0,
        )


def test_illegal_move_rejected(tmp_path: Path) -> None:
    pgn = "[Event \"IllegalMove\"]\n[Result \"*\"]\n\n1. e5 *\n"
    archive = write_zst_text(tmp_path, pgn)
    game = _first_game_from_archive(archive)
    with pytest.raises(GameParseError, match="illegal san|Illegal move"):
        parse_game_to_records(
            game=game,
            source_archive="illegal.pgn.zst",
            source_archive_sha256="sha",
            source_game_index=0,
        )


def test_zero_game_limit(tmp_path: Path) -> None:
    archive = write_zst_text(tmp_path, "[Event \"A\"]\n\n1. e4 *\n")
    assert list(iter_raw_games(archive, max_games=0)) == []


def test_negative_limit_rejected(tmp_path: Path) -> None:
    archive = write_zst_text(tmp_path, "[Event \"A\"]\n\n1. e4 *\n")
    with pytest.raises(ValueError, match="non-negative"):
        list(iter_raw_games(archive, max_games=-1))


def test_unsupported_variant_rejected(tmp_path: Path) -> None:
    pgn = "[Event \"Variant\"]\n[Variant \"Crazyhouse\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0\n"
    archive = write_zst_text(tmp_path, pgn)
    game = _first_game_from_archive(archive)
    with pytest.raises(GameParseError, match="Unsupported variant"):
        parse_game_to_records(
            game=game,
            source_archive="variant.pgn.zst",
            source_archive_sha256="sha",
            source_game_index=0,
        )


def test_incomplete_final_game_detected(tmp_path: Path) -> None:
    pgn = "[Event \"Incomplete\"]\n[Result \"*\"]\n\n1. e4 e5 *\n"
    archive = write_zst_text(tmp_path, pgn)
    game = _first_game_from_archive(archive)
    parsed = parse_game_to_records(
        game=game,
        source_archive="incomplete.pgn.zst",
        source_archive_sha256="sha",
        source_game_index=0,
    )
    assert is_complete_game(parsed.game_record) is False


def test_duplicate_normalized_positions_deduplicated(tmp_path: Path) -> None:
    pgn = (
        "[Event \"Repeat\"]\n"
        "[Result \"1/2-1/2\"]\n\n"
        "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 1/2-1/2\n"
    )
    archive = write_zst_text(tmp_path, pgn)
    parsed = next(
        iter_parsed_games(
            archive_path=archive,
            max_games=1,
            strict=True,
            source_archive_sha256="sha",
        )
    )
    assert len(parsed.position_records) < len(parsed.move_records)
