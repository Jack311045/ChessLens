from __future__ import annotations

import io
from pathlib import Path

import pytest
import zstandard

from chesslens.ingestion.pgn_boundaries import (
    PgnBoundaryError,
    count_games_in_zst,
    iter_game_bytes,
    iter_game_bytes_from_zst,
)
from tests.conftest import write_zst_text


def _game(index: int, *, result: str = "1-0", extra_moves: str = "1. e4 e5 2. Qh5") -> str:
    return (
        f'[Event "Rated Classical game {index}"]\n'
        f'[Site "https://example.invalid/game/{index:05d}"]\n'
        f'[Result "{result}"]\n'
        f'[Opening "Weird [bracket] name"]\n'
        "\n"
        f"{extra_moves} {result}\n"
    )


def _archive_text(count: int, *, line_ending: str = "\n") -> str:
    text = "\n".join(_game(i) for i in range(count)) + "\n"
    if line_ending != "\n":
        text = text.replace("\n", line_ending)
    return text


def _split_bytes(text: str) -> list[bytes]:
    return list(iter_game_bytes(io.BytesIO(text.encode("utf-8")), chunk_size=17))


def test_splits_expected_game_count() -> None:
    games = _split_bytes(_archive_text(5))
    assert len(games) == 5
    for i, game in enumerate(games):
        assert game.startswith(b"[Event ")
        assert f"game {i}".encode() in game


def test_single_game_archive() -> None:
    games = _split_bytes(_game(0))
    assert len(games) == 1


def test_empty_archive_yields_nothing() -> None:
    assert _split_bytes("") == []
    assert _split_bytes("\n\n   \n") == []


def test_comment_and_header_brackets_do_not_fool_split() -> None:
    tricky = (
        '[Event "g0"]\n[Result "1-0"]\n\n'
        "1. e4 { annotation with [Event fake] and [Site x] } e5 1-0\n"
    )
    text = tricky + "\n" + _game(1)
    games = _split_bytes(text)
    assert len(games) == 2
    assert b"[Event fake]" in games[0]


def test_crlf_line_endings_supported() -> None:
    games = _split_bytes(_archive_text(3, line_ending="\r\n"))
    assert len(games) == 3
    assert games[0].startswith(b"[Event ")


def test_oversized_single_game_guard() -> None:
    big = '[Event "g"]\n[Result "1-0"]\n\n' + ("1. e4 e5 " * 100000) + "1-0\n"
    with pytest.raises(PgnBoundaryError, match="max_game_bytes"):
        list(iter_game_bytes(io.BytesIO(big.encode("utf-8")), max_game_bytes=1024))


def test_zst_round_trip_count(tmp_path: Path) -> None:
    archive = write_zst_text(tmp_path, _archive_text(6), filename="mini.pgn.zst")
    assert count_games_in_zst(archive) == 6
    games = list(iter_game_bytes_from_zst(archive))
    assert len(games) == 6


def test_exactly_divisible_and_partial_counts() -> None:
    assert len(_split_bytes(_archive_text(6))) == 6
    assert len(_split_bytes(_archive_text(7))) == 7


def test_concatenated_games_preserve_order_and_bytes() -> None:
    text = _archive_text(4)
    games = _split_bytes(text)
    reconstructed = b"".join(games)
    assert reconstructed == text.encode("utf-8")


def test_corrupt_zstd_input_raises(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pgn.zst"
    corrupt.write_bytes(b"this is not a valid zstd frame")
    with pytest.raises(zstandard.ZstdError):
        count_games_in_zst(corrupt)
