from __future__ import annotations

import io

from hypothesis import given
from hypothesis import strategies as st

from chesslens.ingestion.pgn_boundaries import iter_game_bytes


def _game(index: int) -> str:
    return (
        f'[Event "Game {index}"]\n'
        f'[Result "1-0"]\n'
        "\n"
        f"1. e4 e5 2. Qh5 {{ comment [Event trap] }} Nc6 1-0\n"
    )


def _archive_text(count: int) -> str:
    if count == 0:
        return ""
    return "\n".join(_game(i) for i in range(count)) + "\n"


@given(count=st.integers(min_value=0, max_value=40), chunk=st.integers(min_value=1, max_value=64))
def test_boundary_count_matches_game_count(count: int, chunk: int) -> None:
    text = _archive_text(count)
    games = list(iter_game_bytes(io.BytesIO(text.encode("utf-8")), chunk_size=chunk))
    assert len(games) == count
    for game in games:
        assert game.startswith(b"[Event ")


@given(count=st.integers(min_value=1, max_value=25), chunk=st.integers(min_value=1, max_value=64))
def test_boundary_split_is_lossless(count: int, chunk: int) -> None:
    text = _archive_text(count).encode("utf-8")
    games = list(iter_game_bytes(io.BytesIO(text), chunk_size=chunk))
    assert b"".join(games) == text
