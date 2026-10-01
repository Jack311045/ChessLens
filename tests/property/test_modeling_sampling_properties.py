from __future__ import annotations

from typing import cast

import hypothesis.strategies as st
from hypothesis import given, settings

from chesslens.modeling.sampling import select_by_hash_mod, select_top_k_by_score


def _row_sort_key(item: dict[str, object]) -> tuple[int, str]:
    score = cast(int, item["sample_score_u64"])
    game_id = cast(str, item["game_id"])
    return score, game_id


@given(
    token=st.text(min_size=1, max_size=40),
    seed=st.text(min_size=1, max_size=20),
)
@settings(max_examples=120)
def test_hash_selection_is_deterministic(token: str, seed: str) -> None:
    first = select_by_hash_mod(
        namespace="game_sampling",
        token=token,
        seed=seed,
        hash_modulus=10000,
        hash_threshold=1234,
    )
    second = select_by_hash_mod(
        namespace="game_sampling",
        token=token,
        seed=seed,
        hash_modulus=10000,
        hash_threshold=1234,
    )
    assert first == second


@given(
    raw_rows=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=200),
            st.integers(min_value=0, max_value=2**63 - 1),
        ),
        min_size=1,
        max_size=60,
        unique_by=lambda item: item[0],
    ),
    k=st.integers(min_value=0, max_value=60),
)
@settings(max_examples=120)
def test_top_k_matches_sorted_baseline(raw_rows: list[tuple[int, int]], k: int) -> None:
    rows = [
        {
            "game_id": f"g{idx}",
            "sample_score_u64": score,
        }
        for idx, score in raw_rows
    ]

    selected = select_top_k_by_score(rows, k=k)
    expected = sorted(rows, key=_row_sort_key)[:k]

    assert selected == expected
