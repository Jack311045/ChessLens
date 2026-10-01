from __future__ import annotations

from chesslens.modeling.sampling import (
    effective_rate_percent,
    select_by_hash_mod,
    select_top_k_by_score,
    stable_hash_u64,
)


def test_stable_hash_u64_is_deterministic() -> None:
    assert stable_hash_u64("abc") == stable_hash_u64("abc")
    assert stable_hash_u64("abc") != stable_hash_u64("abcd")


def test_select_by_hash_mod_is_deterministic() -> None:
    first = select_by_hash_mod(
        namespace="game_sampling",
        token="game-123",
        seed="seed-a",
        hash_modulus=10000,
        hash_threshold=1000,
    )
    second = select_by_hash_mod(
        namespace="game_sampling",
        token="game-123",
        seed="seed-a",
        hash_modulus=10000,
        hash_threshold=1000,
    )
    assert first == second


def test_effective_rate_percent() -> None:
    assert effective_rate_percent(hash_modulus=10000, hash_threshold=1000) == 10.0


def test_select_top_k_by_score_deterministic_tie_break() -> None:
    rows = [
        {"game_id": "b", "sample_score_u64": 7},
        {"game_id": "a", "sample_score_u64": 7},
        {"game_id": "c", "sample_score_u64": 8},
    ]

    selected = select_top_k_by_score(rows, k=1)

    assert len(selected) == 1
    assert selected[0]["game_id"] == "a"


def test_select_top_k_by_score_sorted_output() -> None:
    rows = [
        {"game_id": "g3", "sample_score_u64": 30},
        {"game_id": "g2", "sample_score_u64": 20},
        {"game_id": "g1", "sample_score_u64": 10},
    ]

    selected = select_top_k_by_score(rows, k=2)

    assert [item["game_id"] for item in selected] == ["g1", "g2"]


def test_select_top_k_with_zero_k_is_empty() -> None:
    rows = [{"game_id": "g1", "sample_score_u64": 1}]
    assert select_top_k_by_score(rows, k=0) == []
