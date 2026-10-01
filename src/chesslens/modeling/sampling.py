"""Deterministic hash-based sampling and holdout helpers."""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass
from typing import Any

_HEX_DESC_TRANSLATION = str.maketrans("0123456789abcdef", "fedcba9876543210")


def _descending_lex_key(token: str) -> str:
    """Return a key where smaller value means lexicographically larger token."""
    return token.lower().translate(_HEX_DESC_TRANSLATION)


@dataclass(frozen=True)
class HashRule:
    rule_version: str
    seed: str
    hash_modulus: int
    hash_threshold: int


def stable_hash_u64(payload: str) -> int:
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def stable_hash_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_mod_score(*, namespace: str, token: str, seed: str) -> int:
    return stable_hash_u64(f"{namespace}|{seed}|{token}")


def select_by_hash_mod(
    *,
    namespace: str,
    token: str,
    seed: str,
    hash_modulus: int,
    hash_threshold: int,
) -> tuple[bool, int]:
    score = hash_mod_score(namespace=namespace, token=token, seed=seed)
    return (score % hash_modulus) < hash_threshold, score


def effective_rate_percent(*, hash_modulus: int, hash_threshold: int) -> float:
    return round((hash_threshold / hash_modulus) * 100.0, 6)


def select_top_k_by_score(rows: list[dict[str, Any]], *, k: int) -> list[dict[str, Any]]:
    """Deterministically keep the k rows with smallest sample_score_u64.

    Uses O(k) memory and stable tie-breaking by game_id.
    """
    if k <= 0:
        return []

    heap: list[tuple[int, str, dict[str, Any]]] = []
    for row in rows:
        score = int(row["sample_score_u64"])
        game_id = str(row["game_id"])
        # The heap root tracks the current worst retained row.
        item = (-score, _descending_lex_key(game_id), row)
        if len(heap) < k:
            heapq.heappush(heap, item)
            continue

        if item > heap[0]:
            heapq.heapreplace(heap, item)

    selected = [entry[2] for entry in heap]
    selected.sort(key=lambda item: (int(item["sample_score_u64"]), str(item["game_id"])))
    return selected
