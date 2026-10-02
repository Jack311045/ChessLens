"""Metrics helpers for Phase 2.2 ranking and multiclass calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RankingGroup:
    game_id: str
    group_id: str
    action_indices: tuple[int, ...]
    labels: tuple[int, ...]
    scores: tuple[float, ...]
    metadata: dict[str, Any]


def _validate_group(group: RankingGroup) -> None:
    if len(group.action_indices) == 0:
        raise ValueError(f"Ranking group {group.group_id!r} has no candidates")
    if len(group.labels) != len(group.action_indices):
        raise ValueError(f"Ranking group {group.group_id!r} has mismatched labels")
    if len(group.scores) != len(group.action_indices):
        raise ValueError(f"Ranking group {group.group_id!r} has mismatched scores")
    positives = sum(1 for item in group.labels if item == 1)
    if positives != 1:
        raise ValueError(
            f"Ranking group {group.group_id!r} must contain exactly one positive label"
        )


def _order_indices(group: RankingGroup) -> list[int]:
    return sorted(
        range(len(group.scores)),
        key=lambda idx: (-group.scores[idx], group.action_indices[idx]),
    )


def _positive_rank(group: RankingGroup) -> int:
    _validate_group(group)
    ordered = _order_indices(group)
    positive_index = next(idx for idx, label in enumerate(group.labels) if label == 1)
    return ordered.index(positive_index) + 1


def compute_ranking_metrics(groups: list[RankingGroup], cutoffs: tuple[int, ...]) -> dict[str, Any]:
    if not groups:
        raise ValueError("Ranking metrics require at least one group")
    if not cutoffs:
        raise ValueError("Ranking metrics require at least one cutoff")

    ranks: list[int] = []
    candidate_counts: list[int] = []
    for group in groups:
        _validate_group(group)
        ranks.append(_positive_rank(group))
        candidate_counts.append(len(group.action_indices))

    metrics: dict[str, Any] = {
        "group_count": len(groups),
        "legal_candidate_coverage": 1.0,
        "candidate_count": {
            "mean": float(np.mean(candidate_counts)),
            "p50": float(np.percentile(candidate_counts, 50)),
            "p90": float(np.percentile(candidate_counts, 90)),
            "p95": float(np.percentile(candidate_counts, 95)),
            "max": int(np.max(candidate_counts)),
            "min": int(np.min(candidate_counts)),
        },
        "mrr": float(np.mean([1.0 / rank for rank in ranks])),
    }

    for cutoff in cutoffs:
        recall = np.mean([1.0 if rank <= cutoff else 0.0 for rank in ranks])
        ndcg = np.mean(
            [
                (1.0 / math.log2(rank + 1.0)) if rank <= cutoff else 0.0
                for rank in ranks
            ]
        )
        metrics[f"recall_at_{cutoff}"] = float(recall)
        metrics[f"ndcg_at_{cutoff}"] = float(ndcg)

    return metrics


def multiclass_brier_score(
    y_true_indices: np.ndarray,
    probs: np.ndarray,
    class_count: int,
) -> float:
    if probs.ndim != 2 or probs.shape[1] != class_count:
        raise ValueError("Probability matrix shape does not match class_count")
    if probs.shape[0] != y_true_indices.shape[0]:
        raise ValueError("Probability rows do not match y_true length")

    targets = np.zeros_like(probs)
    targets[np.arange(y_true_indices.shape[0]), y_true_indices] = 1.0
    squared_error = np.square(probs - targets)
    return float(np.mean(np.sum(squared_error, axis=1)))


def expected_calibration_error(
    y_true_indices: np.ndarray,
    probs: np.ndarray,
    *,
    bins: int,
) -> tuple[float, list[dict[str, float | int]]]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D matrix")

    predicted = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correct = (predicted == y_true_indices).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    summaries: list[dict[str, float | int]] = []
    total_weighted_gap = 0.0

    for idx in range(bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if idx == bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)

        count = int(np.sum(mask))
        if count == 0:
            summaries.append(
                {
                    "bin_index": idx,
                    "left": float(left),
                    "right": float(right),
                    "count": 0,
                    "avg_confidence": 0.0,
                    "accuracy": 0.0,
                    "gap": 0.0,
                }
            )
            continue

        avg_conf = float(np.mean(confidence[mask]))
        accuracy = float(np.mean(correct[mask]))
        gap = abs(avg_conf - accuracy)
        total_weighted_gap += (count / len(confidence)) * gap

        summaries.append(
            {
                "bin_index": idx,
                "left": float(left),
                "right": float(right),
                "count": count,
                "avg_confidence": avg_conf,
                "accuracy": accuracy,
                "gap": float(gap),
            }
        )

    return float(total_weighted_gap), summaries


def reliability_bins_per_class(
    y_true_indices: np.ndarray,
    probs: np.ndarray,
    *,
    class_labels: tuple[str, ...],
    bins: int,
) -> dict[str, list[dict[str, float | int]]]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D matrix")
    if probs.shape[1] != len(class_labels):
        raise ValueError("probs class dimension does not match class_labels")

    edges = np.linspace(0.0, 1.0, bins + 1)
    output: dict[str, list[dict[str, float | int]]] = {}

    for class_idx, label in enumerate(class_labels):
        class_probs = probs[:, class_idx]
        class_truth = (y_true_indices == class_idx).astype(np.float64)
        rows: list[dict[str, float | int]] = []

        for bin_idx in range(bins):
            left = edges[bin_idx]
            right = edges[bin_idx + 1]
            if bin_idx == bins - 1:
                mask = (class_probs >= left) & (class_probs <= right)
            else:
                mask = (class_probs >= left) & (class_probs < right)

            count = int(np.sum(mask))
            if count == 0:
                rows.append(
                    {
                        "bin_index": bin_idx,
                        "left": float(left),
                        "right": float(right),
                        "count": 0,
                        "avg_probability": 0.0,
                        "empirical_frequency": 0.0,
                    }
                )
                continue

            rows.append(
                {
                    "bin_index": bin_idx,
                    "left": float(left),
                    "right": float(right),
                    "count": count,
                    "avg_probability": float(np.mean(class_probs[mask])),
                    "empirical_frequency": float(np.mean(class_truth[mask])),
                }
            )

        output[label] = rows

    return output
