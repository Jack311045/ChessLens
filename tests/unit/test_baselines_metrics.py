from __future__ import annotations

import numpy as np

from chesslens.modeling.baselines_metrics import (
    RankingGroup,
    compute_ranking_metrics,
    expected_calibration_error,
    multiclass_brier_score,
    reliability_bins_per_class,
)


def test_compute_ranking_metrics_basic_case() -> None:
    groups = [
        RankingGroup(
            game_id="g1",
            group_id="g1:0",
            action_indices=(10, 20, 30),
            labels=(1, 0, 0),
            scores=(0.9, 0.2, 0.1),
            metadata={},
        ),
        RankingGroup(
            game_id="g2",
            group_id="g2:0",
            action_indices=(40, 50, 60),
            labels=(0, 1, 0),
            scores=(0.8, 0.7, 0.1),
            metadata={},
        ),
    ]

    metrics = compute_ranking_metrics(groups, cutoffs=(1, 3, 5))

    assert metrics["group_count"] == 2
    assert metrics["candidate_count"]["mean"] == 3.0
    assert metrics["candidate_count"]["max"] == 3
    assert metrics["candidate_count"]["min"] == 3
    assert metrics["recall_at_1"] == 0.5
    assert metrics["mrr"] == 0.75


def test_expected_calibration_error_and_bins_shape() -> None:
    y_true = np.asarray([0, 1, 2, 0], dtype=np.int32)
    probs = np.asarray(
        [
            [0.9, 0.05, 0.05],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.2, 0.2],
        ],
        dtype=np.float64,
    )

    ece, bins = expected_calibration_error(y_true, probs, bins=5)

    assert 0.0 <= ece <= 1.0
    assert len(bins) == 5
    assert all("left" in row and "right" in row for row in bins)


def test_multiclass_brier_and_reliability_bins() -> None:
    y_true = np.asarray([0, 1, 2], dtype=np.int32)
    probs = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
        ],
        dtype=np.float64,
    )

    brier = multiclass_brier_score(y_true, probs, class_count=3)
    rel = reliability_bins_per_class(
        y_true,
        probs,
        class_labels=("win", "draw", "loss"),
        bins=4,
    )

    assert 0.0 <= brier <= 1.0
    assert set(rel) == {"win", "draw", "loss"}
    assert all(len(payload) == 4 for payload in rel.values())
