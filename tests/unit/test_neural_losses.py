from __future__ import annotations

import pytest
import torch

from chesslens.modeling.neural_losses import (
    LossWeights,
    compute_supervised_losses,
    mask_policy_logits,
)


def test_mask_policy_logits_sets_illegal_logits_to_large_negative() -> None:
    logits = torch.tensor([[0.1, 2.5, -0.3, 1.1]], dtype=torch.float32)
    legal_mask = torch.tensor([[True, False, True, False]], dtype=torch.bool)

    masked = mask_policy_logits(logits, legal_mask)

    assert masked.shape == logits.shape
    assert float(masked[0, 0]) == pytest.approx(0.1)
    assert float(masked[0, 2]) == pytest.approx(-0.3)
    assert float(masked[0, 1]) == pytest.approx(-1.0e9)
    assert float(masked[0, 3]) == pytest.approx(-1.0e9)


def test_compute_supervised_losses_rejects_illegal_policy_target() -> None:
    policy_logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    value_logits = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    legal_mask = torch.tensor([[True, False, True]], dtype=torch.bool)
    policy_targets = torch.tensor([1], dtype=torch.int64)
    value_targets = torch.tensor([2], dtype=torch.int64)

    with pytest.raises(ValueError, match="not legal"):
        compute_supervised_losses(
            policy_logits=policy_logits,
            value_logits=value_logits,
            legal_mask=legal_mask,
            policy_targets=policy_targets,
            value_targets=value_targets,
            weights=LossWeights(policy=1.0, value=0.4),
        )
