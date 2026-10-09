"""Loss helpers for Phase 3.1 policy/value supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class LossWeights:
    policy: float
    value: float


@dataclass(frozen=True)
class LossBundle:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor


def validate_loss_weights(weights: LossWeights) -> None:
    if weights.policy < 0.0 or weights.value < 0.0:
        raise ValueError("Loss weights must be non-negative")
    if (weights.policy + weights.value) <= 0.0:
        raise ValueError("Loss weights must have positive total")


def mask_policy_logits(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if policy_logits.ndim != 2:
        raise ValueError("policy_logits must be rank-2 [B,4672]")
    if legal_mask.ndim != 2:
        raise ValueError("legal_mask must be rank-2 [B,4672]")
    if policy_logits.shape != legal_mask.shape:
        raise ValueError("policy_logits and legal_mask must share shape")

    if not torch.all(legal_mask.any(dim=1)):
        raise ValueError("Each row must contain at least one legal action")

    masked = policy_logits.masked_fill(~legal_mask, -1.0e9)
    return masked


def _validate_targets_are_legal(legal_mask: torch.Tensor, policy_targets: torch.Tensor) -> None:
    rows = torch.arange(policy_targets.shape[0], device=policy_targets.device)
    legal_targets = legal_mask[rows, policy_targets]
    if not torch.all(legal_targets):
        illegal_rows = rows[~legal_targets].detach().cpu().tolist()
        raise ValueError(
            "Policy target action index is not legal in mask for row(s): "
            + ", ".join(str(int(item)) for item in illegal_rows[:5])
        )


def compute_supervised_losses(
    *,
    policy_logits: torch.Tensor,
    value_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    weights: LossWeights,
) -> LossBundle:
    validate_loss_weights(weights)

    masked_policy_logits = mask_policy_logits(policy_logits, legal_mask)
    _validate_targets_are_legal(legal_mask, policy_targets)

    policy_loss = functional.cross_entropy(masked_policy_logits, policy_targets)
    value_loss = functional.cross_entropy(value_logits, value_targets)

    total = (weights.policy * policy_loss) + (weights.value * value_loss)
    return LossBundle(total=total, policy=policy_loss, value=value_loss)
