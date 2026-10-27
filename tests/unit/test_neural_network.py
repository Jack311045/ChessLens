from __future__ import annotations

import torch

from chesslens.features.action_encoding import ACTION_SPACE_SIZE
from chesslens.modeling.neural_losses import LossWeights, compute_supervised_losses
from chesslens.modeling.neural_network import (
    NeuralModelSizes,
    PolicyValueResidualCNN,
    flatten_policy_spatial_logits,
)


def test_flatten_policy_spatial_logits_matches_action_layout() -> None:
    spatial = torch.zeros((1, 73, 8, 8), dtype=torch.float32)
    spatial[0, 1, 1, 4] = 42.0  # e2 square, north 2-plane -> e2e4 index 877

    flattened = flatten_policy_spatial_logits(spatial)

    assert flattened.shape == (1, ACTION_SPACE_SIZE)
    assert float(flattened[0, 877]) == 42.0


def test_policy_value_forward_shapes() -> None:
    model = PolicyValueResidualCNN(
        trunk_channels=32,
        residual_blocks=2,
        context_embedding_dim=8,
        context_hidden_dim=16,
        value_hidden_dim=32,
        sizes=NeuralModelSizes(
            continuous_features=8,
            rating_band_vocab_size=4,
            time_control_vocab_size=3,
        ),
    )

    board = torch.zeros((5, 18, 8, 8), dtype=torch.float32)
    continuous = torch.zeros((5, 8), dtype=torch.float32)
    rating_band = torch.zeros((5,), dtype=torch.int64)
    time_control = torch.zeros((5,), dtype=torch.int64)

    policy_logits, value_logits = model(board, continuous, rating_band, time_control)

    assert policy_logits.shape == (5, ACTION_SPACE_SIZE)
    assert value_logits.shape == (5, 3)


def test_optimizer_step_updates_model_parameters() -> None:
    torch.manual_seed(7)
    model = PolicyValueResidualCNN(
        trunk_channels=16,
        residual_blocks=1,
        context_embedding_dim=4,
        context_hidden_dim=8,
        value_hidden_dim=16,
        sizes=NeuralModelSizes(
            continuous_features=8,
            rating_band_vocab_size=3,
            time_control_vocab_size=3,
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    board = torch.randn((4, 18, 8, 8), dtype=torch.float32)
    continuous = torch.randn((4, 8), dtype=torch.float32)
    rating_band = torch.tensor([0, 1, 2, 0], dtype=torch.int64)
    time_control = torch.tensor([0, 1, 2, 1], dtype=torch.int64)
    legal_mask = torch.ones((4, ACTION_SPACE_SIZE), dtype=torch.bool)
    policy_targets = torch.tensor([12, 100, 877, 1200], dtype=torch.int64)
    value_targets = torch.tensor([0, 1, 2, 1], dtype=torch.int64)

    before = [param.detach().clone() for param in model.parameters()]

    policy_logits, value_logits = model(board, continuous, rating_band, time_control)
    losses = compute_supervised_losses(
        policy_logits=policy_logits,
        value_logits=value_logits,
        legal_mask=legal_mask,
        policy_targets=policy_targets,
        value_targets=value_targets,
        weights=LossWeights(policy=1.0, value=0.4),
    )
    losses.total.backward()
    optimizer.step()

    changed = False
    for old, new in zip(before, model.parameters(), strict=True):
        if not torch.allclose(old, new.detach()):
            changed = True
            break

    assert changed is True
