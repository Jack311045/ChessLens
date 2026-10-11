"""Phase 3.1 residual CNN policy/value network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from chesslens.features.action_encoding import ACTION_PLANES, ACTION_SPACE_SIZE


def flatten_policy_spatial_logits(spatial_logits: torch.Tensor) -> torch.Tensor:
    """Flatten [B,73,8,8] into [B,4672] using encoder-compatible order."""
    if spatial_logits.ndim != 4:
        raise ValueError("policy spatial logits must be rank-4 tensor")
    if spatial_logits.shape[1:] != (ACTION_PLANES, 8, 8):
        raise ValueError("policy spatial logits must have shape [B,73,8,8]")
    flattened = spatial_logits.permute(0, 2, 3, 1).reshape(
        spatial_logits.shape[0],
        ACTION_SPACE_SIZE,
    )
    return flattened


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = output + residual
        output = self.relu(output)
        return cast(torch.Tensor, output)


@dataclass(frozen=True)
class NeuralModelSizes:
    continuous_features: int
    rating_band_vocab_size: int
    time_control_vocab_size: int


class PolicyValueResidualCNN(nn.Module):
    def __init__(
        self,
        *,
        trunk_channels: int,
        residual_blocks: int,
        context_embedding_dim: int,
        context_hidden_dim: int,
        value_hidden_dim: int,
        sizes: NeuralModelSizes,
    ) -> None:
        super().__init__()
        self.sizes = sizes

        self.stem = nn.Sequential(
            nn.Conv2d(18, trunk_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True),
        )
        self.residual_stack = nn.Sequential(
            *(ResidualBlock(trunk_channels) for _ in range(residual_blocks))
        )

        self.rating_band_embedding = nn.Embedding(
            sizes.rating_band_vocab_size,
            context_embedding_dim,
        )
        self.time_control_embedding = nn.Embedding(
            sizes.time_control_vocab_size,
            context_embedding_dim,
        )
        self.continuous_projection = nn.Sequential(
            nn.Linear(sizes.continuous_features, context_hidden_dim),
            nn.ReLU(inplace=True),
        )

        combined_context_dim = context_hidden_dim + (2 * context_embedding_dim)
        self.trunk_context_projection = nn.Linear(combined_context_dim, trunk_channels)

        self.policy_head = nn.Conv2d(
            trunk_channels,
            ACTION_PLANES,
            kernel_size=1,
            bias=True,
        )

        self.value_head = nn.Sequential(
            nn.Linear(trunk_channels + combined_context_dim, value_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden_dim, 3),
        )

    def forward(
        self,
        board_tensor: torch.Tensor,
        continuous_context: torch.Tensor,
        rating_band_index: torch.Tensor,
        time_control_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if board_tensor.ndim != 4:
            raise ValueError("board_tensor must be [B,18,8,8]")

        trunk = self.stem(board_tensor)
        trunk = self.residual_stack(trunk)

        cont_embed = self.continuous_projection(continuous_context)
        rating_embed = self.rating_band_embedding(rating_band_index)
        tc_embed = self.time_control_embedding(time_control_index)
        context = torch.cat([cont_embed, rating_embed, tc_embed], dim=1)

        trunk_bias = self.trunk_context_projection(context).unsqueeze(-1).unsqueeze(-1)
        trunk_with_context = trunk + trunk_bias

        policy_spatial = self.policy_head(trunk_with_context)
        policy_logits = flatten_policy_spatial_logits(policy_spatial)

        pooled = trunk_with_context.mean(dim=(2, 3))
        value_input = torch.cat([pooled, context], dim=1)
        value_logits = self.value_head(value_input)

        return policy_logits, value_logits


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
