"""Shared compact 1-D convolutional encoder.

With ~30 participants and a few thousand windows, capacity is the enemy. The
encoder is deliberately small: strided convolutions halve the temporal
resolution at each stage and a global average+max pool collapses time, giving a
fixed-size embedding regardless of window length.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int = 7, stride: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=kernel, stride=stride,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(cout),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalEncoder(nn.Module):
    """[B, Cin, T] -> [B, emb] where emb = 2 * channels (avg-pool ++ max-pool)."""

    def __init__(self, cin: int, channels: int = 32, depth: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        blocks = []
        c = cin
        for _ in range(depth):
            blocks.append(ConvBlock(c, channels, dropout=dropout))
            c = channels
        self.blocks = nn.Sequential(*blocks)
        self.out_dim = 2 * channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)


class RegressionHeads(nn.Module):
    """Separate heads for valence and arousal, as specified in the protocol."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        def head():
            return nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        self.valence = head()
        self.arousal = head()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.valence(z), self.arousal(z)], dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
