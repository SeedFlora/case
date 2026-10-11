"""Baseline 2: early-fusion temporal network.

All synchronised sensor channels are concatenated at the input and consumed by a
single encoder. This is the conventional approach and the one most exposed to a
missing sensor: an absent channel becomes a zero row that the first convolution
mixes into every filter, with no mechanism to discount it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import RegressionHeads, TemporalEncoder


class EarlyFusionNet(nn.Module):
    def __init__(self, n_modalities: int, channels: int = 96, depth: int = 4,
                 dropout: float = 0.2, **_):
        super().__init__()
        self.n_modalities = n_modalities
        self.encoder = TemporalEncoder(n_modalities, channels, depth, dropout)
        self.heads = RegressionHeads(self.encoder.out_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, q: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        # x [B, C, T]; absent modalities arrive already zeroed by the caller
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return self.heads(self.encoder(x))

    def gates(self, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return mask.float()
