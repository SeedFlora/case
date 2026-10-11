"""Baseline 3: equal-weight late fusion.

One encoder per modality, fused by a plain masked average. This is the correct
control for the proposed model: it has the same per-modality encoder structure
and the same masking behaviour, and differs ONLY in that its fusion weights are
fixed and equal rather than driven by signal quality. Any difference between
this model and the reliability-gated model is therefore attributable to the
gate, not to the architecture around it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import RegressionHeads, TemporalEncoder

EPS = 1e-6


class LateFusionNet(nn.Module):
    def __init__(self, n_modalities: int, enc_channels: int = 32, depth: int = 4,
                 dropout: float = 0.2, **_):
        super().__init__()
        self.n_modalities = n_modalities
        self.encoders = nn.ModuleList([
            TemporalEncoder(1, enc_channels, depth, dropout) for _ in range(n_modalities)
        ])
        self.emb_dim = self.encoders[0].out_dim
        self.heads = RegressionHeads(self.emb_dim, dropout=dropout)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] -> [B, C, E]"""
        hs = [enc(x[:, i:i + 1, :]) for i, enc in enumerate(self.encoders)]
        return torch.stack(hs, dim=1)

    def forward(self, x: torch.Tensor, q: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.embed(x)                                   # [B, C, E]
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        w = mask.float().unsqueeze(-1)                      # [B, C, 1]
        z = (w * h).sum(dim=1) / (w.sum(dim=1) + EPS)       # masked mean
        return self.heads(z)

    def gates(self, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return mask.float()
