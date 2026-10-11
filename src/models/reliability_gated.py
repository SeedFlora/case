"""Phase 7: the proposed reliability-gated fusion model.

Per-modality encoders produce embeddings h_m. A small MLP maps each modality's
label-free signal-quality vector q_m to a scalar reliability weight

    g_m = sigmoid(MLP(q_m))

and fusion is a normalised, mask-aware weighted average

    z = sum_m (a_m * g_m * h_m) / (sum_m (a_m * g_m) + eps)

where a_m in {0,1} is the availability mask. Because a_m multiplies both
numerator and denominator, an ABSENT modality contributes nothing and does not
dilute the average -- which is precisely the failure mode of early fusion, where
a zeroed channel still occupies its slot in every convolution.

Ablation switches (`use_quality`, `learned_gates`) let the same class serve as
its own controls with identical capacity.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import RegressionHeads, TemporalEncoder

EPS = 1e-6


class QualityGate(nn.Module):
    """q_m -> g_m in (0, 1). One small MLP per modality."""

    def __init__(self, n_quality: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_quality, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        # Start near g=1 so training begins close to equal-weight fusion and the
        # gate has to *earn* any deviation from it.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 2.0)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(q)).squeeze(-1)


class ReliabilityGatedNet(nn.Module):
    def __init__(self, n_modalities: int, n_quality: int = 10,
                 enc_channels: int = 32, depth: int = 4, dropout: float = 0.2,
                 gate_hidden: int = 16, use_quality: bool = True,
                 learned_gates: bool = True, **_):
        super().__init__()
        self.n_modalities = n_modalities
        self.n_quality = n_quality
        self.use_quality = use_quality
        self.learned_gates = learned_gates

        self.encoders = nn.ModuleList([
            TemporalEncoder(1, enc_channels, depth, dropout) for _ in range(n_modalities)
        ])
        self.emb_dim = self.encoders[0].out_dim
        if learned_gates:
            self.gate_mlps = nn.ModuleList([
                QualityGate(n_quality, gate_hidden) for _ in range(n_modalities)
            ])
        else:
            self.gate_mlps = None
        self.heads = RegressionHeads(self.emb_dim, dropout=dropout)

    # ------------------------------------------------------------------ #
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        hs = [enc(x[:, i:i + 1, :]) for i, enc in enumerate(self.encoders)]
        return torch.stack(hs, dim=1)                       # [B, C, E]

    def gates(self, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reliability weights [B, C]; absent modalities are forced to zero."""
        b, c = mask.shape
        if not self.learned_gates:
            g = torch.ones(b, c, device=mask.device, dtype=torch.float32)
        else:
            qq = q if self.use_quality else torch.zeros_like(q)
            g = torch.stack([self.gate_mlps[i](qq[:, i, :]) for i in range(c)], dim=1)
        return g * mask.float()

    def forward(self, x: torch.Tensor, q: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.embed(x)                                   # [B, C, E]
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=torch.float32)
        g = self.gates(q, mask).unsqueeze(-1)               # [B, C, 1]
        z = (g * h).sum(dim=1) / (g.sum(dim=1) + EPS)
        return self.heads(z)
