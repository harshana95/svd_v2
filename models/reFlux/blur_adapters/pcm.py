"""Perceptual Cross-Modulation (FiLM) from JEPA connector tokens."""

from __future__ import annotations

import torch
import torch.nn as nn

from .modulation import BlockModulation


class PCMModule(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * dim)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, h: torch.Tensor, jepa_tokens: torch.Tensor, mod: BlockModulation
    ) -> torch.Tensor:
        in_dtype = h.dtype
        h = h.float()
        film = self.mlp(jepa_tokens.float())
        g0, z0 = film.chunk(2, dim=-1)
        gamma = mod.alpha_pcm[:, None] * g0 + mod.beta_pcm[:, None]
        zeta = mod.alpha_pcm[:, None] * z0 + mod.beta_pcm[:, None]
        return (h + gamma * self.ln(h) + zeta).to(in_dtype)
