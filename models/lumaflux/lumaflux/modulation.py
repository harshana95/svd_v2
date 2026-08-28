"""Timestep-Layer Adaptive Modulation  Psi(t, ell)  (paper Sec. 4.2, App. D.2).

A single shared conditioner produces all block-wise modulation coefficients

    (alpha_pga, beta_pga, alpha_pcm, beta_pcm, n_spec, lambda) = Psi(t, ell)

where ``t`` is the (continuous) flow time and ``ell`` the block index.  We embed
``t`` with sinusoidal features and ``ell`` with a learned embedding, fuse them by
addition, apply SiLU, then read out each coefficient through a small affine head.
Outputs are bounded (sigmoid / softplus / tanh) for positivity and stability:

  * alpha_pga  >= 0   (softplus)      strength of the PGA low-rank residual
  * beta_pga   ~ 0    (tanh * s)      identity-mix term of PGA
  * alpha_pcm  ~ 1    (1 + tanh * s)  FiLM scale gain
  * beta_pcm   ~ 0    (tanh * s)      FiLM shift gain
  * n_spec     >= 0   (softplus)      spectral-gating strength
  * lambda     in (0,1)(sigmoid)      HDR-residual-coupler gate

The paper notes early layers / large ``t`` should receive stronger global tone
gains while late layers / small ``t`` focus on highlight detail; the learned
conditioner is free to discover this schedule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BlockModulation:
    """Per-block modulation coefficients, each shaped ``[B, 1]``."""

    alpha_pga: torch.Tensor
    beta_pga: torch.Tensor
    alpha_pcm: torch.Tensor
    beta_pcm: torch.Tensor
    n_spec: torch.Tensor
    lam: torch.Tensor


def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of ``t`` (``[B]`` in [0,1]) -> ``[B, dim]``."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :] * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepLayerModulation(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int = 256, hidden_dim: int = 512,
                 beta_scale: float = 0.1, alpha_pcm_scale: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.beta_scale = beta_scale
        self.alpha_pcm_scale = alpha_pcm_scale

        self.t_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_embed = nn.Embedding(num_layers, embed_dim)
        self.trunk = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
        )
        # 6 coefficients, one affine head each.
        self.head = nn.Linear(hidden_dim, 6)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, t: torch.Tensor, ell: int) -> BlockModulation:
        """``t``: ``[B]`` flow time in [0, 1]; ``ell``: python int block index."""
        b = t.shape[0]
        te = self.t_proj(_sinusoidal(t, self.embed_dim))
        le = self.layer_embed(torch.full((b,), int(ell), device=t.device, dtype=torch.long))
        h = self.trunk(te + le)
        raw = self.head(h)  # [B, 6]
        a_pga, b_pga, a_pcm, b_pcm, n_spec, lam = raw.split(1, dim=-1)
        return BlockModulation(
            alpha_pga=torch.nn.functional.softplus(a_pga),
            beta_pga=torch.tanh(b_pga) * self.beta_scale,
            alpha_pcm=1.0 + self.alpha_pcm_scale * torch.tanh(a_pcm),
            beta_pcm=torch.tanh(b_pcm) * self.beta_scale,
            n_spec=torch.nn.functional.softplus(n_spec),
            lam=torch.sigmoid(lam),
        )
