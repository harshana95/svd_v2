"""Blur + JEPA residual coupler (LumaFlux HDR Residual Coupler analogue)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .modulation import BlockModulation


class BlurResidualCoupler(nn.Module):
    """``h + λ (W_p(blur_tokens) + W_c(jepa_tokens))`` with zero-init projections."""

    def __init__(self, dim: int, blur_dim: int, hidden: int = 64):
        super().__init__()
        self.W_p = nn.Linear(blur_dim, dim)
        self.W_c = nn.Sequential(nn.Linear(dim, hidden), nn.Linear(hidden, dim))
        nn.init.zeros_(self.W_p.weight)
        nn.init.zeros_(self.W_p.bias)
        nn.init.zeros_(self.W_c[-1].weight)
        nn.init.zeros_(self.W_c[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        blur_tokens: torch.Tensor,
        jepa_tokens: torch.Tensor,
        mod: BlockModulation,
    ) -> torch.Tensor:
        in_dtype = h.dtype
        residual = self.W_p(blur_tokens.float()) + self.W_c(jepa_tokens.float())
        return (h.float() + mod.lam[:, None] * residual).to(in_dtype)
