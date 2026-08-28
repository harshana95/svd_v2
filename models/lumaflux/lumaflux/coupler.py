"""HDR Residual Coupler -- paper Sec. 4.5, Eq. (15).

Fuses the physical and perceptual tracks into the latent with a time/layer-gated
additive residual:

    z^l_out = z^l_res + lambda^l_t ( W_p Tphys + W_c C_perc(Tperc) )

``lambda`` (from Psi) decays low-frequency luminance corrections as t -> 0, letting
late layers focus on fine highlight structure.  ``W_p`` and ``W_c`` are zero-init
1x1 projections so the coupler is identity at initialisation (preserves the frozen
FLUX prior).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .modulation import BlockModulation


class HDRResidualCoupler(nn.Module):
    """Time/layer-gated additive fusion of physical + perceptual tracks.

    ``W_p`` is a small phys->dim projection; ``W_c`` is a low-rank dim->dim
    projection (perceptual track).  Both are zero-init -> identity at start.
    """

    def __init__(self, dim: int, phys_dim: int, hidden: int = 64):
        super().__init__()
        self.W_p = nn.Linear(phys_dim, dim)
        self.W_c = nn.Sequential(nn.Linear(dim, hidden), nn.Linear(hidden, dim))
        nn.init.zeros_(self.W_p.weight)
        nn.init.zeros_(self.W_p.bias)
        nn.init.zeros_(self.W_c[-1].weight)
        nn.init.zeros_(self.W_c[-1].bias)

    def forward(self, h: torch.Tensor, tphys_tokens: torch.Tensor,
                tperc_tokens: torch.Tensor, mod: BlockModulation) -> torch.Tensor:
        in_dtype = h.dtype
        residual = self.W_p(tphys_tokens.float()) + self.W_c(tperc_tokens.float())
        return (h.float() + mod.lam[:, None] * residual).to(in_dtype)
