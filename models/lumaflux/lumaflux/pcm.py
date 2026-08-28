"""Perceptual Cross-Modulation (PCM) -- paper Sec. 4.4, Eq. (13)-(14).

PCM conditions transformer hidden states on frozen SigLIP embeddings via FiLM
(Perez et al., 2018):

    [gamma, zeta] = alpha_pcm * MLP(C_perc(Tperc)) + beta_pcm
    PCM(h)        = gamma (.) LN(h) + zeta

To preserve the frozen FLUX prior, we realise this as an *identity-initialised
residual* FiLM correction (zero-init MLP, so the block sees the unmodified latent
at step 0 and learns perceptual corrections during training):

    PCM(h) = h + gamma (.) LN(h) + zeta
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .modulation import BlockModulation


class PCMModule(nn.Module):
    """FiLM from SigLIP tokens via a parameter-efficient bottleneck MLP."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * dim)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor, tperc_tokens: torch.Tensor,
                mod: BlockModulation) -> torch.Tensor:
        in_dtype = h.dtype
        h = h.float()
        film = self.mlp(tperc_tokens.float())
        g0, z0 = film.chunk(2, dim=-1)
        gamma = mod.alpha_pcm[:, None] * g0 + mod.beta_pcm[:, None]
        zeta = mod.alpha_pcm[:, None] * z0 + mod.beta_pcm[:, None]
        return (h + gamma * self.ln(h) + zeta).to(in_dtype)
