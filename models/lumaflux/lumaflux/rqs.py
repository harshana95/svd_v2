"""Rational-Quadratic Spline (RQS) tone-field decoder (paper Sec. 4.6, App. D.3).

A frozen VAE decoder is trained for 8-bit SDR statistics, so directly relying on
it to expand dynamic range causes highlight clipping / banding.  LumaFlux adds a
lightweight, learnable, *monotone* rational-quadratic spline (Durkan et al.,
2019) that reshapes the decoded luma into a smooth HDR tone field while keeping
chroma coherent:

    Yhat   = RQS(Y_out; xi, eta, s)          (K = 8 knots by default)
    Uhat,Vhat = U,V + Conv1x1([U, V])        (chroma refinement)
    xhat   = YCbCr2020 -> RGB  (PQ / BT.2020)

The spline parameters (xi=widths, eta=heights, s=derivatives) are predicted from
a pooled latent by a small MLP, giving one monotone tone curve per image.
``spline_smoothness`` returns the adjacent-knot slope penalty L_spline-smooth
used in the training loss (Eq. 18).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .color import rgb_to_ycbcr2020, ycbcr2020_to_rgb

_MIN_BIN = 1e-3
_MIN_DERIV = 1e-3


def monotone_rqs(
    x: torch.Tensor,
    unnorm_widths: torch.Tensor,
    unnorm_heights: torch.Tensor,
    unnorm_derivs: torch.Tensor,
) -> torch.Tensor:
    """Monotone rational-quadratic spline on ``[0, 1] -> [0, 1]``.

    Args:
        x:              ``[B, N]`` inputs in [0, 1].
        unnorm_widths:  ``[B, K]`` unnormalised bin widths.
        unnorm_heights: ``[B, K]`` unnormalised bin heights.
        unnorm_derivs:  ``[B, K+1]`` unnormalised knot derivatives.

    Returns:
        ``[B, N]`` spline outputs in [0, 1].
    """
    b, k = unnorm_widths.shape
    x = x.clamp(0.0, 1.0)

    widths = F.softmax(unnorm_widths, dim=-1) * (1.0 - _MIN_BIN * k) + _MIN_BIN
    heights = F.softmax(unnorm_heights, dim=-1) * (1.0 - _MIN_BIN * k) + _MIN_BIN
    derivs = F.softplus(unnorm_derivs) + _MIN_DERIV  # [B, K+1]

    cumw = torch.cumsum(widths, dim=-1)
    cumw = F.pad(cumw, (1, 0), value=0.0)
    cumw = cumw / cumw[:, -1:].clamp_min(1e-6)  # force last edge == 1
    cumh = torch.cumsum(heights, dim=-1)
    cumh = F.pad(cumh, (1, 0), value=0.0)
    cumh = cumh / cumh[:, -1:].clamp_min(1e-6)

    # Locate bin for each input.  searchsorted over per-row edges.
    idx = torch.searchsorted(cumw.detach(), x, right=True) - 1
    idx = idx.clamp(0, k - 1)  # [B, N]

    def gather(src, index):
        return src.gather(1, index)

    xk = gather(cumw, idx)
    xk1 = gather(cumw, idx + 1)
    yk = gather(cumh, idx)
    yk1 = gather(cumh, idx + 1)
    dk = gather(derivs, idx)
    dk1 = gather(derivs, idx + 1)

    wk = (xk1 - xk).clamp_min(1e-6)
    hk = (yk1 - yk).clamp_min(1e-6)
    s = hk / wk
    xi = ((x - xk) / wk).clamp(0.0, 1.0)
    xi1 = 1.0 - xi

    num = hk * (s * xi * xi + dk * xi * xi1)
    den = s + (dk1 + dk - 2.0 * s) * xi * xi1
    y = yk + num / den.clamp_min(1e-6)
    return y.clamp(0.0, 1.0)


class RQSToneDecoder(nn.Module):
    """Predicts a per-image monotone tone curve + chroma refinement."""

    def __init__(self, latent_dim: int, num_knots: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.num_knots = num_knots
        self.param_net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * num_knots + 1),  # K widths, K heights, K+1 derivs
        )
        # Start close to identity (small params -> near-uniform spline).
        nn.init.zeros_(self.param_net[-1].weight)
        nn.init.zeros_(self.param_net[-1].bias)

        self.chroma = nn.Conv2d(2, 2, kernel_size=1)
        nn.init.zeros_(self.chroma.weight)
        nn.init.zeros_(self.chroma.bias)

        self._last_derivs: torch.Tensor | None = None

    def _split_params(self, latent_pool: torch.Tensor):
        p = self.param_net(latent_pool)
        k = self.num_knots
        uw = p[:, :k]
        uh = p[:, k : 2 * k]
        ud = p[:, 2 * k :]  # [B, K+1]
        return uw, uh, ud

    def forward(self, rgb_signal: torch.Tensor, latent_pool: torch.Tensor) -> torch.Tensor:
        """rgb_signal: ``[B,3,H,W]`` PQ/BT.2020 in [0,1]; latent_pool: ``[B, latent_dim]``."""
        b, _, h, w = rgb_signal.shape
        ycc = rgb_to_ycbcr2020(rgb_signal)
        y = ycc[:, 0:1]
        chroma = ycc[:, 1:3]

        uw, uh, ud = self._split_params(latent_pool)
        self._last_derivs = F.softplus(ud) + _MIN_DERIV

        y_hat = monotone_rqs(y.reshape(b, -1), uw, uh, ud).reshape(b, 1, h, w)
        chroma_hat = chroma + self.chroma(chroma)

        out = ycbcr2020_to_rgb(torch.cat([y_hat, chroma_hat], dim=1))
        return out.clamp(0.0, 1.0)

    def spline_smoothness(self) -> torch.Tensor:
        """L_spline-smooth: penalise adjacent knot slope changes (Eq. 18)."""
        if self._last_derivs is None:
            return torch.zeros((), device=self.chroma.weight.device)
        d = self._last_derivs
        return ((d[:, 1:] - d[:, :-1]) ** 2).mean()
