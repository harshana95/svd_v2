"""Color science for LumaFlux (paper arXiv 2604.02787).

All functions are differentiable torch ops operating on channels-first tensors
``[B, 3, H, W]`` (RGB) or ``[B, 1, H, W]`` (luma), values in ``[0, 1]`` unless
noted. We implement:

  * PQ (SMPTE ST 2084) EOTF / OETF      -- HDR signal <-> linear light (1.0 == 10000 nits)
  * BT.709 OETF / EOTF                  -- SDR signal <-> linear light
  * BT.709 <-> BT.2020 gamut matrices   -- linear RGB
  * BT.2020 non-constant-luminance Y'CbCr <-> R'G'B'  (for the RQS tone-field decoder)

Conventions follow the paper: physical luma uses the BT.2020 weights
``m2020 = [0.2627, 0.6780, 0.0593]`` (Sec. 4.1); the training loss linearises
PQ/BT.2020 signals via the inverse PQ EOTF (Eq. 18).
"""

from __future__ import annotations

import torch

# BT.2020 luma weights (Sec. 4.1).
M2020 = (0.2627, 0.6780, 0.0593)

# --- PQ (SMPTE ST 2084) constants ---
_PQ_M1 = 2610.0 / 16384.0          # 0.1593017578125
_PQ_M2 = 2523.0 / 4096.0 * 128.0   # 78.84375
_PQ_C1 = 3424.0 / 4096.0           # 0.8359375
_PQ_C2 = 2413.0 / 4096.0 * 32.0    # 18.8515625
_PQ_C3 = 2392.0 / 4096.0 * 32.0    # 18.6875

_EPS = 1e-7

# Linear-RGB gamut conversion matrices (rows multiply an RGB column vector).
_M_709_TO_2020 = (
    (0.6274039, 0.3292830, 0.0433131),
    (0.0690970, 0.9195400, 0.0113612),
    (0.0163916, 0.0880132, 0.8955953),
)
_M_2020_TO_709 = (
    (1.6604910, -0.5876411, -0.0728499),
    (-0.1245505, 1.1328999, -0.0083494),
    (-0.0181508, -0.1005789, 1.1187297),
)


def _apply_matrix(rgb: torch.Tensor, mat) -> torch.Tensor:
    """Apply a 3x3 colour matrix to ``[B, 3, H, W]`` linear RGB."""
    m = torch.as_tensor(mat, dtype=rgb.dtype, device=rgb.device)
    b, c, h, w = rgb.shape
    out = torch.einsum("ij,bjhw->bihw", m, rgb)
    return out


# --------------------------------------------------------------------------- #
# PQ transfer functions
# --------------------------------------------------------------------------- #
def pq_eotf(signal: torch.Tensor) -> torch.Tensor:
    """PQ signal in [0, 1] -> normalised linear light in [0, 1] (1.0 == 10000 nits)."""
    e = signal.clamp(0.0, 1.0)
    ep = e.pow(1.0 / _PQ_M2)
    num = (ep - _PQ_C1).clamp_min(0.0)
    den = (_PQ_C2 - _PQ_C3 * ep).clamp_min(_EPS)
    lin = (num / den).pow(1.0 / _PQ_M1)
    return lin.clamp(0.0, 1.0)


def pq_oetf(linear: torch.Tensor) -> torch.Tensor:
    """Normalised linear light in [0, 1] -> PQ signal in [0, 1]."""
    lin = linear.clamp(0.0, 1.0)
    lm = lin.pow(_PQ_M1)
    num = _PQ_C1 + _PQ_C2 * lm
    den = 1.0 + _PQ_C3 * lm
    return (num / den).pow(_PQ_M2).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# BT.709 transfer functions
# --------------------------------------------------------------------------- #
def bt709_eotf(signal: torch.Tensor) -> torch.Tensor:
    """BT.709 (gamma) signal -> linear light. Inverse of :func:`bt709_oetf`."""
    v = signal.clamp(0.0, 1.0)
    low = v / 4.5
    high = ((v + 0.099) / 1.099).clamp_min(_EPS).pow(1.0 / 0.45)
    return torch.where(v < 0.081, low, high)


def bt709_oetf(linear: torch.Tensor) -> torch.Tensor:
    """Linear light -> BT.709 (gamma) signal."""
    l = linear.clamp(0.0, 1.0)
    low = 4.5 * l
    high = 1.099 * l.clamp_min(_EPS).pow(0.45) - 0.099
    return torch.where(l < 0.018, low, high)


# --------------------------------------------------------------------------- #
# Gamut conversion (linear RGB)
# --------------------------------------------------------------------------- #
def rgb709_to_rgb2020_linear(rgb: torch.Tensor) -> torch.Tensor:
    return _apply_matrix(rgb, _M_709_TO_2020)


def rgb2020_to_rgb709_linear(rgb: torch.Tensor) -> torch.Tensor:
    return _apply_matrix(rgb, _M_2020_TO_709)


def luma2020(rgb: torch.Tensor) -> torch.Tensor:
    """Weighted BT.2020 luma of an RGB tensor -> ``[B, 1, H, W]``."""
    w = torch.tensor(M2020, dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    return (rgb * w).sum(dim=1, keepdim=True)


# --------------------------------------------------------------------------- #
# SDR (BT.709 display) -> physical linear BT.2020 light, used by PGA features.
# --------------------------------------------------------------------------- #
def sdr_to_linear2020(sdr: torch.Tensor) -> torch.Tensor:
    """8-bit SDR BT.709 display signal [0,1] -> linear BT.2020 light [0,1]."""
    lin709 = bt709_eotf(sdr)
    return rgb709_to_rgb2020_linear(lin709).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# BT.2020 non-constant-luminance Y'CbCr (operates on display *signal* values).
# Used by the RQS tone-field decoder (Sec. 4.6): split luma, expand, recombine.
# --------------------------------------------------------------------------- #
_CB_SCALE = 1.8814
_CR_SCALE = 1.4746


def rgb_to_ycbcr2020(rgb: torch.Tensor) -> torch.Tensor:
    """R'G'B' signal -> [Y', Cb, Cr] (Cb, Cr centred at 0), channels-first."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = M2020[0] * r + M2020[1] * g + M2020[2] * b
    cb = (b - y) / _CB_SCALE
    cr = (r - y) / _CR_SCALE
    return torch.cat([y, cb, cr], dim=1)


def ycbcr2020_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`rgb_to_ycbcr2020`."""
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
    r = y + _CR_SCALE * cr
    b = y + _CB_SCALE * cb
    g = (y - M2020[0] * r - M2020[2] * b) / M2020[1]
    return torch.cat([r, g, b], dim=1)
