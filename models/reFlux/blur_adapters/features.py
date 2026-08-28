"""LQ blur-amount feature map (physical track analogue of LumaFlux Tphys)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BlurFeatures:
    blur_map: torch.Tensor  # [B, Cblur, H, W]
    g: torch.Tensor  # [B, Cg]
    r: torch.Tensor  # [B, K]
    y: torch.Tensor  # [B, 1, H, W]


def _sobel_grad_mag(y: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=y.device,
        dtype=y.dtype,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _local_variance(y: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Cheap local contrast proxy; low variance correlates with blur."""
    pad = k // 2
    mu = F.avg_pool2d(y, kernel_size=k, stride=1, padding=pad)
    mu2 = F.avg_pool2d(y * y, kernel_size=k, stride=1, padding=pad)
    return (mu2 - mu * mu).clamp_min(0.0)


class BlurMapExtractor(nn.Module):
    """Predict a multi-dim blur-amount map from LQ RGB in [0, 1].

    Handcrafted cues (luma, log-grad, local variance) are lifted by a 3x3 conv to
    ``blur_dim`` channels, matching LumaFlux's thin physical extractor.
    """

    def __init__(self, blur_dim: int = 64, global_dim: int = 64, num_bands: int = 16):
        super().__init__()
        self.num_bands = num_bands
        self.conv = nn.Conv2d(3, blur_dim, kernel_size=3, padding=1)
        self.global_mlp = nn.Sequential(
            nn.Linear(4, global_dim),
            nn.SiLU(),
            nn.Linear(global_dim, global_dim),
        )

    def _spectral_bands(self, y: torch.Tensor) -> torch.Tensor:
        b, _, h, w = y.shape
        spec = torch.fft.rfft2(y.float(), norm="ortho").abs().squeeze(1)
        hh, wf = spec.shape[-2], spec.shape[-1]
        fy = torch.fft.fftfreq(hh, device=y.device).abs()[:, None]
        fx = torch.fft.rfftfreq(w, device=y.device)[None, :wf]
        radius = torch.sqrt(fy * fy + fx * fx)
        radius = (radius / radius.max().clamp_min(1e-6)) * (self.num_bands - 1)
        band_idx = radius.round().long().clamp(0, self.num_bands - 1).reshape(-1)
        flat = spec.reshape(b, -1)
        out = torch.zeros(b, self.num_bands, device=y.device, dtype=flat.dtype)
        out.index_add_(1, band_idx, flat)
        counts = torch.zeros(self.num_bands, device=y.device).index_add_(
            0, band_idx, torch.ones_like(band_idx, dtype=flat.dtype)
        )
        out = out / counts.clamp_min(1.0)[None, :]
        return torch.log1p(out)

    def forward(self, lq01: torch.Tensor) -> BlurFeatures:
        # Accept [-1,1] or [0,1].
        if float(lq01.min()) < -1e-3:
            lq01 = (lq01.float() + 1.0) * 0.5
        lq01 = lq01.float().clamp(0.0, 1.0)
        y = 0.2126 * lq01[:, 0:1] + 0.7152 * lq01[:, 1:2] + 0.0722 * lq01[:, 2:3]
        grad = torch.log1p(_sobel_grad_mag(y))
        # Invert variance so high response ≈ more blur (low local contrast).
        blur_proxy = torch.exp(-_local_variance(y) * 10.0)
        blur_map = self.conv(torch.cat([y, grad, blur_proxy], dim=1))

        flat_y = y.flatten(1)
        mu = flat_y.mean(dim=1)
        sigma = flat_y.std(dim=1)
        p95 = torch.quantile(flat_y, 0.95, dim=1)
        p99 = torch.quantile(flat_y, 0.99, dim=1)
        g = self.global_mlp(torch.stack([mu, sigma, p95, p99], dim=1))
        r = self._spectral_bands(y)
        return BlurFeatures(blur_map=blur_map, g=g, r=r, y=y)


def tokens_to_grid(tokens: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Resample square token grid ``[B, S*S, C]`` to ``[B, out_h*out_w, C]``."""
    b, n, c = tokens.shape
    s = int(round(n**0.5))
    grid = tokens.transpose(1, 2).reshape(b, c, s, s)
    grid = F.interpolate(grid, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return grid.reshape(b, c, out_h * out_w).transpose(1, 2)


def map_to_tokens(feat_map: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Resample ``[B, C, H, W]`` to flattened tokens ``[B, out_h*out_w, C]``."""
    grid = F.interpolate(feat_map, size=(out_h, out_w), mode="bilinear", align_corners=False)
    b, c, _, _ = grid.shape
    return grid.reshape(b, c, out_h * out_w).transpose(1, 2)
