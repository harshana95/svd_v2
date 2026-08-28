"""Feature extraction & conditioning (paper Sec. 4.1).

Physical stream (``Tphys``, global ``g``, spectral ``r``) and perceptual stream
(``Tperc`` from a frozen SigLIP encoder + a trainable connector).  These feed the
PGA / PCM / Coupler modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .color import sdr_to_linear2020, luma2020


@dataclass
class PhysicalFeatures:
    tphys_map: torch.Tensor  # [B, Cphys, H, W]  local physical descriptors
    g: torch.Tensor          # [B, Cg]           global stats embedding
    r: torch.Tensor          # [B, K]            spectral band energies
    y: torch.Tensor          # [B, 1, H, W]      BT.2020 linear luma


def _sobel_grad_mag(y: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      device=y.device, dtype=y.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


class PhysicalFeatureExtractor(nn.Module):
    """Tphys = Conv3x3([Y, log(1+|grad Y|), sat]); global g; spectral r (Sec. 4.1)."""

    def __init__(self, phys_dim: int = 64, global_dim: int = 64, num_bands: int = 16):
        super().__init__()
        self.num_bands = num_bands
        self.conv = nn.Conv2d(3, phys_dim, kernel_size=3, padding=1)
        self.global_mlp = nn.Sequential(
            nn.Linear(4, global_dim), nn.SiLU(), nn.Linear(global_dim, global_dim)
        )

    def _spectral_bands(self, y: torch.Tensor) -> torch.Tensor:
        # rfft2 magnitude pooled into ``num_bands`` radial-frequency bands.
        b, _, h, w = y.shape
        spec = torch.fft.rfft2(y.float(), norm="ortho").abs()  # [B,1,H,Wf]
        spec = spec.squeeze(1)
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

    def forward(self, sdr: torch.Tensor) -> PhysicalFeatures:
        lin2020 = sdr_to_linear2020(sdr)
        y = luma2020(lin2020)  # [B,1,H,W]
        grad = torch.log1p(_sobel_grad_mag(y))
        sat = sdr.amax(dim=1, keepdim=True) - sdr.amin(dim=1, keepdim=True)
        tphys = self.conv(torch.cat([y, grad, sat], dim=1))

        flat_y = y.flatten(1)
        mu = flat_y.mean(dim=1)
        sigma = flat_y.std(dim=1)
        p95 = torch.quantile(flat_y, 0.95, dim=1)
        p99 = torch.quantile(flat_y, 0.99, dim=1)
        g = self.global_mlp(torch.stack([mu, sigma, p95, p99], dim=1))

        r = self._spectral_bands(y)
        return PhysicalFeatures(tphys_map=tphys, g=g, r=r, y=y)


class PerceptualEncoder(nn.Module):
    """Frozen SigLIP image encoder + trainable connector to the MM-DiT width.

    The SigLIP model is frozen; only ``connector`` trains.  When ``mock=True`` the
    encoder emits a learned-projection of a deterministic stand-in (for offline
    smoke tests without the SigLIP weights).
    """

    def __init__(self, model_dim: int, siglip_dim: int = 1152, mock: bool = False,
                 siglip_name: str = "google/siglip2-so400m-patch16-512"):
        super().__init__()
        self.model_dim = model_dim
        self.siglip_dim = siglip_dim
        self.mock = mock
        self.image_size = 512
        self.connector = nn.Linear(siglip_dim, model_dim)

        self.vision = None
        if not mock:
            from transformers import SiglipVisionModel
            self.vision = SiglipVisionModel.from_pretrained(siglip_name)
            self.vision.requires_grad_(False)
            self.vision.eval()
            # SigLIP normalisation (mean/std 0.5).
            self.register_buffer("_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
            self.register_buffer("_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    @torch.no_grad()
    def _siglip_tokens(self, sdr: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(sdr, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        x = (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)
        out = self.vision(pixel_values=x.to(self.vision.dtype))
        return out.last_hidden_state.float()  # [B, Np, siglip_dim]

    def forward(self, sdr: torch.Tensor) -> torch.Tensor:
        """Returns connector tokens ``[B, Np, model_dim]`` (Np a perfect square)."""
        if self.mock:
            b = sdr.shape[0]
            np_tok = (self.image_size // 16) ** 2
            feat = sdr.mean(dim=[2, 3], keepdim=False)  # [B,3] deterministic stand-in
            base = feat.new_zeros(b, np_tok, self.siglip_dim)
            base[:, :, :3] = feat[:, None, :]
            return self.connector(base)
        tokens = self._siglip_tokens(sdr)
        return self.connector(tokens)


def tokens_to_grid(tokens: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Resample square token grid ``[B, S*S, C]`` to ``[B, out_h*out_w, C]`` (row-major)."""
    b, n, c = tokens.shape
    s = int(round(n ** 0.5))
    grid = tokens.transpose(1, 2).reshape(b, c, s, s)
    grid = F.interpolate(grid, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return grid.reshape(b, c, out_h * out_w).transpose(1, 2)


def map_to_tokens(feat_map: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Resample a feature map ``[B, C, H, W]`` to flattened tokens ``[B, out_h*out_w, C]``."""
    grid = F.interpolate(feat_map, size=(out_h, out_w), mode="bilinear", align_corners=False)
    b, c, _, _ = grid.shape
    return grid.reshape(b, c, out_h * out_w).transpose(1, 2)
