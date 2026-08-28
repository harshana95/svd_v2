"""Pre-paired SDR/HDR folder dataset for LumaFlux.

The dataset loads already-paired frames from two directories with matching file
stems:

    sdr_dir/<name>.png   8-bit SDR, BT.709 display, normalised to [0, 1]
    hdr_dir/<name>.png   >=10-bit HDR, PQ / BT.2020 signal, normalised to [0, 1]
                         (16-bit PNG/TIFF or .npy; .exr is treated as *linear*
                          BT.2020 and PQ-encoded on load)

Frames are pixel-aligned, so SDR and HDR are resized/cropped identically.  A
synthetic pair generator is also provided for offline pipeline tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .color import pq_oetf
from .imgio import imread01

_IMG_EXTS = (".png", ".tif", ".tiff", ".npy", ".exr", ".jpg", ".jpeg")


def _load_image(path: str) -> np.ndarray:
    """Load an image as float32 HxWx3 in [0, 1] (PQ-encode .exr linear data)."""
    p = str(path)
    if p.endswith(".exr"):
        import imageio.v3 as iio
        arr = iio.imread(p).astype(np.float32)  # linear
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        t = torch.from_numpy(arr[:, :, :3])[None].permute(0, 3, 1, 2)
        arr = pq_oetf(t / t.amax().clamp_min(1.0)).permute(0, 2, 3, 1)[0].numpy()
        return np.clip(arr, 0.0, 1.0)
    return imread01(p)


def _to_chw(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()


class PairedHDRDataset(Dataset):
    def __init__(self, sdr_dir: str, hdr_dir: str, size: int = 512,
                 random_crop: bool = True):
        self.sdr_dir = Path(sdr_dir)
        self.hdr_dir = Path(hdr_dir)
        self.size = size
        self.random_crop = random_crop

        def index(folder: Path):
            return {p.stem: p for p in folder.iterdir() if p.suffix.lower() in _IMG_EXTS}

        sdr_map = index(self.sdr_dir)
        hdr_map = index(self.hdr_dir)
        self.keys = sorted(set(sdr_map) & set(hdr_map))
        if not self.keys:
            raise RuntimeError(f"No paired stems between {sdr_dir} and {hdr_dir}")
        self.sdr_map = sdr_map
        self.hdr_map = hdr_map

    def __len__(self) -> int:
        return len(self.keys)

    def _crop_resize(self, sdr: torch.Tensor, hdr: torch.Tensor):
        import torch.nn.functional as F
        _, h, w = sdr.shape
        s = self.size
        if min(h, w) < s:
            scale = s / min(h, w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            sdr = F.interpolate(sdr[None], (nh, nw), mode="bilinear", align_corners=False)[0]
            hdr = F.interpolate(hdr[None], (nh, nw), mode="bilinear", align_corners=False)[0]
            _, h, w = sdr.shape
        if self.random_crop:
            top = torch.randint(0, h - s + 1, (1,)).item()
            left = torch.randint(0, w - s + 1, (1,)).item()
        else:
            top, left = (h - s) // 2, (w - s) // 2
        sdr = sdr[:, top:top + s, left:left + s]
        hdr = hdr[:, top:top + s, left:left + s]
        return sdr, hdr

    def __getitem__(self, i: int):
        k = self.keys[i]
        sdr = _to_chw(_load_image(str(self.sdr_map[k])))
        hdr = _to_chw(_load_image(str(self.hdr_map[k])))
        sdr, hdr = self._crop_resize(sdr, hdr)
        return {"sdr": sdr, "hdr": hdr, "name": k}


class SyntheticPairDataset(Dataset):
    """Random PQ HDR frames + a simple tone-mapped SDR (offline pipeline tests only)."""

    def __init__(self, length: int = 32, size: int = 256, seed: int = 0):
        self.length = length
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(self.seed + i)
        base = torch.rand(3, 1, 1, generator=g)
        ramp = torch.linspace(0, 1, self.size)
        field = (ramp[None, :, None] * ramp[None, None, :]).expand(3, -1, -1)
        hdr = (0.3 * base + 0.7 * field + 0.05 * torch.rand(3, self.size, self.size, generator=g)).clamp(0, 1)
        sdr = hdr.pow(0.5).clamp(0, 1)  # crude inverse-of-ITM stand-in
        return {"sdr": sdr, "hdr": hdr, "name": f"synthetic_{i:04d}"}
