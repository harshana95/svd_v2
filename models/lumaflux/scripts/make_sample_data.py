#!/usr/bin/env python
"""Generate a few aligned SDR/HDR sample pairs for pipeline testing.

Builds procedural linear-HDR scenes (gradients, highlights, coloured shapes),
PQ-encodes them to a 16-bit HDR PNG (PQ/BT.2020), and derives an aligned 8-bit
SDR PNG via a simple Reinhard tone-map + BT.709 OETF.  Not a real dataset --
just enough to exercise the loader, training and inference end to end.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from lumaflux import color
from lumaflux.imgio import imwrite8, imwrite16


def scene(i: int, size: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(i)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    base = torch.stack([xx, yy, (1 - xx) * yy], 0)
    base = base * (0.3 + 0.7 * torch.rand(3, 1, 1, generator=g))
    # add a couple of bright specular highlights (high linear luminance)
    img = base.clone()
    for _ in range(3):
        cx, cy = torch.rand(2, generator=g)
        r = 0.05 + 0.1 * torch.rand(1, generator=g).item()
        d = (xx - cx) ** 2 + (yy - cy) ** 2
        img += (8.0 * torch.exp(-d / (2 * r * r)))[None]
    img += 0.02 * torch.rand(3, size, size, generator=g)
    return img.clamp_min(0.0)  # linear HDR, range can exceed 1


def main(out="data/sample", n=6, size=512):
    sdr_dir, hdr_dir = os.path.join(out, "sdr"), os.path.join(out, "hdr")
    os.makedirs(sdr_dir, exist_ok=True)
    os.makedirs(hdr_dir, exist_ok=True)
    for i in range(n):
        lin = scene(i, size)[None]                       # [1,3,H,W] linear, >1 allowed
        lin_norm = (lin / lin.amax().clamp_min(1.0)).clamp(0, 1)
        hdr_pq = color.pq_oetf(lin_norm)                 # PQ/BT.2020 signal [0,1]

        # Reinhard tone-map -> SDR (aligned with the HDR scene).
        tm = lin / (1.0 + lin)
        sdr = color.bt709_oetf((tm / tm.amax().clamp_min(1e-3)).clamp(0, 1))

        imwrite16(os.path.join(hdr_dir, f"img_{i:03d}.png"), hdr_pq[0].permute(1, 2, 0).numpy())
        imwrite8(os.path.join(sdr_dir, f"img_{i:03d}.png"), sdr[0].permute(1, 2, 0).numpy())
    print(f"wrote {n} SDR/HDR pairs to {sdr_dir} and {hdr_dir}")


if __name__ == "__main__":
    main()
