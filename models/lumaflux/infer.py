#!/usr/bin/env python
"""LumaFlux inference / testing (paper Algorithm 1).

Loads the frozen FLUX backbone + trained adapters and converts SDR frames to
10-bit PQ/BT.2020 HDR.  Outputs are saved as 16-bit PNG (PQ-encoded) and, if
``ffmpeg`` is available, optionally as 10-bit PQ BT.2020 HEVC.

Usage:
    python infer.py --config configs/lumaflux.yaml --ckpt runs/lumaflux/adapters_final.pt \
        --input data/sample/sdr --output runs/lumaflux/preds
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from lumaflux import LumaFlux, LumaFluxConfig
from lumaflux.dataset import _load_image, _to_chw
from lumaflux.imgio import imwrite16

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lumaflux.yaml")
    ap.add_argument("--ckpt", default=None, help="adapter checkpoint (.pt); omit for untrained")
    ap.add_argument("--input", required=True, help="SDR image file or directory")
    ap.add_argument("--output", default="runs/lumaflux/preds")
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--save_hevc", action="store_true")
    return ap.parse_args()


def list_inputs(path: str):
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted([q for q in p.iterdir() if q.suffix.lower() in _IMG_EXTS])


def save_png16(path: str, img: np.ndarray):
    imwrite16(path, img)


def save_hevc(png_path: str, mp4_path: str):
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if ff is None:
        return False
    cmd = [ff, "-y", "-i", png_path, "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
           "-x265-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
           mp4_path]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.exists(mp4_path)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _DTYPES[cfg.get("dtype", "bfloat16")]
    size = args.size or cfg["size"]
    num_steps = args.num_steps or cfg["num_steps"]

    model_cfg = LumaFluxConfig(
        phys_dim=cfg["phys_dim"], global_dim=cfg["global_dim"], num_bands=cfg["num_bands"],
        pga_rank=cfg["pga_rank"], rqs_knots=cfg["rqs_knots"], siglip_name=cfg["siglip_name"],
        siglip_dim=cfg["siglip_dim"], mock_siglip=cfg.get("mock_siglip", False),
        guidance_scale=cfg.get("guidance_scale", 1.0),
    )
    model = LumaFlux.from_pretrained_flux(cfg["flux_path"], model_cfg, torch_dtype=dtype)
    model.to(device).eval()

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(ck["adapters"], strict=False)
        print(f"[lumaflux] loaded adapters from {args.ckpt} "
              f"(missing={len(missing)} unexpected={len(unexpected)})")

    os.makedirs(args.output, exist_ok=True)
    import torch.nn.functional as F
    for path in list_inputs(args.input):
        sdr = _to_chw(_load_image(str(path)))[None].to(device)
        sdr = F.interpolate(sdr, (size, size), mode="bilinear", align_corners=False)
        with torch.no_grad():
            hdr = model.sample(sdr, num_steps=num_steps,
                               init_from_sdr=cfg.get("init_from_sdr", False),
                               sdr_strength=cfg.get("sdr_strength", 0.6))
        img = hdr[0].permute(1, 2, 0).float().cpu().numpy()
        out_png = os.path.join(args.output, path.stem + "_hdr.png")
        save_png16(out_png, img)
        line = f"[lumaflux] {path.name} -> {out_png}  range[{img.min():.3f},{img.max():.3f}]"
        if args.save_hevc:
            ok = save_hevc(out_png, os.path.join(args.output, path.stem + "_hdr.mp4"))
            line += "  hevc=" + ("ok" if ok else "skipped")
        print(line, flush=True)


if __name__ == "__main__":
    main()
