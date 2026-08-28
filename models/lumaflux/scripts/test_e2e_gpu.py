#!/usr/bin/env python
"""End-to-end GPU test with the REAL FLUX.1-dev backbone.

Loads FLUX.1-dev + SigLIP from the HF cache, runs a few real training steps on the
sample SDR/HDR pairs, then runs inference -- validating the whole pipeline on an
A100.  Not a real training run.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lumaflux import LumaFlux, LumaFluxConfig, LumaFluxLoss, PairedHDRDataset
from lumaflux.dataset import _load_image, _to_chw
from lumaflux.imgio import imwrite16

DEVICE = "cuda"
DTYPE = torch.bfloat16
SIZE = 256

_CACHE = "/scratch/gilbreth/wweligam/.cache/hub/hub"
FLUX_PATH = f"{_CACHE}/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
SIGLIP_PATH = f"{_CACHE}/models--google--siglip2-so400m-patch16-512/snapshots/ceea1cba8130d8271436da4828633198c176a775"


def main():
    cfg = LumaFluxConfig(mock_siglip=False, siglip_name=SIGLIP_PATH)
    t0 = time.time()
    model = LumaFlux.from_pretrained_flux(FLUX_PATH, cfg, torch_dtype=DTYPE)
    model.to(DEVICE)
    print(f"[load] FLUX + adapters in {time.time()-t0:.1f}s")
    # (gradient checkpointing disabled: FLUX attention dispatch trips non-reentrant
    #  checkpoint recompute; not needed at 256px on 80GB.)
    tot = sum(p.numel() for p in model.parameters())
    print(f"[params] trainable={model.num_trainable()/1e6:.2f}M  total={tot/1e9:.2f}B")

    ds = PairedHDRDataset("data/sample/sdr", "data/sample/hdr", size=SIZE)
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    loss_fn = LumaFluxLoss(5.0, 1.0, 0.1, fm_weight=0.1)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4, betas=(0.9, 0.999))

    model.train()
    model.transformer.eval()
    model.vae.eval()

    print("[train] running 6 steps...")
    it = iter(loader)
    for s in range(6):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        sdr = batch["sdr"].to(DEVICE)
        hdr = batch["hdr"].to(DEVICE)
        tic = time.time()
        x_hat, v_pred, v_target = model.training_step(sdr, hdr)
        loss, logs = loss_fn(x_hat, hdr, model.rqs.spline_smoothness(), v_pred, v_target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step()
        msg = " ".join(f"{k}={v.item():.4f}" for k, v in logs.items())
        print(f"  step {s}  {msg}  gnorm={gnorm:.3f}  {time.time()-tic:.2f}s", flush=True)

    print(f"[mem] peak allocated = {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    print("[infer] sampling 2 images...")
    model.eval()
    os.makedirs("runs/test/preds", exist_ok=True)
    for name in ["img_000", "img_001"]:
        sdr = _to_chw(_load_image(f"data/sample/sdr/{name}.png"))[None].to(DEVICE)
        sdr = F.interpolate(sdr, (SIZE, SIZE), mode="bilinear", align_corners=False)
        tic = time.time()
        with torch.no_grad():
            hdr = model.sample(sdr, num_steps=8)
        img = hdr[0].permute(1, 2, 0).float().cpu().numpy()
        out = f"runs/test/preds/{name}_hdr.png"
        imwrite16(out, img)
        print(f"  {name} -> {out}  range[{img.min():.3f},{img.max():.3f}]  {time.time()-tic:.2f}s",
              flush=True)

    print("E2E GPU TEST DONE")


if __name__ == "__main__":
    main()
