#!/usr/bin/env python
"""LumaFlux training (paper Sec. 6.1).

Frozen FLUX.1-dev backbone + VAE; only the LumaFlux adapters train.  Loss is the
pixel-space Eq. (18) computed on the RQS-decoded HDR output (optional
flow-matching aux for stability).

Usage:
    python train.py --config configs/lumaflux.yaml [--key value ...]
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import yaml
from torch.utils.data import DataLoader

from lumaflux import LumaFlux, LumaFluxConfig, LumaFluxLoss
from lumaflux.dataset import PairedHDRDataset, SyntheticPairDataset

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_config():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lumaflux.yaml")
    args, extra = ap.parse_known_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # Allow `--key value` overrides (parsed as YAML scalars).
    it = iter(extra)
    for key in it:
        val = next(it)
        cfg[key.lstrip("-")] = yaml.safe_load(val)
    cfg["config_path"] = args.config
    return cfg


def adapter_state_dict(model: LumaFlux):
    skip = ("transformer.", "vae.", "perceptual.vision.")
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not k.startswith(skip)}


def save_checkpoint(model, optimizer, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"step": step, "adapters": adapter_state_dict(model),
                "optimizer": optimizer.state_dict()}, path)


def maybe_set_null_text(model, cfg, device):
    """Prompt-free conditioning: use real "" T5/CLIP embeds if requested."""
    if not cfg.get("use_real_null", False):
        return
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    fp = cfg["flux_path"]
    tok = CLIPTokenizer.from_pretrained(fp, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(fp, subfolder="text_encoder", torch_dtype=torch.float32).to(device)
    tok2 = T5TokenizerFast.from_pretrained(fp, subfolder="tokenizer_2")
    te2 = T5EncoderModel.from_pretrained(fp, subfolder="text_encoder_2", torch_dtype=torch.float32).to(device)
    with torch.no_grad():
        ids = tok("", padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
        pooled = te(ids).pooler_output
        ids2 = tok2("", padding="max_length", max_length=512, return_tensors="pt").input_ids.to(device)
        t5 = te2(ids2)[0]
    model.set_null_text(t5, pooled)
    del te, te2
    torch.cuda.empty_cache()


def cosine_warmup(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def build_dataset(cfg):
    if cfg["dataset"] == "synthetic":
        return SyntheticPairDataset(length=cfg.get("synthetic_len", 64), size=cfg["size"])
    return PairedHDRDataset(cfg["sdr_dir"], cfg["hdr_dir"], size=cfg["size"],
                            random_crop=cfg.get("random_crop", True))


def main():
    cfg = load_config()
    torch.manual_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _DTYPES[cfg.get("dtype", "bfloat16")]

    model_cfg = LumaFluxConfig(
        phys_dim=cfg["phys_dim"], global_dim=cfg["global_dim"], num_bands=cfg["num_bands"],
        pga_rank=cfg["pga_rank"], rqs_knots=cfg["rqs_knots"], siglip_name=cfg["siglip_name"],
        siglip_dim=cfg["siglip_dim"], mock_siglip=cfg.get("mock_siglip", False),
        guidance_scale=cfg.get("guidance_scale", 1.0),
    )
    print(f"[lumaflux] loading FLUX backbone from {cfg['flux_path']}")
    model = LumaFlux.from_pretrained_flux(cfg["flux_path"], model_cfg, torch_dtype=dtype)
    model.to(device)
    if cfg.get("gradient_checkpointing", True):
        model.enable_gradient_checkpointing()
    maybe_set_null_text(model, cfg, device)
    print(f"[lumaflux] trainable params: {model.num_trainable()/1e6:.2f}M "
          f"/ total {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    loss_fn = LumaFluxLoss(cfg["lambda1"], cfg["lambda2"], cfg["lambda3"], cfg.get("fm_weight", 0.0))
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg["lr"],
                            betas=tuple(cfg["adam_betas"]), weight_decay=cfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_warmup(s, cfg["warmup_steps"], cfg["max_steps"]))

    ds = build_dataset(cfg)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg.get("num_workers", 4), drop_last=True, pin_memory=True)
    print(f"[lumaflux] dataset: {len(ds)} pairs, batch {cfg['batch_size']}")

    model.train()
    model.transformer.eval()  # keep frozen norms in eval
    model.vae.eval()
    step, t0 = 0, time.time()
    accum = cfg.get("grad_accum", 1)
    opt.zero_grad(set_to_none=True)
    while step < cfg["max_steps"]:
        for batch in loader:
            sdr = batch["sdr"].to(device, non_blocking=True)
            hdr = batch["hdr"].to(device, non_blocking=True)
            x_hat, v_pred, v_target = model.training_step(sdr, hdr)
            loss, logs = loss_fn(x_hat, hdr, model.rqs.spline_smoothness(), v_pred, v_target)
            (loss / accum).backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.get("grad_clip", 1.0))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            if step % cfg["log_every"] == 0:
                rate = (step + 1) / (time.time() - t0)
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in logs.items())
                print(f"[step {step:>7}] {msg} lr={sched.get_last_lr()[0]:.2e} {rate:.2f} it/s",
                      flush=True)
            if step > 0 and step % cfg["ckpt_every"] == 0:
                p = os.path.join(cfg["output_dir"], f"adapters_{step:07d}.pt")
                save_checkpoint(model, opt, step, p)
                print(f"[lumaflux] saved {p}", flush=True)

            step += 1
            if step >= cfg["max_steps"]:
                break

    final = os.path.join(cfg["output_dir"], "adapters_final.pt")
    save_checkpoint(model, opt, step, final)
    print(f"[lumaflux] done. saved {final}")


if __name__ == "__main__":
    main()
