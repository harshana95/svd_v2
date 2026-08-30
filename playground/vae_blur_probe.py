#!/usr/bin/env python3
"""
Cheap frozen-VAE probe: does the FLUX.2 Klein VAE's own (untouched) latent
space already carry enough signal that a lightweight linear map can bridge
VAE-encode(blur) -> VAE-encode(GT), decoded back through the frozen decoder?

This is the near-zero-cost sanity check for "train the VAE encoder JEPA-style
(blur context -> GT-latent target)" (papers/main.tex Sec. 5/6 discussion):
before spending any GPU-hours on LoRA-adapting the VAE encoder, check whether
even a closed-form ridge regression on top of the FROZEN VAE's own packed
latent tokens already moves the needle. No VAE weights are touched here.

Reuses the ridge-regression harness from jepa_linear_probe.py (closed-form
W = (X^T X + lambda I)^-1 X^T Y) and the packed-VAE-token helpers from
models/LucidFlux/helpers.py (encode_vae_packed_tokens / decode_packed_vae_tokens),
the same functions ContextProjectorAlign_model.py's Stage A uses.

Usage
-----
  python playground/vae_blur_probe.py \\
      --gt-dir /depot/chan129/users/harshana/eval_sets/gopro_test/gt \\
      --blur-dir /depot/chan129/users/harshana/eval_sets/gopro_test/blur \\
      --n-train 250 --n-val 50 --crop-size 512 \\
      --out-dir playground/vae_blur_probe
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from torchvision import transforms as T

try:
    from diffusers import AutoencoderKLFlux2
except ImportError:
    from diffusers import AutoencoderKL as AutoencoderKLFlux2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.LucidFlux.helpers import (
    decode_packed_vae_tokens,
    encode_vae_packed_tokens,
    get_flux2_pipeline_class,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warning] matplotlib not found -- saving .npy/.json only.")


# ─── image utilities ──────────────────────────────────────────────────────────

def load_image_pm1(path: str, crop_size: int) -> torch.Tensor:
    """Load, center-crop to crop_size, return in [-1, 1] as [1,3,H,W]."""
    img = T.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)  # [1,3,H,W] in [0,1]
    _, _, h, w = img.shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"{path}: {h}x{w} smaller than crop_size={crop_size}")
    top, left = (h - crop_size) // 2, (w - crop_size) // 2
    img = img[:, :, top:top + crop_size, left:left + crop_size]
    return img * 2.0 - 1.0


def collect_image_paths(gt_dir, blur_dir, n: int):
    gt_paths = sorted(Path(gt_dir).glob("*"))[:n]
    blur_map = {p.name: p for p in Path(blur_dir).glob("*")}
    blur_paths = [blur_map.get(p.name) for p in gt_paths]
    keep = [(g, b) for g, b in zip(gt_paths, blur_paths) if b is not None]
    if len(keep) != len(gt_paths):
        print(f"[warn] {len(gt_paths) - len(keep)} GT images have no matching blur pair.")
    gt_paths, blur_paths = zip(*keep)
    return list(gt_paths), list(blur_paths)


def to_01_numpy(img_pm1: torch.Tensor) -> np.ndarray:
    """[1,3,H,W] in [-1,1] -> [H,W,3] in [0,1]."""
    return (img_pm1.squeeze(0).permute(1, 2, 0).clamp(-1, 1).float().cpu().numpy() + 1.0) / 2.0


def eval_reconstruction(pred_01: np.ndarray, gt_01: np.ndarray) -> tuple[float, float]:
    psnr = sk_psnr(gt_01, pred_01, data_range=1.0)
    ssim = sk_ssim(gt_01, pred_01, data_range=1.0, channel_axis=-1)
    return float(psnr), float(ssim)


# ─── ridge regression (same closed form as jepa_linear_probe.py) ─────────────

def fit_ridge(X: torch.Tensor, Y: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    D = X.shape[1]
    reg = ridge_lambda * torch.eye(D, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(X.T @ X + reg, X.T @ Y)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Frozen-VAE blur->GT latent probe.")
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--blur-dir", required=True)
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--crop-size", type=int, default=512,
                    help="Center-crop size fed to the VAE (must be divisible by the "
                         "VAE's spatial downsample factor; 512 is safe).")
    ap.add_argument("--ridge-lambda", type=float, default=10.0)
    ap.add_argument("--pretrained", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--n-qualitative", type=int, default=6)
    ap.add_argument("--out-dir", default="playground/vae_blur_probe")
    ap.add_argument("--label", default="gopro", help="Tag used in printed output only.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading frozen FLUX.2 Klein VAE from {args.pretrained} ...")
    vae = AutoencoderKLFlux2.from_pretrained(args.pretrained, subfolder="vae")
    vae.requires_grad_(False)
    vae.eval()
    vae = vae.to(device, dtype=torch.float32)  # fp32: ridge solve needs numerical stability
    flux2_pipe_cls = get_flux2_pipeline_class(args.pretrained)
    print(f"Pipeline class for patchify/pack/unpack: {flux2_pipe_cls.__name__}")

    n_total = args.n_train + args.n_val
    gt_paths, blur_paths = collect_image_paths(args.gt_dir, args.blur_dir, n_total)
    if len(gt_paths) < n_total:
        print(f"[warn] only {len(gt_paths)} pairs available, wanted {n_total}.")
    train_gt, val_gt = gt_paths[: args.n_train], gt_paths[args.n_train:]
    train_blur, val_blur = blur_paths[: args.n_train], blur_paths[args.n_train:]
    print(f"[{args.label}] Train pairs: {len(train_gt)}   Val pairs: {len(val_gt)}")

    @torch.no_grad()
    def encode(path):
        img = load_image_pm1(str(path), args.crop_size).to(device, dtype=torch.float32)
        tokens, patchified = encode_vae_packed_tokens(vae, img, flux2_pipe_cls=flux2_pipe_cls)
        return tokens.squeeze(0).float(), patchified, img  # tokens: [N_slots, token_dim]

    # ── probe one pair to report the token layout ──
    probe_tokens, probe_patchified, _ = encode(train_blur[0])
    n_slots, token_dim = probe_tokens.shape
    print(f"  VAE packed-token layout: n_slots={n_slots}  token_dim={token_dim}  "
          f"(patchified latent shape {tuple(probe_patchified.shape)})")

    # ── build training data: X = VAE(blur) tokens, Y = VAE(GT) tokens ──
    X_list, Y_list = [], []
    for i, (gp, bp) in enumerate(zip(train_gt, train_blur)):
        x_tokens, _, _ = encode(bp)
        y_tokens, _, _ = encode(gp)
        X_list.append(x_tokens)
        Y_list.append(y_tokens)
        if (i + 1) % 50 == 0:
            print(f"  [train] {i + 1}/{len(train_gt)}")
    X = torch.cat(X_list, dim=0)
    Y = torch.cat(Y_list, dim=0)
    print(f"  Fitting ridge regression: X {tuple(X.shape)}  Y {tuple(Y.shape)}")
    W = fit_ridge(X, Y, args.ridge_lambda)

    # ── evaluate on held-out pairs ──
    # Three numbers per image: (a) linear-probe prediction vs GT, (b) blurred
    # input vs GT (the floor), (c) the VAE's own round-trip of GT (encode(GT)
    # -> decode) vs GT -- an upper-bound sanity check on what this VAE/pipe
    # combination can achieve even with no blur at all.
    psnrs, ssims = [], []
    psnrs_floor, ssims_floor = [], []
    psnrs_roundtrip, ssims_roundtrip = [], []
    qualitative = []

    for i, (gp, bp) in enumerate(zip(val_gt, val_blur)):
        x_tokens, x_patchified, blur_img = encode(bp)
        y_tokens, y_patchified, gt_img = encode(gp)

        pred_tokens = (x_tokens @ W).unsqueeze(0)
        with torch.no_grad():
            pred_decoded = decode_packed_vae_tokens(
                pred_tokens, x_patchified, vae, flux2_pipe_cls=flux2_pipe_cls
            )
            roundtrip_decoded = decode_packed_vae_tokens(
                y_tokens.unsqueeze(0), y_patchified, vae, flux2_pipe_cls=flux2_pipe_cls
            )

        gt_01 = to_01_numpy(gt_img)
        blur_01 = to_01_numpy(blur_img)
        pred_01 = to_01_numpy(pred_decoded).clip(0, 1)
        roundtrip_01 = to_01_numpy(roundtrip_decoded).clip(0, 1)

        p, s = eval_reconstruction(pred_01, gt_01)
        pf, sf = eval_reconstruction(blur_01, gt_01)
        pr, sr = eval_reconstruction(roundtrip_01, gt_01)
        psnrs.append(p); ssims.append(s)
        psnrs_floor.append(pf); ssims_floor.append(sf)
        psnrs_roundtrip.append(pr); ssims_roundtrip.append(sr)

        if i < args.n_qualitative:
            qualitative.append((blur_01, pred_01, roundtrip_01, gt_01, Path(gp).name))
        if (i + 1) % 25 == 0 or (i + 1) == len(val_gt):
            print(f"  [val]   {i + 1}/{len(val_gt)}")

    psnrs, ssims = np.array(psnrs), np.array(ssims)
    psnrs_floor, ssims_floor = np.array(psnrs_floor), np.array(ssims_floor)
    psnrs_roundtrip, ssims_roundtrip = np.array(psnrs_roundtrip), np.array(ssims_roundtrip)

    print(f"\n{'='*60}\nFrozen-VAE blur->GT-latent probe (val, n={len(val_gt)}, crop={args.crop_size})\n{'='*60}")
    print(f"  Linear-probe prediction : PSNR {psnrs.mean():.3f} +/- {psnrs.std():.3f} dB   "
          f"SSIM {ssims.mean():.4f} +/- {ssims.std():.4f}")
    print(f"  Blurred-input floor     : PSNR {psnrs_floor.mean():.3f} +/- {psnrs_floor.std():.3f} dB   "
          f"SSIM {ssims_floor.mean():.4f} +/- {ssims_floor.std():.4f}")
    print(f"  VAE round-trip of GT    : PSNR {psnrs_roundtrip.mean():.3f} +/- {psnrs_roundtrip.std():.3f} dB   "
          f"SSIM {ssims_roundtrip.mean():.4f} +/- {ssims_roundtrip.std():.4f}   "
          f"(ceiling: max this VAE/pipe can do even without blur)")
    delta = psnrs.mean() - psnrs_floor.mean()
    print(f"  Delta vs floor: {delta:+.3f} dB")

    summary = dict(
        label=args.label, n_val=len(val_gt), n_train=len(train_gt),
        crop_size=args.crop_size, n_slots=n_slots, token_dim=token_dim,
        ridge_lambda=args.ridge_lambda,
        psnr_mean=float(psnrs.mean()), psnr_std=float(psnrs.std()),
        ssim_mean=float(ssims.mean()), ssim_std=float(ssims.std()),
        psnr_floor_mean=float(psnrs_floor.mean()), psnr_floor_std=float(psnrs_floor.std()),
        ssim_floor_mean=float(ssims_floor.mean()), ssim_floor_std=float(ssims_floor.std()),
        psnr_roundtrip_mean=float(psnrs_roundtrip.mean()), psnr_roundtrip_std=float(psnrs_roundtrip.std()),
        ssim_roundtrip_mean=float(ssims_roundtrip.mean()), ssim_roundtrip_std=float(ssims_roundtrip.std()),
        delta_vs_floor_db=float(delta),
    )
    with open(out_dir / f"summary_{args.label}.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.save(out_dir / f"W_{args.label}.npy", W.cpu().numpy())
    np.savez(
        out_dir / f"val_metrics_{args.label}.npz",
        psnr=psnrs, ssim=ssims,
        psnr_floor=psnrs_floor, ssim_floor=ssims_floor,
        psnr_roundtrip=psnrs_roundtrip, ssim_roundtrip=ssims_roundtrip,
    )
    print(f"\nSummary saved -> {out_dir}/summary_{args.label}.json")

    if HAS_MPL and qualitative:
        n_q = len(qualitative)
        fig, axes = plt.subplots(n_q, 4, figsize=(12, 3 * n_q), squeeze=False)
        for row, (blur_01, pred_01, roundtrip_01, gt_01, name) in enumerate(qualitative):
            for col, (img, title) in enumerate([
                (blur_01, "blur input"),
                (pred_01, "ridge pred (frozen VAE)"),
                (roundtrip_01, "VAE round-trip(GT)"),
                (gt_01, "GT"),
            ]):
                ax = axes[row][col]
                ax.imshow(np.clip(img, 0, 1))
                if row == 0:
                    ax.set_title(title, fontsize=10)
                if col == 0:
                    ax.set_ylabel(name, fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Frozen-VAE blur->GT-latent probe -- {args.label}", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_dir / f"qualitative_{args.label}.png", dpi=130)
        plt.close(fig)
        print(f"Qualitative panel saved -> {out_dir}/qualitative_{args.label}.png")


if __name__ == "__main__":
    main()
