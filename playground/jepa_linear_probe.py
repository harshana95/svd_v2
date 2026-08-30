#!/usr/bin/env python3
"""
Linear-probe restorability: fit a closed-form ridge-regression read-out from
FROZEN blurred-input encoder tokens back to the corresponding sharp-image
patch, per encoder (V-JEPA / DINOv2 / CLIP).

This is the experiment that makes the blur-invariance analysis predictive
rather than descriptive: if the ranking it produces (which encoder's LQ
tokens best predict the sharp patch) matches the deblurring ranking from the
trained ContextFlux models, the paper's thesis holds regardless of which
encoder wins — blur-robustness of a representation predicts its value as
conditioning for a frozen generative restorer.

Reuses the extraction path from analyze_jepa_svd.py (load_encoder,
extract_features, collect_image_paths, ENCODER_CONFIGS) rather than
reimplementing it.

Usage
-----
  python playground/jepa_linear_probe.py \\
      --gt-dir /depot/chan129/users/harshana/eval_sets/gopro_test/gt \\
      --blur-dir /depot/chan129/users/harshana/eval_sets/gopro_test/blur \\
      --n-train 250 --n-val 50 \\
      --encoders vjepa dino clip \\
      --out-dir playground/jepa_linear_probe
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_jepa_svd as ajs

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warning] matplotlib not found — saving .npy/.json only.")


# ─── patch <-> image utilities ────────────────────────────────────────────────

def resize_01(img_01: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(img_01, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)


def image_to_patches(img_01: torch.Tensor, grid: int, patch_size: int) -> torch.Tensor:
    """[1,3,H,W] -> [grid*grid, patch_size*patch_size*3], row-major to match token order."""
    x = img_01[:, :, : grid * patch_size, : grid * patch_size]
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(grid * grid, -1)
    return patches


def patches_to_image(patches: torch.Tensor, grid: int, patch_size: int) -> torch.Tensor:
    """[grid*grid, patch_size*patch_size*3] -> [1,3,H,W]."""
    x = patches.view(grid, grid, 3, patch_size, patch_size).permute(2, 0, 3, 1, 4)
    return x.reshape(1, 3, grid * patch_size, grid * patch_size)


# ─── ridge regression ─────────────────────────────────────────────────────────

def fit_ridge(X: torch.Tensor, Y: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    """Closed-form ridge regression: W = (X^T X + λI)^-1 X^T Y. X:[M,D] Y:[M,P] -> W:[D,P]."""
    D = X.shape[1]
    reg = ridge_lambda * torch.eye(D, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(X.T @ X + reg, X.T @ Y)


def eval_reconstruction(pred_img: np.ndarray, gt_img: np.ndarray) -> tuple[float, float]:
    psnr = sk_psnr(gt_img, pred_img, data_range=1.0)
    ssim = sk_ssim(gt_img, pred_img, data_range=1.0, channel_axis=-1)
    return float(psnr), float(ssim)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Linear-probe restorability from frozen encoder tokens.")
    parser.add_argument("--gt-dir", required=True, help="Directory of sharp/GT images.")
    parser.add_argument("--blur-dir", required=True, help="Directory of blurred images (matched by filename).")
    parser.add_argument("--n-train", type=int, default=250, help="Pairs used to fit the ridge probe.")
    parser.add_argument("--n-val", type=int, default=50, help="Held-out pairs used to evaluate it.")
    parser.add_argument("--encoders", nargs="+", default=["vjepa", "dino", "clip"],
                        choices=["vjepa", "dino", "clip"])
    parser.add_argument("--ridge-lambda", type=float, default=10.0,
                        help="L2 regularization strength for the closed-form solve.")
    parser.add_argument("--n-qualitative", type=int, default=6,
                        help="Number of val reconstructions to save as a qualitative panel.")
    parser.add_argument("--out-dir", default="playground/jepa_linear_probe")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = args.n_train + args.n_val
    gt_paths, blur_paths = ajs.collect_image_paths(args.gt_dir, args.blur_dir, n_total)
    if len(gt_paths) < n_total:
        print(f"[warn] only {len(gt_paths)} pairs available, wanted {n_total}.")
    train_gt, val_gt = gt_paths[: args.n_train], gt_paths[args.n_train:]
    train_blur, val_blur = blur_paths[: args.n_train], blur_paths[args.n_train:]
    print(f"Train pairs: {len(train_gt)}   Val pairs: {len(val_gt)}")

    results = {}
    for enc_name in args.encoders:
        print(f"\n{'='*56}\nEncoder: {enc_name}\n{'='*56}")
        cfg = ajs.ENCODER_CONFIGS[enc_name]
        size = cfg["image_size"]
        model = ajs.load_encoder(enc_name, device)

        # ── infer grid / patch size from one image ──
        probe_tokens = ajs.extract_features(enc_name, model, ajs.load_image_01(str(train_blur[0])), device)
        n_tokens = probe_tokens.shape[0]
        grid = int(round(n_tokens ** 0.5))
        patch_size = size // grid
        patch_dim = patch_size * patch_size * 3
        print(f"  image_size={size}  grid={grid}  patch_size={patch_size}  patch_dim={patch_dim}")

        # ── build training data: X = LQ tokens, Y = matching sharp patch pixels ──
        X_list, Y_list = [], []
        for i, (gp, bp) in enumerate(zip(train_gt, train_blur)):
            img_blur = ajs.load_image_01(str(bp)).to(device)
            img_gt = ajs.load_image_01(str(gp)).to(device)
            tokens = ajs.extract_features(enc_name, model, img_blur, device)
            gt_patches = image_to_patches(resize_01(img_gt, size), grid, patch_size)
            X_list.append(tokens)
            Y_list.append(gt_patches)
            if (i + 1) % 50 == 0:
                print(f"  [train] {i + 1}/{len(train_gt)}")
        X = torch.cat(X_list, dim=0).float()
        Y = torch.cat(Y_list, dim=0).float().to(X.device)
        print(f"  Fitting ridge regression: X {tuple(X.shape)}  Y {tuple(Y.shape)}")
        W = fit_ridge(X, Y, args.ridge_lambda)

        # ── evaluate on held-out pairs, vs. the blurred-input floor ──
        psnrs, ssims, psnrs_floor, ssims_floor = [], [], [], []
        qualitative = []
        for i, (gp, bp) in enumerate(zip(val_gt, val_blur)):
            img_blur = ajs.load_image_01(str(bp)).to(device)
            img_gt = ajs.load_image_01(str(gp)).to(device)
            tokens = ajs.extract_features(enc_name, model, img_blur, device)
            pred_img = patches_to_image(tokens.float() @ W, grid, patch_size).clamp(0, 1)
            gt_resized = resize_01(img_gt, size)
            blur_resized = resize_01(img_blur, size)

            pred_np = pred_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt_np = gt_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
            blur_np = blur_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()

            p, s = eval_reconstruction(pred_np, gt_np)
            pf, sf = eval_reconstruction(blur_np, gt_np)
            psnrs.append(p); ssims.append(s)
            psnrs_floor.append(pf); ssims_floor.append(sf)

            if i < args.n_qualitative:
                qualitative.append((blur_np, pred_np, gt_np, Path(gp).name))
            if (i + 1) % 25 == 0 or (i + 1) == len(val_gt):
                print(f"  [val]   {i + 1}/{len(val_gt)}")

        psnrs, ssims = np.array(psnrs), np.array(ssims)
        psnrs_floor, ssims_floor = np.array(psnrs_floor), np.array(ssims_floor)

        print(f"  Linear-probe reconstruction (val, n={len(val_gt)}):")
        print(f"    PSNR: {psnrs.mean():.3f} ± {psnrs.std():.3f} dB   "
              f"(blurred-input floor: {psnrs_floor.mean():.3f} ± {psnrs_floor.std():.3f} dB)")
        print(f"    SSIM: {ssims.mean():.4f} ± {ssims.std():.4f}      "
              f"(blurred-input floor: {ssims_floor.mean():.4f} ± {ssims_floor.std():.4f})")

        results[enc_name] = dict(
            psnr_mean=float(psnrs.mean()), psnr_std=float(psnrs.std()),
            ssim_mean=float(ssims.mean()), ssim_std=float(ssims.std()),
            psnr_floor_mean=float(psnrs_floor.mean()), psnr_floor_std=float(psnrs_floor.std()),
            ssim_floor_mean=float(ssims_floor.mean()), ssim_floor_std=float(ssims_floor.std()),
            n_val=len(val_gt), n_train=len(train_gt), grid=grid, patch_size=patch_size,
        )
        np.save(out_dir / f"W_{enc_name}.npy", W.cpu().numpy())
        np.savez(out_dir / f"val_metrics_{enc_name}.npz",
                 psnr=psnrs, ssim=ssims, psnr_floor=psnrs_floor, ssim_floor=ssims_floor)

        if HAS_MPL and qualitative:
            n_q = len(qualitative)
            fig, axes = plt.subplots(n_q, 3, figsize=(9, 3 * n_q), squeeze=False)
            for row, (blur_np, pred_np, gt_np, name) in enumerate(qualitative):
                for col, (img, title) in enumerate(
                    [(blur_np, "blur input"), (pred_np, "linear-probe recon"), (gt_np, "GT")]
                ):
                    ax = axes[row][col]
                    ax.imshow(np.clip(img, 0, 1))
                    if row == 0:
                        ax.set_title(title, fontsize=10)
                    if col == 0:
                        ax.set_ylabel(name, fontsize=7)
                    ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"Linear-probe restorability — {enc_name}", fontsize=12)
            plt.tight_layout()
            plt.savefig(out_dir / f"qualitative_{enc_name}.png", dpi=130)
            plt.close(fig)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── cross-encoder ranking ──
    print(f"\n{'='*56}\nSummary — linear-probe restorability ranking\n{'='*56}")
    ranked = sorted(results.items(), key=lambda kv: kv[1]["psnr_mean"], reverse=True)
    for rank, (name, r) in enumerate(ranked, 1):
        print(f"  {rank}. {name:6s}  PSNR={r['psnr_mean']:.3f}±{r['psnr_std']:.3f} dB   "
              f"SSIM={r['ssim_mean']:.4f}±{r['ssim_std']:.4f}   (n={r['n_val']})")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved → {out_dir}/summary.json")

    if HAS_MPL:
        names = list(results.keys())
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(names, [results[n]["psnr_mean"] for n in names],
               yerr=[results[n]["psnr_std"] for n in names], capsize=5, color="steelblue")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"Linear-probe restorability (n_val={args.n_val})")
        plt.tight_layout()
        plt.savefig(out_dir / "restorability_ranking.png", dpi=150)
        plt.close(fig)
        print(f"Ranking figure saved → {out_dir}/restorability_ranking.png")


if __name__ == "__main__":
    main()
