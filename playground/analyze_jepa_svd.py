#!/usr/bin/env python3
"""
SVD / PCA decomposition analysis of encoder features (V-JEPA, DINO, CLIP).

Four analyses:

  1. Singular-value spectrum
       Log-scale singular values + cumulative explained variance.
       Reports k at 90 / 95 / 99 % thresholds.

  2. Blur invariance per component
       For matched GT / blur pairs: σ_lq[i] / σ_gt[i] (energy ratio) and
       cos²(V_gt[:,i], V_lq[:,i]) (directional alignment) for each component.
       High values = this component survives blur = safe to keep in projector.

  3. Dataset-level global V matrix
       SVD across all collected features → global projection V_k [D, k].
       Reports per-token cosine similarity (reconstruction quality) at
       k = 32, 64, 128, 256, 512.  Saves V_k.npy for use as a fixed
       preprocessing layer.

  4. Spatial importance maps
       Left singular vectors U[:,i] weight each patch's contribution to
       component i.  Visualised as heatmaps overlaid on the source image.

Usage
-----
  # Single image, JEPA only
  python playground/analyze_jepa_svd.py --gt path/to/image.png

  # Matched GT / blur pair
  python playground/analyze_jepa_svd.py \\
      --gt /depot/chan129/data/Deblur_MIOR/val/gt/05714.png \\
      --blur /depot/chan129/data/Deblur_MIOR/val/blur/05714.png

  # Many pairs from directories (better dataset SVD statistics)
  python playground/analyze_jepa_svd.py \\
      --gt-dir /depot/chan129/data/Deblur_MIOR/val/gt \\
      --blur-dir /depot/chan129/data/Deblur_MIOR/val/blur \\
      --n-images 50

  # Multi-encoder comparison
  python playground/analyze_jepa_svd.py \\
      --gt-dir ... --blur-dir ... --n-images 20 \\
      --encoders vjepa dino clip
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.LucidFlux.jepa_helpers import (
    extract_vjepa_tokens, load_vjepa2_model, vjepa_from_unit_tensor,
)
from models.reFlux.dino_helpers import (
    extract_dino_tokens, load_dinov2_model,
)
from models.reFlux.clip_helpers import (
    clip_from_unit_tensor, extract_clip_tokens, load_clip_model,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warning] matplotlib not found — saving .npy arrays only.")


# ─── encoder registry ────────────────────────────────────────────────────────

ENCODER_CONFIGS = {
    "vjepa": dict(
        hub_entry="vjepa2_1_vit_large_384",
        image_size=384,
        feature_dim=1024,
    ),
    "dino": dict(
        model_name="facebook/dinov2-large",
        image_size=518,
        feature_dim=1024,
    ),
    "clip": dict(
        model_name="openai/clip-vit-large-patch14",
        image_size=224,
        feature_dim=1024,
    ),
}


def load_encoder(name: str, device, dtype=torch.float32):
    cfg = ENCODER_CONFIGS[name]
    if name == "vjepa":
        model = load_vjepa2_model(cfg["hub_entry"], device, dtype, pretrained=True)
    elif name == "dino":
        model = load_dinov2_model(cfg["model_name"], device, dtype)
    elif name == "clip":
        model = load_clip_model(cfg["model_name"], device, dtype)
    return model


def extract_features(name: str, model, img_01: torch.Tensor, device,
                     dtype=torch.float32) -> torch.Tensor:
    """Return [N, D] patch tokens for one image (batch dim removed)."""
    cfg = ENCODER_CONFIGS[name]
    size = cfg["image_size"]

    with torch.no_grad():
        if name == "vjepa":
            pixels = vjepa_from_unit_tensor(img_01, size=size, device=device, out_dtype=dtype)
            tokens = extract_vjepa_tokens(model, pixels, device=device, dtype=dtype)
        elif name == "dino":
            pixels = vjepa_from_unit_tensor(img_01, size=size, device=device, out_dtype=dtype)
            tokens = extract_dino_tokens(model, pixels, device=device, dtype=dtype)
        elif name == "clip":
            pixels = clip_from_unit_tensor(img_01, size=size, device=device, out_dtype=dtype)
            tokens = extract_clip_tokens(model, pixels, device=device, dtype=dtype)

    return tokens.squeeze(0).float().detach()   # [N, D]


# ─── image utilities ─────────────────────────────────────────────────────────

def load_image_01(path: str) -> torch.Tensor:
    return T.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)


def collect_image_paths(gt_dir, blur_dir, n: int):
    gt_paths = sorted(Path(gt_dir).glob("*"))[:n]
    blur_paths = None
    if blur_dir:
        blur_map = {p.name: p for p in Path(blur_dir).glob("*")}
        blur_paths = [blur_map.get(p.name) for p in gt_paths]
        blur_paths = [b for b in blur_paths if b is not None]
        if len(blur_paths) != len(gt_paths):
            print(f"[warn] {len(gt_paths) - len(blur_paths)} GT images have no matching blur pair.")
            gt_paths = gt_paths[:len(blur_paths)]
    return gt_paths, blur_paths


# ─── SVD utilities ───────────────────────────────────────────────────────────

def compute_svd(feats: torch.Tensor):
    """SVD of [N, D] token matrix.  Returns U [N,K], S [K], Vh [K,D]."""
    feats = feats.detach().float()
    return torch.linalg.svd(feats, full_matrices=False)


def cumulative_variance(S: torch.Tensor):
    """Fraction of total variance explained by top-k components."""
    var = S.detach() ** 2
    return (var.cumsum(0) / var.sum()).cpu().numpy()


def k_at_threshold(cum_var: np.ndarray, thresh: float) -> int:
    idx = np.searchsorted(cum_var, thresh)
    return int(np.clip(idx + 1, 1, len(cum_var)))


def align_signs(V_ref: torch.Tensor, V_tgt: torch.Tensor) -> torch.Tensor:
    """Flip sign of each column of V_tgt so it aligns with V_ref."""
    dots = (V_ref * V_tgt).sum(0)           # [K]
    V_tgt = V_tgt * dots.sign().unsqueeze(0)
    return V_tgt


def component_alignment(V_gt: torch.Tensor, V_lq: torch.Tensor, max_k: int = 128):
    """cos²(V_gt[:,i], V_lq[:,i]) for top max_k components → [max_k]."""
    k = min(max_k, V_gt.shape[1], V_lq.shape[1])
    V_g = F.normalize(V_gt[:, :k], dim=0)
    V_l = F.normalize(V_lq[:, :k], dim=0)
    V_l = align_signs(V_g, V_l)
    return ((V_g * V_l).sum(0) ** 2).detach().cpu().numpy()   # cos² per component


def sigma_ratio(S_gt: torch.Tensor, S_lq: torch.Tensor, max_k: int = 128):
    """σ_lq[i] / σ_gt[i] for top max_k components (energy ratio per component)."""
    k = min(max_k, len(S_gt), len(S_lq))
    ratio = (S_lq[:k].detach() / (S_gt[:k].detach() + 1e-8)).cpu().numpy()
    return np.clip(ratio, 0, 2)


def dataset_svd(all_feats: list, ks=(32, 64, 128, 256, 512)):
    """
    Compute global V matrix from all collected features.
    Returns V_full [D, D], singular values S, and reconstruction cosines at each k.
    """
    stacked = torch.cat(all_feats, dim=0).float()         # [N_total, D]
    stacked = stacked - stacked.mean(0, keepdim=True)
    _, S, V = torch.pca_lowrank(stacked, q=min(stacked.shape[1], 512), center=False)
    # V: [D, q] — principal directions as columns (no transpose needed)
    cum_var = cumulative_variance(S)

    # Reconstruction quality: project and reconstruct, measure per-token cos sim
    rec_cos = {}
    sample = stacked[:min(500, len(stacked))]             # check on a subset
    for k in ks:
        k = min(k, V.shape[1])
        proj = sample @ V[:, :k]                          # [N, k]
        recon = proj @ V[:, :k].T                         # [N, D]
        cos = F.cosine_similarity(sample, recon, dim=-1).mean().item()
        rec_cos[k] = cos

    return V, S, cum_var, rec_cos


# ─── visualisation ────────────────────────────────────────────────────────────

def plot_spectrum(axes_row, encoder_name: str,
                 S_gt: np.ndarray, cum_gt: np.ndarray,
                 S_lq: np.ndarray | None, cum_lq: np.ndarray | None,
                 thresholds=(0.90, 0.95, 0.99)):
    ax_sv, ax_cv = axes_row

    x = np.arange(1, len(S_gt) + 1)

    # Singular values (log scale)
    ax_sv.semilogy(x, S_gt, color="steelblue", lw=1.5, label="GT")
    if S_lq is not None:
        ax_sv.semilogy(x[:len(S_lq)], S_lq, color="tomato", lw=1.5,
                       linestyle="--", label="blur")
    ax_sv.set_xlabel("Component index")
    ax_sv.set_ylabel("Singular value (log)")
    ax_sv.set_title(f"{encoder_name} — singular values")
    ax_sv.legend(fontsize=8)
    ax_sv.set_xlim(1, len(S_gt))

    # Cumulative variance
    ax_cv.plot(x, cum_gt * 100, color="steelblue", lw=1.5, label="GT")
    if cum_lq is not None:
        ax_cv.plot(x[:len(cum_lq)], cum_lq * 100, color="tomato", lw=1.5,
                   linestyle="--", label="blur")
    for t in thresholds:
        k = k_at_threshold(cum_gt, t)
        ax_cv.axvline(k, color="gray", lw=0.8, linestyle=":")
        ax_cv.text(k + 1, t * 99, f"{int(t*100)}% @ k={k}", fontsize=7, color="gray")
    ax_cv.axhline(95, color="gray", lw=0.5, linestyle=":")
    ax_cv.set_xlabel("Number of components k")
    ax_cv.set_ylabel("Cumulative variance (%)")
    ax_cv.set_title(f"{encoder_name} — cumulative variance")
    ax_cv.legend(fontsize=8)
    ax_cv.set_ylim(0, 101)
    ax_cv.set_xlim(1, len(S_gt))


def plot_blur_invariance(axes_row, encoder_name: str,
                          cos2: np.ndarray, ratio: np.ndarray, max_k: int = 64):
    ax_cos, ax_rat = axes_row
    x = np.arange(1, max_k + 1)
    c = cos2[:max_k]
    r = ratio[:max_k]

    ax_cos.bar(x, c, color="steelblue", alpha=0.8)
    ax_cos.axhline(0.9, color="tomato", lw=1, linestyle="--", label="cos²=0.9")
    ax_cos.set_xlabel("Component index")
    ax_cos.set_ylabel("cos²(V_gt, V_lq)")
    ax_cos.set_title(f"{encoder_name} — direction alignment (GT vs blur)")
    ax_cos.set_ylim(0, 1.05)
    ax_cos.legend(fontsize=8)

    color = np.where(r < 0.8, "tomato", np.where(r > 1.2, "darkorange", "steelblue"))
    ax_rat.bar(x, r, color=color, alpha=0.8)
    ax_rat.axhline(1.0, color="black", lw=0.8, linestyle="-")
    ax_rat.axhline(0.8, color="tomato", lw=1, linestyle="--", label="ratio=0.8")
    ax_rat.set_xlabel("Component index")
    ax_rat.set_ylabel("σ_blur[i] / σ_gt[i]")
    ax_rat.set_title(f"{encoder_name} — energy ratio (blur / GT)")
    ax_rat.set_ylim(0, max(r.max() * 1.1, 1.5))
    ax_rat.legend(fontsize=8)


def plot_spatial_maps(fig, outer_cell, encoder_name: str,
                      U: torch.Tensor, img_01: torch.Tensor, grid: int, n_comp: int = 6):
    """Draw n_comp spatial U heatmaps next to the source image."""
    inner = gridspec.GridSpecFromSubplotSpec(
        2, n_comp + 1, subplot_spec=outer_cell, wspace=0.05, hspace=0.05
    )

    # Source image
    for row in range(2):
        ax = fig.add_subplot(inner[row, 0])
        if row == 0:
            img_np = img_01.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
            ax.imshow(img_np)
            ax.set_title(encoder_name, fontsize=8)
        ax.axis("off")

    U_np = U.cpu().numpy()   # [N, K]
    for i in range(n_comp):
        for row, (sign, label) in enumerate([(1, f"U[:,{i}]"), (-1, "")]):
            ax = fig.add_subplot(inner[row, i + 1])
            comp = sign * U_np[:grid * grid, i].reshape(grid, grid)
            vmax = np.abs(comp).max() + 1e-8
            ax.imshow(comp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            if row == 0:
                ax.set_title(f"U[:,{i}]", fontsize=7)
            ax.axis("off")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SVD analysis of encoder features.")
    # Image inputs
    parser.add_argument("--gt",      help="Single GT image path.")
    parser.add_argument("--blur",    help="Single blur image path (optional).")
    parser.add_argument("--gt-dir",  help="Directory of GT images.")
    parser.add_argument("--blur-dir",help="Directory of blur images (matched by filename).")
    parser.add_argument("--n-images", type=int, default=20,
                        help="Max images to load from directories (default: 20).")
    # Encoders
    parser.add_argument("--encoders", nargs="+", default=["vjepa"],
                        choices=["vjepa", "dino", "clip"],
                        help="Encoders to analyse (default: vjepa).")
    # Analysis options
    parser.add_argument("--max-k", type=int, default=64,
                        help="Max components shown in blur-invariance plots (default: 64).")
    parser.add_argument("--n-spatial", type=int, default=6,
                        help="Number of spatial U maps to show (default: 6).")
    parser.add_argument("--save-vk", action="store_true",
                        help="Save global V_k projection matrices to out-dir.")
    # Output
    parser.add_argument("--out-dir", default="playground/jepa_svd",
                        help="Output directory (default: playground/jepa_svd).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── collect image paths ──
    if args.gt_dir:
        gt_paths, blur_paths = collect_image_paths(args.gt_dir, args.blur_dir, args.n_images)
    elif args.gt:
        gt_paths = [Path(args.gt)]
        blur_paths = [Path(args.blur)] if args.blur else None
    else:
        parser.error("Provide --gt or --gt-dir.")

    has_blur = blur_paths is not None and len(blur_paths) > 0
    print(f"GT images:   {len(gt_paths)}")
    print(f"Blur images: {len(blur_paths) if has_blur else 0}")

    # ── load first GT image for spatial maps ──
    first_gt_01 = load_image_01(str(gt_paths[0]))

    # ── figure layout ──
    n_enc = len(args.encoders)
    has_blur_int = int(has_blur)
    # Rows per encoder: 1 (spectrum) + 1 (blur invariance, if blur) + 1 (spatial maps)
    n_rows_per_enc = 2 + has_blur_int     # spectrum row + spatial row + (optional blur row)
    total_rows = n_enc * n_rows_per_enc

    if HAS_MPL:
        fig = plt.figure(figsize=(16, 4 * total_rows))
        outer = gridspec.GridSpec(total_rows, 1, figure=fig, hspace=0.5)

    # ── per-encoder analysis ──
    for enc_idx, enc_name in enumerate(args.encoders):
        print(f"\n{'='*56}")
        print(f"Encoder: {enc_name}")
        print(f"{'='*56}")

        model = load_encoder(enc_name, device)

        # Collect features
        feats_gt_list, feats_lq_list = [], []
        for i, gp in enumerate(gt_paths):
            img_gt = load_image_01(str(gp))
            f_gt = extract_features(enc_name, model, img_gt, device)
            feats_gt_list.append(f_gt)
            if has_blur and blur_paths[i] is not None:
                img_lq = load_image_01(str(blur_paths[i]))
                f_lq = extract_features(enc_name, model, img_lq, device)
                feats_lq_list.append(f_lq)
            if (i + 1) % 5 == 0:
                print(f"  Extracted {i+1}/{len(gt_paths)} images")

        print(f"  Token shape (per image): {feats_gt_list[0].shape}")

        # ── Analysis 1: spectrum on first image ──
        f0 = feats_gt_list[0]
        N, D = f0.shape
        grid = int(round(N ** 0.5))
        U_gt, S_gt, Vh_gt = compute_svd(f0)
        V_gt = Vh_gt.T
        cum_gt = cumulative_variance(S_gt)

        U_lq, S_lq, V_lq, cum_lq = None, None, None, None
        if feats_lq_list:
            f0_lq = feats_lq_list[0]
            U_lq, S_lq, Vh_lq = compute_svd(f0_lq)
            V_lq = Vh_lq.T
            cum_lq = cumulative_variance(S_lq)

        k90  = k_at_threshold(cum_gt, 0.90)
        k95  = k_at_threshold(cum_gt, 0.95)
        k99  = k_at_threshold(cum_gt, 0.99)
        print(f"  Per-image intrinsic dimensionality (first GT image):")
        print(f"    90% variance: k = {k90}")
        print(f"    95% variance: k = {k95}")
        print(f"    99% variance: k = {k99}")

        # ── Analysis 2: blur invariance (per-image, paired) ──
        if feats_lq_list:
            # Average blur-invariance metrics across all pairs
            all_cos2, all_ratio = [], []
            for f_g, f_l in zip(feats_gt_list, feats_lq_list):
                _, S_g, Vh_g = compute_svd(f_g)
                _, S_l, Vh_l = compute_svd(f_l)
                all_cos2.append(component_alignment(Vh_g.T, Vh_l.T, args.max_k))
                all_ratio.append(sigma_ratio(S_g, S_l, args.max_k))
            cos2_mean = np.stack(all_cos2).mean(0)
            ratio_mean = np.stack(all_ratio).mean(0)

            stable_k = int((cos2_mean > 0.9).sum())
            print(f"  Blur invariance (avg over {len(feats_lq_list)} pairs):")
            print(f"    Components with cos²>0.9:  {stable_k} / {args.max_k}")
            print(f"    Mean σ_blur/σ_gt (top-{args.max_k}): {ratio_mean.mean():.3f}")
        else:
            cos2_mean, ratio_mean = None, None

        # ── Analysis 3: dataset-level SVD ──
        print(f"  Running dataset-level SVD on {len(feats_gt_list)} GT images...")
        V_global, S_global, cum_global, rec_cos = dataset_svd(feats_gt_list)
        k90d = k_at_threshold(cum_global, 0.90)
        k95d = k_at_threshold(cum_global, 0.95)
        k99d = k_at_threshold(cum_global, 0.99)
        print(f"  Dataset-level intrinsic dimensionality:")
        print(f"    90% variance: k = {k90d}")
        print(f"    95% variance: k = {k95d}")
        print(f"    99% variance: k = {k99d}")
        print(f"  Reconstruction cosine similarity (project → reconstruct):")
        for k, cos in rec_cos.items():
            print(f"    k={k:4d}: {cos:.4f}")

        if args.save_vk:
            vk_path = out_dir / f"V_global_{enc_name}.npy"
            np.save(vk_path, V_global.cpu().numpy())
            print(f"  Global V matrix saved → {vk_path}")

        np.save(out_dir / f"singular_values_{enc_name}_gt.npy",
                S_gt.cpu().numpy())
        if S_lq is not None:
            np.save(out_dir / f"singular_values_{enc_name}_blur.npy",
                    S_lq.cpu().numpy())

        # ── Visualisation ──
        if not HAS_MPL:
            continue

        row_base = enc_idx * n_rows_per_enc

        # Row A: spectrum (2 axes: singular values + cumulative variance)
        spec_cell = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[row_base], wspace=0.35
        )
        ax_sv  = fig.add_subplot(spec_cell[0])
        ax_cv  = fig.add_subplot(spec_cell[1])
        plot_spectrum(
            (ax_sv, ax_cv), enc_name,
            S_gt.cpu().numpy(), cum_gt,
            S_lq.cpu().numpy() if S_lq is not None else None, cum_lq,
        )

        # Row B (optional): blur invariance
        if has_blur and cos2_mean is not None:
            inv_cell = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer[row_base + 1], wspace=0.35
            )
            ax_cos = fig.add_subplot(inv_cell[0])
            ax_rat = fig.add_subplot(inv_cell[1])
            plot_blur_invariance((ax_cos, ax_rat), enc_name,
                                 cos2_mean, ratio_mean, args.max_k)
            spatial_row = row_base + 2
        else:
            spatial_row = row_base + 1

        # Row C: spatial U maps for first GT image
        plot_spatial_maps(
            fig, outer[spatial_row], enc_name,
            U_gt, first_gt_01, grid, n_comp=args.n_spatial,
        )

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── save figure ──
    if HAS_MPL:
        fig.suptitle(
            f"SVD analysis — encoders: {', '.join(args.encoders)}\n"
            f"n_images={len(gt_paths)}, blur={has_blur}",
            fontsize=12,
        )
        out_png = out_dir / "svd_analysis.png"
        plt.savefig(out_png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"\nFigure saved → {out_png}")

    print(f"Arrays saved → {out_dir}/")


if __name__ == "__main__":
    main()
