#!/usr/bin/env python3
"""
Compare V-JEPA 2.1 token features extracted at different input resolutions.

V-JEPA 2.1 (vjepa2_1_vit_large_384) uses Rotary Position Embedding (RoPE) and
handle_nonsquare_inputs=True, so it can process images at any resolution — not
just the 384px training resolution.  This script tests that empirically and
visualises what spatial structure the tokens capture at each resolution.

Visualises (saved as PNG):
  Row 1 — input image at each resolution
  Row 2 — PCA feature map  (top-3 PCA components → RGB)
  Row 3 — centre-token cosine-similarity heatmap
  Row 4 — cross-resolution cosine similarity vs the 384px baseline
           (tgt token grid bilinearly resampled to ref grid size before comparison)

Console output: token shapes, mean L2 norm, and cross-384px cosine similarity.

Usage
-----
  python playground/check_jepa_resolutions.py --image path/to/image.jpg
  python playground/check_jepa_resolutions.py --image path/to/image.jpg \\
      --resolutions 224 384 512 768 \\
      --no-pretrained               # skip downloading weights (shape-only test)
      --out-dir playground/jepa_res
"""

import argparse
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision import transforms as T

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warning] matplotlib not found — saving text summary only.")


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ─── image utilities ──────────────────────────────────────────────────────────

def load_image(path: str) -> torch.Tensor:
    """Load image → [1, 3, H, W] float32 in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return T.ToTensor()(img).unsqueeze(0)


def preprocess(x: torch.Tensor, size: int, device) -> torch.Tensor:
    """Resize + ImageNet normalise → [1, 3, size, size]."""
    mean = IMAGENET_MEAN.to(device)
    std  = IMAGENET_STD.to(device)
    x = F.interpolate(x.to(device).float(), size=(size, size),
                      mode="bicubic", align_corners=False)
    return (x - mean) / std


# ─── model utilities ──────────────────────────────────────────────────────────

def load_model(hub_entry: str, pretrained: bool):
    raw = torch.hub.load(
        "facebookresearch/vjepa2",
        hub_entry,
        pretrained=pretrained,
        trust_repo=True,
    )
    model = raw[0] if isinstance(raw, (list, tuple)) else raw
    return model.eval()


def run_at_resolution(model, img_01: torch.Tensor, res: int, device) -> dict | None:
    """
    Preprocess image to `res` px and run V-JEPA.
    Returns a result dict or None on failure.
    """
    pixels = preprocess(img_01, res, device)   # [1,3,res,res]
    vid    = pixels.unsqueeze(2)               # [1,3,1,res,res]  (T=1 image mode)

    try:
        with torch.inference_mode():
            out = model(vid.to(device))
    except Exception as e:
        return {"error": str(e), "resolution": res}

    # Parse output
    if isinstance(out, torch.Tensor):
        feats = out
    elif isinstance(out, (list, tuple)):
        feats = out[0]
    elif hasattr(out, "last_hidden_state"):
        feats = out.last_hidden_state
    else:
        feats = next(iter(out.values()))

    feats = feats.float()

    # Flatten spatiotemporal dims → [B, N, D]
    if feats.ndim == 5:
        feats = rearrange(feats, "b t h w d -> b (t h w) d")
    elif feats.ndim == 4:
        feats = rearrange(feats, "b t n d -> b (t n) d")

    B, N, D = feats.shape
    grid = int(round(N ** 0.5))

    return {
        "resolution": res,
        "tokens":     feats,          # [1, N, D]
        "N": N, "D": D, "grid": grid,
        "mean_norm":  feats.norm(dim=-1).mean().item(),
        "error":      None,
    }


# ─── analysis ─────────────────────────────────────────────────────────────────

def pca_rgb(tokens: torch.Tensor, grid: int) -> np.ndarray:
    """Project [1, N, D] tokens to top-3 PCA → [grid, grid, 3] uint8."""
    toks = tokens.squeeze(0)                          # [N, D]
    centered = toks - toks.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(centered, q=3, center=False)
    proj = (centered @ V).view(grid, grid, 3).cpu().float().numpy()
    for c in range(3):
        lo, hi = proj[:, :, c].min(), proj[:, :, c].max()
        proj[:, :, c] = (proj[:, :, c] - lo) / (hi - lo + 1e-8)
    return proj  # [H, W, 3] in [0, 1]


def center_cos_map(tokens: torch.Tensor, grid: int) -> np.ndarray:
    """Cosine similarity of center token vs all tokens → [grid, grid]."""
    toks = tokens.squeeze(0)                          # [N, D]
    idx  = (grid // 2) * grid + (grid // 2)
    q    = F.normalize(toks[idx : idx + 1], dim=-1)  # [1, D]
    k    = F.normalize(toks, dim=-1)                  # [N, D]
    return (k @ q.T).squeeze(-1).view(grid, grid).cpu().numpy()


def cross_cos(ref: dict, tgt: dict) -> tuple[float, np.ndarray]:
    """
    Mean per-position cosine similarity after spatially resampling tgt to ref grid.
    Returns (mean_cos, spatial_map [ref_grid, ref_grid]).
    """
    rg, tg = ref["grid"], tgt["grid"]
    D = ref["D"]

    # Reshape to spatial grids
    ref_grid = ref["tokens"].squeeze(0).view(rg, rg, D).permute(2, 0, 1).unsqueeze(0)
    tgt_grid = tgt["tokens"].squeeze(0).view(tg, tg, D).permute(2, 0, 1).unsqueeze(0)

    if rg != tg:
        tgt_grid = F.interpolate(tgt_grid, size=(rg, rg), mode="bilinear", align_corners=False)

    ref_flat = ref_grid.squeeze(0).view(D, -1).T   # [N, D]
    tgt_flat = tgt_grid.squeeze(0).view(D, -1).T

    per_pos = (F.normalize(ref_flat, dim=-1) * F.normalize(tgt_flat, dim=-1)).sum(-1)
    return per_pos.mean().item(), per_pos.view(rg, rg).cpu().numpy()


# ─── visualisation ────────────────────────────────────────────────────────────

def visualize(results: dict, img_01: torch.Tensor, ref_res: int, hub_entry: str,
              image_name: str, out_path: Path) -> None:
    ok = {r: v for r, v in results.items() if v["error"] is None}
    if not ok:
        print("No successful results to visualise.")
        return

    n_cols = len(ok)
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(max(4 * n_cols, 8), 4 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"V-JEPA resolution comparison — {image_name}\nmodel: {hub_entry}",
        fontsize=11, y=1.01,
    )

    row_labels = [
        "Input image",
        "PCA feature map\n(top-3 → RGB)",
        "Centre-token\ncosine similarity",
        f"Cross-res cosine\nvs {ref_res}px (ref)",
    ]
    for r, lbl in enumerate(row_labels):
        axes[r][0].set_ylabel(lbl, fontsize=9, labelpad=8)

    ref = results[ref_res] if ref_res in results and results[ref_res]["error"] is None else None

    for col_idx, (res, info) in enumerate(sorted(ok.items())):
        grid = info["grid"]
        toks = info["tokens"]

        # ── Row 0: input image ──
        ax = axes[0][col_idx]
        img_disp = F.interpolate(img_01, size=(res, res), mode="bicubic", align_corners=False)
        img_np   = img_disp.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        ax.imshow(img_np)
        ax.set_title(
            f"{res}px\n{info['N']} tokens  ({grid}²×{info['D']})\nnorm={info['mean_norm']:.2f}",
            fontsize=8,
        )
        ax.axis("off")

        # ── Row 1: PCA feature map ──
        ax = axes[1][col_idx]
        ax.imshow(pca_rgb(toks, grid))
        ax.set_title("PCA (top-3 components)", fontsize=8)
        ax.axis("off")

        # ── Row 2: centre-token cosine heatmap ──
        ax = axes[2][col_idx]
        im = ax.imshow(center_cos_map(toks, grid), cmap="RdYlGn", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("cos(centre, all tokens)", fontsize=8)
        ax.axis("off")

        # ── Row 3: cross-resolution cosine vs ref ──
        ax = axes[3][col_idx]
        if ref is None or res == ref_res:
            ax.text(0.5, 0.5, "(reference)" if res == ref_res else "no ref",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.axis("off")
        else:
            mean_c, cross_map = cross_cos(ref, info)
            im = ax.imshow(cross_map, cmap="RdYlGn", vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"mean cos = {mean_c:.4f}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nVisualization saved → {out_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V-JEPA resolution comparison.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--resolutions", nargs="+", type=int, default=[224, 384, 512, 768],
        help="Resolutions to test (default: 224 384 512 768).",
    )
    parser.add_argument(
        "--hub-entry", default="vjepa2_1_vit_large_384",
        help="torch.hub model entry (default: vjepa2_1_vit_large_384).",
    )
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Load without pretrained weights (random init, fast shape check).",
    )
    parser.add_argument(
        "--ref-res", type=int, default=384,
        help="Reference resolution for cross-resolution comparison (default: 384).",
    )
    parser.add_argument(
        "--out-dir", default="playground/jepa_resolution_check",
        help="Output directory for PNG and .npy token arrays.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── model info ──
    print(f"Loading V-JEPA: {args.hub_entry}  (pretrained={not args.no_pretrained})")
    model = load_model(args.hub_entry, pretrained=not args.no_pretrained)
    model.to(device)
    print(f"  use_rope:               {getattr(model, 'use_rope', '?')}")
    print(f"  handle_nonsquare_inputs:{getattr(model, 'handle_nonsquare_inputs', '?')}")
    print(f"  patch_size:             {getattr(model, 'patch_size', '?')}")
    print(f"  training resolution:    {getattr(model, 'img_height', '?')}×"
          f"{getattr(model, 'img_width', '?')}")

    # ── load image ──
    img_01 = load_image(args.image)
    image_name = Path(args.image).name
    orig_H, orig_W = img_01.shape[-2], img_01.shape[-1]
    print(f"\nImage: {image_name}  original size {orig_H}×{orig_W}")

    # ── run at each resolution ──
    resolutions = sorted(set(args.resolutions))
    if args.ref_res not in resolutions:
        resolutions.append(args.ref_res)
        resolutions = sorted(resolutions)

    results = {}
    for res in resolutions:
        print(f"\n{'─'*48}")
        print(f"  Resolution: {res}px")
        info = run_at_resolution(model, img_01, res, device)
        results[res] = info
        if info["error"]:
            print(f"  FAIL: {info['error']}")
        else:
            g = info["grid"]
            print(f"  OK → {info['N']} tokens  ({g}²×{info['D']}), "
                  f"mean norm={info['mean_norm']:.3f}")

    # ── cross-resolution summary ──
    ref_res = args.ref_res
    if ref_res not in results or results[ref_res]["error"]:
        # Fall back to first successful result
        for r in resolutions:
            if results[r]["error"] is None:
                ref_res = r
                break

    ref = results.get(ref_res)
    print(f"\n{'='*48}")
    print(f"  Cross-resolution cosine similarity (ref = {ref_res}px)")
    print(f"  {'Res':>6}  {'Tokens':>8}  {'Grid':>6}  {'Cos vs ref':>12}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*12}")

    for res in resolutions:
        info = results[res]
        if info["error"]:
            cos_str = "     FAIL"
        elif res == ref_res:
            cos_str = "  1.0000 (ref)"
        else:
            mean_c, _ = cross_cos(ref, info)
            info["cross_cos"] = mean_c
            cos_str = f"  {mean_c:.4f}"
        n_str = str(info["N"]) if not info["error"] else "—"
        g_str = str(info.get("grid", "—"))
        print(f"  {res:>6}px  {n_str:>8}  {g_str:>6}  {cos_str:>14}")

    # ── interpret ──
    print(f"\n  Interpretation:")
    ok_results = [(r, v) for r, v in results.items() if not v["error"] and r != ref_res]
    for res, info in ok_results:
        cos = info.get("cross_cos", float("nan"))
        direction = "more tokens, richer detail" if res > ref_res else "fewer tokens, coarser"
        if cos > 0.90:
            note = "features nearly identical to reference — resolution has little effect"
        elif cos > 0.70:
            note = "moderate feature shift — RoPE adapts but some information differs"
        else:
            note = "significant feature shift — tokens encode different spatial structure"
        print(f"  {res}px ({direction}): {note}  [cos={cos:.3f}]")

    # ── save token arrays ──
    for res, info in results.items():
        if not info["error"]:
            np.save(out_dir / f"tokens_{res}px.npy",
                    info["tokens"].squeeze(0).cpu().numpy())
    print(f"\nToken arrays saved to {out_dir}/")

    # ── visualisation ──
    if HAS_MPL:
        out_png = out_dir / "resolution_comparison.png"
        visualize(results, img_01, ref_res, args.hub_entry, image_name, out_png)
    else:
        print("Install matplotlib to enable visualisation.")


if __name__ == "__main__":
    main()
