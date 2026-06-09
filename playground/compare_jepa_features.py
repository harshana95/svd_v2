#!/usr/bin/env python3
"""
Compare V-JEPA token statistics between GT and blurred images.

Loads pre-extracted token arrays from playground/jepa_gt/ and playground/jepa_blur/
(produced by check_jepa_output.py) and reports cosine similarity, L2 distance,
and a PCA overlap visualization.

Usage:
    python playground/compare_jepa_features.py
    python playground/compare_jepa_features.py --gt-dir playground/jepa_gt --blur-dir playground/jepa_blur
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_tokens(directory: Path) -> torch.Tensor:
    path = directory / "jepa_tokens.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Token file not found: {path}\n"
            "Run check_jepa_output.py with --save-dir to generate it."
        )
    arr = np.load(path)  # [N, D]
    return torch.from_numpy(arr).float()


def cosine_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Per-token cosine similarity between two [N, D] token sets."""
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    per_token = (a_norm * b_norm).sum(dim=-1)  # [N]
    return {
        "mean": per_token.mean().item(),
        "std": per_token.std().item(),
        "min": per_token.min().item(),
        "max": per_token.max().item(),
    }


def l2_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Per-token L2 distance between two [N, D] token sets."""
    per_token = (a - b).norm(dim=-1)  # [N]
    return {
        "mean": per_token.mean().item(),
        "std": per_token.std().item(),
        "min": per_token.min().item(),
        "max": per_token.max().item(),
    }


def pca_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 8) -> float:
    """
    Subspace overlap between top-k PCA directions of GT and blur token sets.
    Returns the mean squared cosine similarity between paired principal components.
    A value near 1 means both sets span the same subspace.
    """
    n = min(a.shape[0] - 1, b.shape[0] - 1, k)
    if n < 1:
        return float("nan")

    def top_k_directions(x, num):
        centered = x - x.mean(dim=0, keepdim=True)
        _, _, v = torch.pca_lowrank(centered, q=num, center=False)
        return v  # [D, num]

    v_a = top_k_directions(a, n)
    v_b = top_k_directions(b, n)
    # Gram matrix of cosines between principal directions
    gram = (v_a.T @ v_b) ** 2  # [n, n]
    return gram.diag().mean().item()


def interpret_cosine(mean_cos: float) -> str:
    if mean_cos > 0.95:
        return "HIGH — JEPA features are nearly identical despite blur. LQ conditioning is robust."
    if mean_cos > 0.80:
        return "MODERATE — some information lost to blur. Consider SwinIR pre-processing (source_image=pre)."
    return "LOW — JEPA encodes texture sensitive to blur. GT oracle will significantly outperform LQ."


def main():
    parser = argparse.ArgumentParser(description="Compare JEPA GT vs blur token statistics.")
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="playground/jepa_gt",
        help="Directory with jepa_tokens.npy for GT images.",
    )
    parser.add_argument(
        "--blur-dir",
        type=str,
        default="playground/jepa_blur",
        help="Directory with jepa_tokens.npy for blurred images.",
    )
    parser.add_argument(
        "--pca-k",
        type=int,
        default=8,
        help="Number of principal components for subspace overlap metric.",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    blur_dir = Path(args.blur_dir)

    print(f"Loading GT tokens from:   {gt_dir}")
    gt_tokens = load_tokens(gt_dir)
    print(f"  shape: {tuple(gt_tokens.shape)}")

    print(f"Loading blur tokens from: {blur_dir}")
    blur_tokens = load_tokens(blur_dir)
    print(f"  shape: {tuple(blur_tokens.shape)}\n")

    if gt_tokens.shape != blur_tokens.shape:
        print(
            f"WARNING: shape mismatch ({gt_tokens.shape} vs {blur_tokens.shape}). "
            "Truncating to common length."
        )
        n = min(gt_tokens.shape[0], blur_tokens.shape[0])
        gt_tokens = gt_tokens[:n]
        blur_tokens = blur_tokens[:n]

    cos = cosine_stats(gt_tokens, blur_tokens)
    l2 = l2_stats(gt_tokens, blur_tokens)
    overlap = pca_overlap(gt_tokens, blur_tokens, k=args.pca_k)

    print("=" * 60)
    print("  JEPA Feature Distance: GT vs Blurred Image")
    print("=" * 60)
    print(f"  Tokens:            {gt_tokens.shape[0]} × {gt_tokens.shape[1]}")
    print()
    print("  Cosine similarity (per token):")
    print(f"    mean ± std :  {cos['mean']:.4f} ± {cos['std']:.4f}")
    print(f"    min  / max :  {cos['min']:.4f} / {cos['max']:.4f}")
    print()
    print("  L2 distance (per token):")
    print(f"    mean ± std :  {l2['mean']:.4f} ± {l2['std']:.4f}")
    print(f"    min  / max :  {l2['min']:.4f} / {l2['max']:.4f}")
    print()
    print(f"  PCA subspace overlap (top-{args.pca_k} directions):")
    print(f"    mean cos²  :  {overlap:.4f}  (1.0 = identical subspace)")
    print()
    print("  Interpretation:")
    print(f"    {interpret_cosine(cos['mean'])}")
    print("=" * 60)

    # Individual token cosine distribution
    a_norm = F.normalize(gt_tokens, dim=-1)
    b_norm = F.normalize(blur_tokens, dim=-1)
    per_token_cos = (a_norm * b_norm).sum(dim=-1).numpy()
    buckets = [
        (0.95, 1.01, "> 0.95"),
        (0.80, 0.95, "0.80–0.95"),
        (0.60, 0.80, "0.60–0.80"),
        (-1.0, 0.60, "< 0.60"),
    ]
    print("\n  Token cosine distribution:")
    for lo, hi, label in buckets:
        count = int(((per_token_cos >= lo) & (per_token_cos < hi)).sum())
        pct = 100.0 * count / len(per_token_cos)
        print(f"    {label:12s}  {count:5d} tokens  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
