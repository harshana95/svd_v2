#!/usr/bin/env python3
"""
X6: blurred-input reference row -- PSNR/SSIM/MSE between the blurred input and GT,
the floor every deblurring method must clear.

CPU-only, no GPU/model downloads needed. Deliberately avoids analysis/calculate_metrics.py,
which walks up the pred_folder path looking for an ancestor literally named "Results/" and
spins forever (0% GPU, 99% CPU) when that assumption doesn't hold -- exactly what happened
when it was pointed at eval_sets/gopro_test/blur directly.

Usage
-----
  python playground/floor_metrics.py \\
      --gt-dir /depot/chan129/users/harshana/eval_sets/gopro_test/gt \\
      --blur-dir /depot/chan129/users/harshana/eval_sets/gopro_test/blur \\
      --out-csv playground/floor_metrics_gopro.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.misc import calculate_mse, calculate_psnr, calculate_ssim


def load_01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)[None] / 255.0


def main():
    parser = argparse.ArgumentParser(description="Blurred-input floor metrics (PSNR/SSIM/MSE).")
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--blur-dir", required=True)
    parser.add_argument("--crop", type=int, default=30)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    gt_paths = sorted(Path(args.gt_dir).glob("*"))
    blur_map = {p.name: p for p in Path(args.blur_dir).glob("*")}
    pairs = [(p, blur_map[p.name]) for p in gt_paths if p.name in blur_map]
    if len(pairs) < len(gt_paths):
        print(f"[warn] {len(gt_paths) - len(pairs)} GT images have no matching blur file.")
    print(f"{len(pairs)} GT/blur pairs from {args.gt_dir} vs {args.blur_dir}")

    rows = []
    for gt_path, blur_path in pairs:
        gt = load_01(gt_path)
        blur = load_01(blur_path)
        if gt.shape != blur.shape:
            h, w = min(gt.shape[1], blur.shape[1]), min(gt.shape[2], blur.shape[2])
            gt, blur = gt[:, :h, :w], blur[:, :h, :w]
        psnr = calculate_psnr(gt, blur, crop=args.crop)
        mse = calculate_mse(gt, blur, crop=args.crop)
        ssim_val, _ = calculate_ssim(gt[0], blur[0], crop=args.crop)
        rows.append((gt_path.name, float(psnr), float(mse), float(ssim_val)))

    psnrs = np.array([r[1] for r in rows])
    mses = np.array([r[2] for r in rows])
    ssims = np.array([r[3] for r in rows])

    print(f"\nBlurred-input floor over {len(rows)} pairs:")
    print(f"  PSNR: {psnrs.mean():.3f} +/- {psnrs.std():.3f} dB")
    print(f"  SSIM: {ssims.mean():.4f} +/- {ssims.std():.4f}")
    print(f"  MSE:  {mses.mean():.3f} +/- {mses.std():.3f}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("name,psnr,mse,ssim\n")
            for name, psnr, mse, ssim_val in rows:
                f.write(f"{name},{psnr:.4f},{mse:.4f},{ssim_val:.4f}\n")
        print(f"Per-image CSV saved -> {out_path}")


if __name__ == "__main__":
    main()
