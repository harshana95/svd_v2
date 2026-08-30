#!/usr/bin/env python3
"""Score predictions against ground truth by FILENAME, at native resolution.

Written 2026-08-29 because analysis/calculate_metrics.py cannot be used for this:
it pairs images by position in a sorted listing (wrong whenever a run is truncated),
force-resizes to a square (which corrupts PSNR/SSIM on 1280x720), and its main()
loops forever unless the path contains a directory literally named "Results".

Pairs on the filename intersection, so a partial run scores correctly against the
full GT folder. Optionally also scores a degraded-input folder as the reference
floor every method must clear -- worth always reporting, since a generative
restorer can land below it.

    python analysis/eval_pairs.py --gt <dir> --pred <dir> [--lq <dir>] [--crop 30] [--lpips]
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image

def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0

def crop_border(a, c):
    return a[c:a.shape[0]-c, c:a.shape[1]-c] if c > 0 else a

def center_match(gt, pr):
    """Center-crop the larger array to the smaller.

    A model whose latent grid is height//16 silently truncates any dimension that is
    not a multiple of 16 -- MIOR is 1920x1080 and returns 1072 rows. Verified
    empirically on 2026-08-29 that the kept region is CENTERED: sweeping the row
    offset peaks at 4 = (1080-1072)/2 and falls off symmetrically either side.
    """
    h = min(gt.shape[0], pr.shape[0]); w = min(gt.shape[1], pr.shape[1])
    def cc(a):
        t = (a.shape[0] - h) // 2; l = (a.shape[1] - w) // 2
        return a[t:t+h, l:l+w]
    return cc(gt), cc(pr)

def psnr(gt, pr):
    mse = float(np.mean((gt - pr) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--lq", default=None, help="degraded inputs, scored as the floor")
    ap.add_argument("--crop", type=int, default=30)
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()

    exts = (".png", ".jpg", ".jpeg", ".tiff")
    names = lambda d: {f for f in os.listdir(d) if f.lower().endswith(exts)}
    gt_n, pr_n = names(a.gt), names(a.pred)
    common = sorted(gt_n & pr_n)
    if not common:
        sys.exit("no filename overlap between --gt and --pred")
    print(f"gt={len(gt_n)} pred={len(pr_n)} matched={len(common)} crop={a.crop}")
    if len(common) < len(pr_n):
        print(f"  warning: {len(pr_n)-len(common)} predictions have no GT counterpart")

    from skimage.metrics import structural_similarity as ssim_fn
    lp = None
    if a.lpips:
        import torch, lpips as lpips_lib
        lp = lpips_lib.LPIPS(net="alex")
        lp.eval()

    rows, floor_rows = [], []
    for i, n in enumerate(common, 1):
        g = crop_border(load(os.path.join(a.gt, n)), a.crop)
        p = crop_border(load(os.path.join(a.pred, n)), a.crop)
        if g.shape != p.shape:
            if i == 1:
                print(f"  size mismatch (pred {p.shape[:2]} vs gt {g.shape[:2]}) "
                      f"-> center-cropping both to the common region")
            g, p = center_match(g, p)
        r = {"name": n, "psnr": psnr(g, p),
             "ssim": float(ssim_fn(g, p, channel_axis=2, data_range=1.0))}
        if lp is not None:
            import torch
            t = lambda x: torch.from_numpy(x.transpose(2,0,1)[None]).float()*2-1
            with torch.no_grad():
                r["lpips"] = float(lp(t(g), t(p)).item())
        rows.append(r)
        if a.lq:
            lqp = os.path.join(a.lq, n)
            if os.path.isfile(lqp):
                l = crop_border(load(lqp), a.crop)
                gl, l = center_match(g, l) if l.shape != g.shape else (g, l)
                floor_rows.append({"psnr": psnr(gl, l),
                                   "ssim": float(ssim_fn(gl, l, channel_axis=2, data_range=1.0))})
        if i % 50 == 0: print(f"  {i}/{len(common)}")

    def agg(rs, keys):
        return {k: {"mean": float(np.mean([r[k] for r in rs])),
                    "std":  float(np.std([r[k] for r in rs]))} for k in keys if k in rs[0]}

    keys = ["psnr", "ssim"] + (["lpips"] if lp is not None else [])
    summary = {"label": a.label or os.path.basename(a.pred.rstrip("/")),
               "n": len(rows), "crop": a.crop, "metrics": agg(rows, keys)}
    if floor_rows:
        summary["input_floor"] = agg(floor_rows, ["psnr", "ssim"])
        summary["input_floor"]["n"] = len(floor_rows)

    print("\n== %s (n=%d) ==" % (summary["label"], summary["n"]))
    for k, v in summary["metrics"].items():
        print("   %-6s %8.4f ± %.4f" % (k, v["mean"], v["std"]))
    if floor_rows:
        f = summary["input_floor"]
        print("   -- degraded-input floor --")
        print("   %-6s %8.4f ± %.4f" % ("psnr", f["psnr"]["mean"], f["psnr"]["std"]))
        print("   %-6s %8.4f ± %.4f" % ("ssim", f["ssim"]["mean"], f["ssim"]["std"]))
        d = summary["metrics"]["psnr"]["mean"] - f["psnr"]["mean"]
        print("   delta vs input: %+.2f dB %s" % (d, "" if d > 0 else "  <-- BELOW the input"))

    if a.out:
        json.dump({"summary": summary, "per_image": rows}, open(a.out, "w"), indent=1)
        print("\nwrote", a.out)

if __name__ == "__main__":
    main()
