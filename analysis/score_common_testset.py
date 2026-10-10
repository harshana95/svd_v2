"""Score every method on the COMMON test set (findings/common_testset/manifest_*.json), single scorer.
Methods, per dataset:
  control  = /scratch/.../common_testset/<ds>/control/pred_<stem>.mp4         (blur x7 floor)
  jin2018  = gopro/adobe: /scratch/.../baselines/jin2018/out/<ds>/pred_<stem>.mp4 (pre-existing, stem-identical to the common set)
             fmo_tbd:     /scratch/.../common_testset/fmo_tbd/jin2018/pred_<stem>.mp4
  tedla    = /scratch/.../common_testset/<ds>/tedla/pred_<stem>.mp4
  ours_*   = experiments/infer/COMMON_<ARM>_<ds>_S1_*/videos/wan_output_<name>_*.mp4 (name = stem minus the ds_index_ prefix for gopro/adobe)
GT (7 frames) = /scratch/.../common_testset/<ds>/gt_<stem>.mp4 for all methods -- same instrument, same target, for everyone.
Usage: python analysis/score_common_testset.py [--workers 6]
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_stats
import score_control_flanked as cf
from concurrent.futures import ProcessPoolExecutor

S = "/scratch/gilbreth/wweligam"
E = f"{S}/common_testset"


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def ours_dir(arm, ds):
    d = sorted(glob.glob(f"{S}/experiments/infer/COMMON_{arm.upper()}_{ds}_S1_2*"), key=os.path.getmtime)
    return d[-1] if d else None


def method_pred(method, ds, stem, ours_cache):
    if method == "control":
        return f"{E}/{ds}/control/pred_{stem}.mp4"
    if method == "jin2018":
        # gopro/adobe: rebuilt on corrected blur_inputs after the pm1-vs-01 GT/input export bugfix
        # (2026-09-27, see findings/Wan22VJEPADynamics_findings.md). fmo_tbd was never affected.
        if ds == "fmo_tbd":
            return f"{E}/fmo_tbd/jin2018/pred_{stem}.mp4"
        return f"{E}/{ds}/jin2018_v2/pred_{stem}.mp4"
    if method == "tedla":
        if ds == "fmo_tbd":
            return f"{E}/fmo_tbd/tedla/pred_{stem}.mp4"
        return f"{E}/{ds}/tedla_v2/pred_{stem}.mp4"
    arm = method[len("ours_"):]
    d = ours_cache[arm][ds]
    if d is None:
        return None
    name = stem.split("_", 2)[2] if ds in ("gopro", "adobe") else stem
    cands = glob.glob(f"{d}/videos/wan_output_{name}_*.mp4")
    cands = [c for c in cands if "flow" not in c and "with_deblur_prefix" not in c and "pred_vs_gt" not in c]
    return cands[0] if cands else None


def _resize_to(frames, hw):
    # Baselines output a fixed 1280x720 (blur2vid_baseline convention); our GT is 736 (Wan's /16 rounding).
    # Bicubic-resize the SMALLER one up to the larger's H,W so no content is cropped away.
    from PIL import Image
    h, w = hw
    if frames.shape[1:3] == (h, w):
        return frames
    u8 = (frames.clip(0, 1) * 255).round().astype(np.uint8)  # cf.read returns float32 in [0,1]
    out = np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.BICUBIC)) for f in u8])
    return out.astype(np.float32) / 255.0


def score_one(args):
    method, ds, stem, pred_path, gt_path = args
    gt, pr = cf.read(gt_path), cf.read(pred_path)
    n = min(len(gt), len(pr)); gt, pr = gt[:n], pr[:n]
    h, w = max(gt.shape[1], pr.shape[1]), max(gt.shape[2], pr.shape[2])
    gt, pr = _resize_to(gt, (h, w)), _resize_to(pr, (h, w))
    mg, mp = cf.moving_map(gt), cf.moving_map(pr)
    p = float(np.mean([psnr(gt[t], pr[t]) for t in range(n)]))
    return method, ds, stem, cf.corr(mg, mp), p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    methods = ["control", "jin2018", "tedla", "ours_nodyn", "ours_stagec", "ours_oracle"]
    ours_cache = {arm: {ds: ours_dir(arm, ds) for ds in ("gopro", "adobe", "fmo_tbd")} for arm in ("nodyn", "stagec", "oracle")}

    jobs = []
    for ds in ("gopro", "adobe", "fmo_tbd"):
        import json
        man = json.load(open(f"/home/wweligam/svd_v2/findings/common_testset/manifest_{ds}.json"))
        for c in man["clips"]:
            stem = c["stem"]; gt_path = f"{E}/{ds}/gt_{stem}.mp4"
            if not os.path.exists(gt_path):
                continue
            for m in methods:
                cache = ours_cache.get(m[len("ours_"):], {}) if m.startswith("ours_") else {}
                p = method_pred(m, ds, stem, ours_cache if m.startswith("ours_") else None)
                if p and os.path.exists(p):
                    jobs.append((m, ds, stem, p, gt_path))

    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(score_one, jobs))

    by = {}  # (method, ds) -> {stem: (corr, psnr)}
    for method, ds, stem, corr, p in res:
        by.setdefault((method, ds), {})[stem] = (corr, p)

    for ds in ("gopro", "adobe", "fmo_tbd"):
        present = [m for m in methods if (m, ds) in by]
        if not present:
            continue
        stems = sorted(set.intersection(*[set(by[(m, ds)]) for m in present]))
        scenes = [cluster_stats.scene_of(s) for s in stems]
        print(f"\n=== {ds}  ({len(stems)} clips common to all present methods, {len(set(scenes))} scenes) ===")
        print(f"{'method':14} {'n':>3} {'corr':>7} {'PSNR':>7}")
        base = np.array([by[("control", ds)][s][0] for s in stems]) if ("control", ds) in by else None
        for m in present:
            c = np.array([by[(m, ds)][s][0] for s in stems]); p = np.array([by[(m, ds)][s][1] for s in stems])
            line = f"{m:14} {len(stems):>3} {c.mean():7.3f} {p.mean():7.2f}"
            if base is not None and m != "control":
                r = cluster_stats.cluster_paired(c - base, scenes)
                line += f"   d-vs-control={r['clip_mean']:+.3f} scene_p={r['p']:.3f}"
            print(line)
        missing = [m for m in methods if (m, ds) not in by]
        if missing:
            print(f"  (missing for {ds}: {missing})")


if __name__ == "__main__":
    main()
