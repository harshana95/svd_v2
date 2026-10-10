"""Control-flanked "did it render the motion" score for blur2video outputs.

PSNR/SSIM are useless on small-object motion: the object is a tiny fraction of
the pixels, so a model that renders none of it still scores 40dB (see
findings/Wan22VJEPADynamics_findings.md, 2026-09-21). This scores whether the
PREDICTED moving content lands where the GT moving content is.

For each clip:
  moving-content map  m[t] = | frame[t] - temporal_median(clip) |   (greyscale)
  score(A, B)         = mean over t of corr( m_A[t], m_B[t] ) over pixels

and reports it between two controls computed on the SAME clip:
  negative  = GT vs time-reversed GT  -> what a plausible-but-wrong motion scores
  positive  = GT vs GT shifted +2     -> what a near-perfect motion scores

The normalized score (pred - neg) / (pos - neg) is what makes two passes
comparable when they differ in resolution or upsampling factor -- absolute
correlations are NOT comparable across such passes, the controls are the whole
point. A clip whose positive control is not clearly above its negative control
has no discriminating power and is dropped.

Usage:
  python analysis/score_control_flanked.py RUN_A RUN_B [--label-a X --label-b Y]
where each RUN is an experiment dir containing videos/{gt_*,wan_output_*}.mp4
"""
import argparse, glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_stats

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio


def read(path):
    v = np.asarray(iio.imread(path))          # [T,H,W,3] uint8
    return v.astype(np.float32) / 255.0


def moving_map(v):
    """[T,H,W,3] -> [T,H,W] deviation from the clip's temporal median."""
    g = v.mean(axis=3)
    return np.abs(g - np.median(g, axis=0, keepdims=True))


def corr(a, b):
    a = a.reshape(len(a), -1); b = b.reshape(len(b), -1)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    a = a - a.mean(axis=1, keepdims=True); b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1)) + 1e-9
    return float((num / den).mean())


def stem_of(p):
    b = os.path.basename(p)
    b = re.sub(r"^(gt|wan_output)_", "", b)
    return re.sub(r"_\d+k\.mp4$", "", b)


def score_run(run, max_frames=None):
    """-> {stem: (pred, neg, pos)}"""
    vd = os.path.join(run, "videos")
    out = {}
    for gt_path in sorted(glob.glob(os.path.join(vd, "gt_*.mp4"))):
        if "gt_flow" in gt_path:
            continue
        stem = stem_of(gt_path)
        pred_path = os.path.join(vd, os.path.basename(gt_path).replace("gt_", "wan_output_", 1))
        if not os.path.exists(pred_path):
            continue
        gt, pr = read(gt_path), read(pred_path)
        if max_frames:
            # Common-window scoring: clips of different generated length (burst-size
            # sweep) are trimmed to the same first N frames. The GT prefix is IDENTICAL
            # across arms (truncation takes wan_frames[:F_S]), so this compares the
            # SAME content under different burst context, removing the
            # 'longer video is harder' confound. NB the temporal median and the
            # controls are then computed over the trimmed window for both arms.
            gt, pr = gt[:max_frames], pr[:max_frames]
        mg, mp = moving_map(gt), moving_map(pr)
        out[stem] = (corr(mg, mp), corr(mg, mg[::-1]), corr(mg[:-2], mg[2:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a"); ap.add_argument("run_b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="trim every clip to its first N frames before scoring")
    ap.add_argument("--min-gap", type=float, default=0.05,
                    help="require pos-neg above this, else the clip cannot discriminate")
    a = ap.parse_args()
    A, B = score_run(a.run_a, a.max_frames), score_run(a.run_b, a.max_frames)
    shared = sorted(set(A) & set(B))
    print(f"{a.label_a}: {len(A)} clips   {a.label_b}: {len(B)} clips   paired: {len(shared)}\n")
    rows = []
    for s in shared:
        pa, na, poa = A[s]; pb, nb, pob = B[s]
        if (poa - na) < a.min_gap or (pob - nb) < a.min_gap:
            continue                      # no discriminating power on this clip
        za = (pa - na) / (poa - na)
        zb = (pb - nb) / (pob - nb)
        rows.append((s, pa, na, poa, za, pb, nb, pob, zb))
    if not rows:
        print("no clip had discriminating power (positive control too weak)"); return
    print(f"{'clip':34} {a.label_a+' norm':>10} {a.label_b+' norm':>10} {'delta':>8}")
    for r in rows:
        print(f"{r[0][:34]:34} {r[4]:>10.3f} {r[8]:>10.3f} {r[8]-r[4]:>8.3f}")
    za = np.array([r[4] for r in rows]); zb = np.array([r[8] for r in rows])
    d = zb - za
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    print(f"\n{'MEAN normalized':34} {za.mean():>10.3f} {zb.mean():>10.3f} {d.mean():>8.3f}")
    print(f"paired delta {d.mean():+.4f} +- {se:.4f} SE   t={d.mean()/se if se else float('nan'):.2f}   "
          f"n={len(d)}   {(d > 0).sum()}/{len(d)} up")

    # --- robustness -------------------------------------------------------
    # The normalized ratio divides by (pos-neg); when that gap is small the
    # ratio explodes (values >1 or <0 in the table above), so the mean is
    # driven by a few unstable clips. Report statistics that do not depend on
    # that denominator, plus a rank test that ignores magnitude entirely.
    from math import comb
    pa = np.array([r[1] for r in rows]); pb = np.array([r[5] for r in rows])
    draw = pb - pa
    sed = draw.std(ddof=1) / np.sqrt(len(draw))
    print(f"\n--- robustness ---")
    print(f"RAW pred correlation (no normalization): {pa.mean():.4f} -> {pb.mean():.4f}"
          f"   delta {draw.mean():+.4f} +- {sed:.4f} SE   t={draw.mean()/sed:.2f}")
    print(f"MEDIAN normalized delta: {np.median(d):+.4f}   (mean {d.mean():+.4f}; "
          f"gap = outlier influence)")
    k, n = int((d > 0).sum()), len(d)
    pv = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    print(f"SIGN test: {k}/{n} up, one-sided p={pv:.3f}")
    try:
        from scipy.stats import wilcoxon
        w = wilcoxon(zb, za)
        print(f"WILCOXON signed-rank on normalized: p={w.pvalue:.3f}")
        w2 = wilcoxon(pb, pa)
        print(f"WILCOXON signed-rank on raw:        p={w2.pvalue:.3f}")
    except Exception as e:
        print(f"(scipy unavailable: {e})")
    # scene-clustered inference: clips within a scene are correlated (overlapping bursts)
    scenes = [cluster_stats.scene_of(r[0]) for r in rows]
    print(cluster_stats.fmt(cluster_stats.cluster_paired(draw, scenes), "CLUSTERED raw"))
    print(cluster_stats.fmt(cluster_stats.cluster_paired(d, scenes), "CLUSTERED normalized"))
    gaps = np.array([min(r[3] - r[2], r[7] - r[6]) for r in rows])
    print(f"control gap (pos-neg) min/median/max: {gaps.min():.3f} / "
          f"{np.median(gaps):.3f} / {gaps.max():.3f}   <- small gap = unstable ratio")
    print("\n(normalized = (pred-neg)/(pos-neg); 0 = no better than a reversed-motion "
          "control, 1 = as good as a 2-frame-shifted GT)")


if __name__ == "__main__":
    main()
