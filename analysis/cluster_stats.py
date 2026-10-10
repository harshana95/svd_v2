"""Scene-clustered statistics for paired clip-level comparisons.

WHY: the held-out evals score several clips per scene -- consecutive bursts whose
centres are only ~8 frames apart and whose spans overlap -- so clips within a scene
are strongly correlated. A plain paired t-test over clips treats them as independent
and overstates significance. The honest unit of replication is the SCENE.

Held-out set sizes in scenes (not clips): FMO/TbD 4, Adobe240 test 8, GoPro test 11.

Two estimators, both scene-level:
  * scene-mean paired t-test: average each scene's clip deltas, then a one-sample
    t-test over the G scene means (df = G-1). Conservative; the standard fix.
  * cluster bootstrap: resample SCENES with replacement, pool their clips, take the
    clip-weighted mean delta; report the 95% percentile interval.
With few scenes (G <= ~5) neither has much power -- that is the honest finding, not a
defect of the estimator.
"""
import re
import numpy as np


def scene_of(stem: str) -> str:
    """Scene id from a clip stem.
    FMO/TbD  'HighFPS_GT_out2_c25_burst3'      -> 'HighFPS_GT_out2'
    GoPro    'test_GOPR0384_11_00_c006_burst3' -> 'test_GOPR0384_11_00'
    Adobe240 'GOPR9637a_00761'                 -> 'GOPR9637a'
    """
    s, n = re.subn(r"_c\d+_burst\d+$", "", stem)
    if n:
        return s
    return re.sub(r"_\d+$", "", stem)


def cluster_paired(deltas, scenes, n_boot=20000, seed=0):
    """deltas: per-clip paired differences (B minus A); scenes: per-clip scene ids."""
    from scipy import stats
    d = np.asarray(deltas, dtype=np.float64)
    sc = np.asarray(scenes)
    uniq = sorted(set(sc.tolist()))
    G = len(uniq)
    means = np.array([d[sc == u].mean() for u in uniq])
    out = {"n_clips": len(d), "n_scenes": G, "clip_mean": float(d.mean()),
           "scene_mean": float(means.mean())}
    if G < 2:
        out.update(t=float("nan"), df=0, p=float("nan"), ci=(float("nan"), float("nan")),
                   up_scenes=int((means > 0).sum()))
        return out
    se = means.std(ddof=1) / np.sqrt(G)
    t = means.mean() / se if se > 0 else float("nan")
    out["t"], out["df"] = float(t), G - 1
    out["p"] = float(2 * stats.t.sf(abs(t), G - 1)) if np.isfinite(t) else float("nan")
    rng = np.random.default_rng(seed)
    groups = [d[sc == u] for u in uniq]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, G, G)
        boots[i] = np.concatenate([groups[j] for j in pick]).mean()
    out["ci"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    out["up_scenes"] = int((means > 0).sum())
    return out


def fmt(res, label="CLUSTERED"):
    lo, hi = res["ci"]
    return (f"{label} ({res['n_scenes']} scenes / {res['n_clips']} clips): "
            f"delta {res['clip_mean']:+.4f}  scene-mean t={res['t']:.2f} (df={res['df']}) "
            f"p={res['p']:.3f}  boot95=[{lo:+.4f},{hi:+.4f}]  "
            f"{res['up_scenes']}/{res['n_scenes']} scenes up")
