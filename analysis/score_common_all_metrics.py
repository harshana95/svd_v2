"""All slide metrics on the S=1 COMMON test set, every method scored against the same 9-frame GT window.

Time alignment (user decision 2026-10-02: score by timestamp, never by truncation/resampling):
  blur = mean of an 11-frame exposure; GT9 = exposure offsets 1..9 (gt9/gt9_<stem>.mp4, exported by
  make_common_testset.py --gt9-only).  GT9 index i  <->  exposure time t_i = (i + 1 + 0.5) / 11.
    ours / extra arms : 9 frames        -> GT9 idx 0..8 (exact)
    jin2018           : 7 frames only   -> GT9 idx [0,1,3,4,5,7,8] (the 7-frame GT subsample)
    tedla ("present") : 16 frames spanning the WHOLE exposure -> frame k_i = floor(16 * t_i), nearest in time
  Per-frame metrics, flow pairs and corr use ONLY these aligned frames.  Reblur error and FVD use the
  method's NATIVE frames (Tedla 16, ours 9, Jin 7 -> Jin has no FVD: < I3D's minimum length).
  Predictions are resized to the GT's H,W (undoes Tedla's 720x1280 resize of the square FMO crops).

Metrics per clip: corr (moving-content correlation) | PSNR / SSIM / LPIPS (forward, whole frame; SSIM on every 3rd
aligned frame) | the same three in the Tedla/blur2vid PATCH protocol (40px patches, whole-clip direction chosen by
PSNR, i.e. direction-BLIND) | RAFT flow EPE: consecutive aligned pairs (forward), first-last pair forward / reversed /
min (= Tedla protocol, direction-blind) | reblur error L1 (x255) | reversal flags (PSNR, corr, flow) | FVD (I3D,
pooled per dataset and over all clips; unstable at n<=50).  GT9's own reblur error is reported as a reference row.
RAFT inputs are normalised to [-1,1] (torchvision's contract); the older compute_video_metrics.py feeds [0,1], so
absolute EPE differs from the 2026-09 single-image tables -- only compare within one run of this script.

Run on a GPU node in the metrics env (lpips/cdfvd/RAFT live there; never the entry node):
  /depot/chan129/users/harshana/conda-env/metrics/bin/python analysis/score_common_all_metrics.py \
      --datasets gopro adobe fmo_tbd --out /scratch/.../common_all_metrics.json
  --limit 2 --no-fvd   for a smoke test.   --extra "LABEL=/path/with/{ds}/glob" adds a 9-frame Wan-style arm
  (e.g. the strict no-burst LoRA run: videos/wan_output_<name>_*.mp4 under each matching dir).
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_control_flanked as cf
import score_common_testset as sc
import cluster_stats

E = sc.E
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ("gopro", "adobe", "fmo_tbd")
GT7_IDX = [0, 1, 3, 4, 5, 7, 8]            # the 7-frame GT = these indices of the 9-frame GT
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- alignment
def gt_times():
    return [(i + 1 + 0.5) / 11.0 for i in range(9)]


def positions(method, n_frames):
    """-> (gt9 indices, pred frame indices) of the frames that sit at the same exposure time."""
    if method == "tedla":
        assert n_frames == 16, f"tedla expected 16 frames, got {n_frames}"
        return list(range(9)), [int(min(15, max(0, np.floor(16 * t)))) for t in gt_times()]
    if method == "jin2018":
        assert n_frames == 7, f"jin2018 expected 7 frames, got {n_frames}"
        return GT7_IDX, list(range(7))
    assert n_frames == 9, f"{method} expected 9 frames, got {n_frames}"
    return list(range(9)), list(range(9))


def resize_to(frames, hw):
    """[T,H,W,3] float [0,1] -> [T,h,w,3] (bicubic, antialiased), on the GPU."""
    if frames.shape[1:3] == tuple(hw):
        return frames
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).to(DEV)
    t = F.interpolate(t, size=tuple(hw), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    return t.permute(0, 2, 3, 1).cpu().numpy()


# ---------------------------------------------------------------- basic metrics
def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def mean_psnr(g, p):
    return float(np.mean([psnr(g[t], p[t]) for t in range(len(g))]))


def ssim_frames(g, p, stride=3):
    from skimage.metrics import structural_similarity
    return float(np.mean([structural_similarity(g[t], p[t], channel_axis=2, data_range=1.0) for t in range(0, len(g), stride)]))


def lpips_frames(g, p, lp):
    out = []
    with torch.no_grad():
        for t in range(len(g)):
            a = torch.from_numpy(g[t]).permute(2, 0, 1)[None].to(DEV) * 2 - 1
            b = torch.from_numpy(p[t]).permute(2, 0, 1)[None].to(DEV) * 2 - 1
            out.append(float(lp(a, b).item()))
    return float(np.mean(out))


def patch_starts(size, patch):
    if size <= patch:
        return [0]
    s = list(range(0, size - patch + 1, patch))
    if s[-1] + patch < size:
        s.append(size - patch)
    return s


def patch_protocol(g, p, lp, patch=40, stride=3):
    """Tedla/blur2vid protocol: score forward and time-reversed pred, keep the better WHOLE-clip direction (by PSNR),
    then patch PSNR / SSIM / LPIPS on every `stride`-th frame. Returns (psnr, ssim, lpips, direction)."""
    from skimage.metrics import structural_similarity
    pf, pr = mean_psnr(g, p), mean_psnr(g, p[::-1])
    d, pu = ("forward", p) if pf >= pr else ("reversed", p[::-1])
    H, W = g.shape[1:3]
    rows, cols = patch_starts(H, patch), patch_starts(W, patch)
    ps, ss, ls = [], [], []
    for f in range(0, len(g), stride):
        gp = np.stack([g[f, r:r + patch, c:c + patch] for r in rows for c in cols])
        pp = np.stack([pu[f, r:r + patch, c:c + patch] for r in rows for c in cols])
        mse = ((gp - pp) ** 2).reshape(len(gp), -1).mean(1)
        ps.append(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-10))))
        ss.append(np.mean([structural_similarity(a, b, data_range=1.0, channel_axis=2) for a, b in zip(gp, pp)]))
        with torch.no_grad():
            a = torch.from_numpy(gp).permute(0, 3, 1, 2).to(DEV) * 2 - 1
            b = torch.from_numpy(pp).permute(0, 3, 1, 2).to(DEV) * 2 - 1
            ls.append(float(lp(a, b).mean().item()))
    return float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls)), d


# ---------------------------------------------------------------- flow
class Flow:
    def __init__(self, norm="pm1"):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self.m = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().to(DEV)
        self.norm = norm

    def prep(self, im):
        t = torch.from_numpy(im).permute(2, 0, 1)[None].float().to(DEV)
        h, w = t.shape[2] // 8 * 8, t.shape[3] // 8 * 8
        t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False) if (h, w) != tuple(t.shape[2:]) else t
        return t * 2 - 1 if self.norm == "pm1" else t

    @torch.no_grad()
    def __call__(self, a, b):
        return self.m(self.prep(a), self.prep(b))[-1]


def epe(f1, f2):
    return float(torch.norm(f1 - f2, dim=1).mean().item())


# ---------------------------------------------------------------- per-clip scoring
def score_clip(method, g9, native, blur, flow, gt_flow_cache, lp, do_patch):
    """g9: [9,H,W,3] GT. native: [T,H,W,3] the method's own frames resized to GT size. blur: [H,W,3]."""
    gi, pi = positions(method, len(native))
    g, p = g9[gi], native[pi]
    r = {"n_native": int(len(native)), "n_aligned": len(gi)}
    mg, mp = cf.moving_map(g), cf.moving_map(p)
    r["corr"] = cf.corr(mg, mp)
    r["corr_rev"] = cf.corr(mg, mp[::-1])
    r["psnr"], r["psnr_rev"] = mean_psnr(g, p), mean_psnr(g, p[::-1])
    r["ssim"] = ssim_frames(g, p)
    r["lpips"] = lpips_frames(g, p, lp)
    if do_patch:
        r["patch_psnr"], r["patch_ssim"], r["patch_lpips"], r["patch_dir"] = patch_protocol(g, p, lp)
    # flow: consecutive aligned pairs + first-last pair, GT flow at the SAME pair of GT frames
    def gtf(a, b):
        if (a, b) not in gt_flow_cache:
            gt_flow_cache[(a, b)] = flow(g9[a], g9[b])
        return gt_flow_cache[(a, b)]
    cons, cons_rev = [], []
    for k in range(len(gi) - 1):
        fg = gtf(gi[k], gi[k + 1])
        cons.append(epe(flow(p[k], p[k + 1]), fg))
        cons_rev.append(epe(flow(p[k + 1], p[k]), fg))
    fg = gtf(gi[0], gi[-1])
    fl_f, fl_r = epe(flow(p[0], p[-1]), fg), epe(flow(p[-1], p[0]), fg)
    r.update(epe_consec=float(np.mean(cons)), epe_consec_rev=float(np.mean(cons_rev)),
             epe_fl=fl_f, epe_fl_rev=fl_r, epe_fl_min=min(fl_f, fl_r))
    # reblur: mean of the method's NATIVE frames vs the input blur
    r["reblur"] = float(np.abs(native.mean(0) - blur).mean() * 255)
    # reversal flags (True = the reversed order matches better)
    r["rev_psnr"] = bool(r["psnr_rev"] > r["psnr"])
    r["rev_corr"] = bool(r["corr_rev"] > r["corr"])
    r["rev_flow"] = bool(fl_r < fl_f)
    return r


def load_blur(ds, stem, hw):
    from PIL import Image
    im = np.asarray(Image.open(f"{E}/{ds}/blur_inputs/{stem}.png").convert("RGB")).astype(np.float32) / 255.0
    return im


def extra_pred(template, ds, stem):
    name = stem.split("_", 2)[2] if ds in ("gopro", "adobe") else stem
    for d in sorted(glob.glob(template.format(ds=ds)), key=os.path.getmtime)[::-1]:
        c = [x for x in glob.glob(f"{d}/videos/wan_output_{name}_*.mp4")
             if "flow" not in x and "with_deblur_prefix" not in x and "pred_vs_gt" not in x]
        if c:
            return c[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DS))
    ap.add_argument("--methods", nargs="+", default=["jin2018", "tedla", "ours_nodyn", "ours_stagec", "ours_oracle"])
    ap.add_argument("--extra", action="append", default=[], metavar="LABEL=GLOB_TEMPLATE_WITH_{ds}")
    ap.add_argument("--limit", type=int, default=0, help="clips per dataset (smoke test)")
    ap.add_argument("--no-fvd", action="store_true"); ap.add_argument("--no-patch", action="store_true")
    ap.add_argument("--raft-norm", default="pm1", choices=["pm1", "01"])
    ap.add_argument("--out", default=f"{sc.S}/common_all_metrics.json")
    a = ap.parse_args()

    import lpips
    lp = lpips.LPIPS(net="alex").to(DEV).eval()
    flow = Flow(a.raft_norm)
    ours_cache = {arm: {ds: sc.ours_dir(arm, ds) for ds in DS} for arm in ("nodyn", "stagec", "oracle")}
    extras = dict(x.split("=", 1) for x in a.extra)
    methods = list(a.methods) + list(extras)
    fvd_objs = {}
    res = {ds: {m: {} for m in methods + ["gt9_ref"]} for ds in a.datasets}

    def fvd_add(key, real, fake):
        if a.no_fvd:
            return
        from cdfvd import fvd as cdfvd_mod
        if key not in fvd_objs:
            fvd_objs[key] = cdfvd_mod.cdfvd(model="i3d", device=DEV)
        o = fvd_objs[key]
        o.add_real_stats(np.expand_dims((real * 255).round().astype(np.uint8), 0))
        o.add_fake_stats(np.expand_dims((fake * 255).round().astype(np.uint8), 0))

    t0 = time.time()
    for ds in a.datasets:
        man = json.load(open(f"{ROOT}/findings/common_testset/manifest_{ds}.json"))
        clips = [c["stem"] for c in man["clips"]]
        clips = [s for s in clips if os.path.exists(f"{E}/{ds}/gt9/gt9_{s}.mp4")]
        if a.limit:
            clips = clips[:a.limit]
        for ci, stem in enumerate(clips):
            g9 = cf.read(f"{E}/{ds}/gt9/gt9_{stem}.mp4")
            hw = g9.shape[1:3]
            blur = load_blur(ds, stem, hw)
            if blur.shape[:2] != tuple(hw):
                blur = resize_to(blur[None], hw)[0]
            gt_flow_cache = {}
            res[ds]["gt9_ref"][stem] = {"reblur": float(np.abs(g9.mean(0) - blur).mean() * 255)}
            for m in methods:
                if m in extras:
                    path = extra_pred(extras[m], ds, stem)
                else:
                    path = sc.method_pred(m, ds, stem, ours_cache if m.startswith("ours_") else None)
                if not path or not os.path.exists(path):
                    print(f"[skip] {m} {ds} {stem}: no prediction"); continue
                native = resize_to(cf.read(path), hw)
                r = score_clip(m, g9, native, blur, flow, gt_flow_cache, lp, not a.no_patch)
                if m != "jin2018":       # Jin has 7 frames < I3D's minimum
                    fvd_add((m, ds), g9, native); fvd_add((m, "ALL"), g9, native)
                res[ds][m][stem] = r
            print(f"{ds} {ci + 1}/{len(clips)} {stem[:40]} t={time.time() - t0:.0f}s", flush=True)

    fvd = {}
    for key, o in fvd_objs.items():
        try:
            fvd[f"{key[0]}|{key[1]}"] = float(o.compute_fvd_from_stats())
        except Exception as e:  # noqa
            fvd[f"{key[0]}|{key[1]}"] = f"failed: {e}"
    out = {"args": vars(a), "per_clip": res, "fvd": fvd}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    summarize(out)


KEYS = ["corr", "psnr", "ssim", "lpips", "patch_psnr", "patch_ssim", "patch_lpips", "epe_consec", "epe_fl", "epe_fl_min", "reblur"]


def summarize(out):
    res = out["per_clip"]
    for ds, ms in res.items():
        print(f"\n=== {ds} ===")
        print(f"{'method':14}{'n':>3}" + "".join(f"{k:>12}" for k in KEYS) + f"{'revP':>6}{'revC':>6}{'revF':>6}")
        for m, d in ms.items():
            if not d:
                continue
            v = list(d.values())
            row = f"{m:14}{len(v):>3}"
            for k in KEYS:
                xs = [x[k] for x in v if k in x]
                row += f"{np.mean(xs):12.3f}" if xs else f"{'-':>12}"
            for k in ("rev_psnr", "rev_corr", "rev_flow"):
                xs = [x[k] for x in v if k in x]
                row += f"{sum(xs):>3}/{len(xs):<2}" if xs else f"{'-':>6}"
            print(row)
    print("\nFVD (I3D, pooled):", json.dumps(out["fvd"], indent=1))


if __name__ == "__main__":
    main()
