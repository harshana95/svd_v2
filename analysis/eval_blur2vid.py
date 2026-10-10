"""Blur-to-video evaluation v2: defensible headline metrics, full test splits, scene-clustered CIs.

Headline metrics (the home-made moving-content correlation is NOT computed here):
  PSNR / LPIPS (forward, whole frame, time-aligned frames) | FVD (I3D, scene-bootstrap CI) |
  RAFT EPE (pred flow vs GT flow on the aligned frame pairs; forward and direction-invariant) |
  re-blur consistency (L1 x255 between mean(native frames) and the input blur; GT9's own value is the reference row).
Each also reported direction-invariant ("*": the better of forward / time-reversed per clip) because a single blur
is time-symmetric -- forward-only PSNR/EPE punish a valid-but-reversed video.

Statistics: every number is over the INTERSECTION of clips all methods have, scene = the manifest's `scene` field
(NOT cluster_stats.scene_of(stem), which does not strip GoPro's `_c006` and would make each clip its own scene).
CI = scene-cluster bootstrap of the clip-weighted mean; deltas vs --ref use cluster_stats.cluster_paired.
Asserted: the gopro full split has 11 scenes, adobe has 8.

Time alignment (GT9 index i <-> exposure time (i+1.5)/11, centre = idx 4). Each method kind declares its mapping:
  wan9        9 frames, exact (ours, PickMoment queried at the 9 GT times)
  tedla16     16 frames over the whole exposure -> frame floor(16 t_i)
  jin7        7 frames = GT9 idx [0,1,3,4,5,7,8]
  span:N      N<=9 frames evenly covering the exposure, frame k at (k+.5)/N -> nearest GT9 idx, no repeats (Zhong: span:7 -> idx [0,1,2,4,6,7,8])
  i2v         image-to-video from ONE conditioning frame placed at frame 0 = exposure CENTRE (t=.5, idx 4):
              pred k <-> GT idx 4+k for k=0..4 (only 5 frames lie inside the exposure); the reversed direction
              compares pred k with GT idx 4-k. Used for plain-Wan-I2V (cond = the blur) and deblur-then-I2V.
FVD always uses each method's NATIVE frames vs GT9 (needs >= 9 frames; methods with fewer get FVD '-').

Run on a GPU node (sbatch), metrics env (Python 3.9):
  /depot/chan129/users/harshana/conda-env/metrics/bin/python analysis/eval_blur2vid.py --root /scratch/.../common_testset_full \
     --datasets gopro adobe --method "ours_stagec=wan9:/scratch/.../infer/FULL_STAGEC_{ds}_*/videos/wan_output_{name}_*.mp4" \
     --method "tedla=tedla16:{root}/{ds}/tedla/pred_{stem}.mp4" --ref ours_stagec --out results.json
Template fields: {root} {ds} {stem} {name} (= stem minus the `<ds>_<index>_` prefix). A glob takes the newest match.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_stats
import score_control_flanked as cf
import score_common_all_metrics as sa   # reuses Flow / resize_to / mean_psnr unchanged, so numbers match older tables

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = sa.DEV
EXPECT_SCENES = {"gopro": 11, "adobe": 8}
GT_T = [(i + 1.5) / 11.0 for i in range(9)]


def align(kind, n):
    """-> (gi, pi, rev_gi): GT9 indices, matching pred frame indices, GT indices the REVERSED prediction is compared to."""
    if kind == "wan9":
        assert n == 9, f"wan9 expects 9 frames, got {n}"
        return list(range(9)), list(range(9)), list(range(8, -1, -1))
    if kind == "jin7":
        assert n == 7, f"jin7 expects 7 frames, got {n}"
        g = sa.GT7_IDX
        return g, list(range(7)), g[::-1]
    if kind == "tedla16":
        assert n == 16, f"tedla16 expects 16 frames, got {n}"
        return list(range(9)), [int(min(15, np.floor(16 * t))) for t in GT_T], list(range(8, -1, -1))
    if kind.startswith("span:"):
        N = int(kind.split(":")[1])
        assert n == N and N <= 9, f"{kind} expects {N}<=9 frames, got {n}"
        # frame k sits at exposure time (k+.5)/N; pair it with the nearest GT9 time. Driven by the PREDICTED frames so no
        # prediction frame is repeated (a repeated frame gives a zero-flow pair that is scored against non-zero GT flow).
        gi = [int(np.argmin([abs(t - (k + 0.5) / N) for t in GT_T])) for k in range(N)]
        assert len(set(gi)) == N, f"{kind}: nearest-time mapping collides {gi}"
        return gi, list(range(N)), gi[::-1]
    if kind == "i2v":
        assert n >= 5, f"i2v expects >=5 frames, got {n}"
        return [4, 5, 6, 7, 8], [0, 1, 2, 3, 4], [4, 3, 2, 1, 0]
    raise ValueError(kind)


def resolve(template, root, ds, stem):
    name = stem.split("_", 2)[2] if ds in ("gopro", "adobe") else stem
    pat = template.format(root=root, ds=ds, stem=stem, name=name)
    hits = [h for h in glob.glob(pat) if not any(x in h for x in ("flow", "with_deblur_prefix", "pred_vs_gt"))]
    # "newest by mtime" silently picked a stale/washed run when several run dirs matched one template. Refuse instead: the template must
    # match ONE run dir (rename or move the others, or pin the dir in the template). ALLOW_STALE_GLOB=1 restores the old behaviour.
    runs = {os.path.dirname(os.path.dirname(h)) for h in hits}
    if len(runs) > 1 and os.environ.get("ALLOW_STALE_GLOB") != "1":
        raise RuntimeError(f"template {template!r} matches {len(runs)} run dirs: {sorted(os.path.basename(r) for r in runs)}")
    return sorted(hits, key=os.path.getmtime)[-1] if hits else None


def lpips_mean(a, b, lp):
    with torch.no_grad():
        out = []
        for t in range(len(a)):
            x = torch.from_numpy(a[t]).permute(2, 0, 1)[None].to(DEV) * 2 - 1
            y = torch.from_numpy(b[t]).permute(2, 0, 1)[None].to(DEV) * 2 - 1
            out.append(float(lp(x, y).item()))
    return float(np.mean(out))


def patch_minpatch(g, p, pr, lp, patch=40, stride=3):
    """Blur2Vid (Tedla et al. 2025) bidirectional PATCH-wise protocol: per 40x40 patch keep whichever of the forward / time-reversed
    prediction has the lower MSE, then PSNR / SSIM / LPIPS on that patch. Stricter than a whole-clip direction choice (a single
    wrong-direction region cannot drag the clip) and the protocol the Tedla baseline reports. Every `stride`-th aligned frame."""
    from skimage.metrics import structural_similarity
    H, W = g.shape[1:3]
    rows, cols = sa.patch_starts(H, patch), sa.patch_starts(W, patch)
    ps, ss, ls = [], [], []
    for f in range(0, len(g), stride):
        gp = np.stack([g[f, r:r + patch, c:c + patch] for r in rows for c in cols])
        pf = np.stack([p[f, r:r + patch, c:c + patch] for r in rows for c in cols])
        prv = np.stack([pr[f, r:r + patch, c:c + patch] for r in rows for c in cols])
        mf = ((gp - pf) ** 2).reshape(len(gp), -1).mean(1)
        mr = ((gp - prv) ** 2).reshape(len(gp), -1).mean(1)
        use_r = mr < mf
        pp = np.where(use_r[:, None, None, None], prv, pf)
        mse = np.minimum(mf, mr)
        ps.append(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-10))))
        ss.append(np.mean([structural_similarity(a, b, data_range=1.0, channel_axis=2) for a, b in zip(gp, pp)]))
        with torch.no_grad():
            a_ = torch.from_numpy(gp).permute(0, 3, 1, 2).to(DEV) * 2 - 1
            b_ = torch.from_numpy(pp).permute(0, 3, 1, 2).to(DEV) * 2 - 1
            ls.append(float(lp(a_, b_).mean().item()))
    return float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls))


def score_clip(kind, g9, native, blur, flow, lp):
    gi, pi, rgi = align(kind, len(native))
    p = native[pi]
    g, gr = g9[gi], g9[rgi]
    r = {"n_native": int(len(native))}
    r["psnr"], r["psnr_rev"] = sa.mean_psnr(g, p), sa.mean_psnr(gr, p)
    r["lpips"], r["lpips_rev"] = lpips_mean(g, p, lp), lpips_mean(gr, p, lp)
    cache = {}

    def gtf(a, b):
        if (a, b) not in cache:
            cache[(a, b)] = flow(g9[a], g9[b])
        return cache[(a, b)]
    f, rv = [], []
    for k in range(len(pi) - 1):
        pf = flow(p[k], p[k + 1])
        f.append(sa.epe(pf, gtf(gi[k], gi[k + 1])))
        rv.append(sa.epe(pf, gtf(rgi[k], rgi[k + 1])))
    r["epe"], r["epe_rev"] = float(np.mean(f)), float(np.mean(rv))
    r["reblur"] = float(np.abs(native.mean(0) - blur).mean() * 255)
    # Blur2Vid protocol: pair each predicted frame with GT in the forward ordering and with the reverse-ordered GT counterpart.
    # (p vs g forward; the reversed prediction is p matched to gr, i.e. the same frames read backwards in time.)
    r["pp_psnr"], r["pp_ssim"], r["pp_lpips"] = patch_minpatch(g, p, p[::-1] if kind != "i2v" else p, lp) if kind != "i2v" else (float("nan"),) * 3
    for m, hi in (("psnr", True), ("lpips", False), ("epe", False)):
        a, b = r[m], r[m + "_rev"]
        r[m + "_best"] = max(a, b) if hi else min(a, b)
    return r


# ---------------------------------------------------------------- statistics
def boot_mean_ci(vals, scenes, n_boot=4000, seed=0):
    v = np.asarray(vals, float)
    sc = np.asarray(scenes)
    groups = [v[sc == u] for u in sorted(set(sc.tolist()))]
    rng = np.random.default_rng(seed)
    b = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n_boot)]
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def frechet(fa, fb):
    import scipy.linalg
    mu1, mu2 = fa.mean(0), fb.mean(0)
    s1, s2 = np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    s, _ = scipy.linalg.sqrtm(s1.dot(s2), disp=False)
    return float(np.real(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * s)))


def fvd_ci(real, fake, scenes, n_boot=100, seed=0):
    sc = np.asarray(scenes)
    uniq = sorted(set(sc.tolist()))
    pt = frechet(real, fake)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([np.where(sc == uniq[j])[0] for j in rng.integers(0, len(uniq), len(uniq))])
        vals.append(frechet(real[idx], fake[idx]))
    return pt, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


METRICS = [("pp_psnr", "PSNRp", 2), ("pp_ssim", "SSIMp", 3), ("pp_lpips", "LPIPSp", 3),
           ("psnr", "PSNR", 2), ("psnr_best", "PSNR*", 2), ("lpips", "LPIPS", 3), ("lpips_best", "LPIPS*", 3),
           ("epe", "EPE", 3), ("epe_best", "EPE*", 3), ("reblur", "reblur", 2)]


def summarize(out, ref):
    lines = []
    for ds, d in out["per_clip"].items():
        methods = [m for m in d if d[m]]
        if not methods:
            continue
        stems = sorted(set.intersection(*[set(d[m]) for m in methods]))
        scenes = [out["scenes"][ds][s] for s in stems]
        lines.append(f"\n### {ds}: {len(stems)} clips common to {len(methods)} methods, {len(set(scenes))} scenes"
                     f"   (* = direction-invariant; mean [95% scene-bootstrap CI])")
        lines += ["| method | " + " | ".join(n for _, n, _ in METRICS) + " | FVD |", "|" + "---|" * (len(METRICS) + 2)]
        for m in methods:
            cells = []
            for k, _, nd in METRICS:
                mu, lo, hi = boot_mean_ci([d[m][s][k] for s in stems], scenes)
                cells.append(f"{mu:.{nd}f} [{lo:.{nd}f},{hi:.{nd}f}]")
            fv = out["fvd"].get(ds, {}).get(m)
            cells.append("-" if not fv else f"{fv[0]:.0f} [{fv[1]:.0f},{fv[2]:.0f}]")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append(f"\nGT9 reblur reference (L1x255, mean(GT9) vs blur): {np.mean([out['gt9_reblur'][ds][s] for s in stems]):.2f}")
        if ref in d and d[ref]:
            lines.append(f"\nPaired deltas vs {ref} (method - ref; clip mean [scene-cluster bootstrap CI], scene-mean t-test p):")
            for m in methods:
                if m == ref:
                    continue
                row = []
                for k, nm, nd in METRICS:
                    r = cluster_stats.cluster_paired(np.array([d[m][s][k] - d[ref][s][k] for s in stems]), scenes, n_boot=5000)
                    row.append(f"{nm} {r['clip_mean']:+.{nd}f} [{r['ci'][0]:+.{nd}f},{r['ci'][1]:+.{nd}f}] p={r['p']:.3f}")
                lines.append(f"- {m}: " + "; ".join(row))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="export root holding <ds>/{blur_inputs,gt9}/ (make_common_testset.py --full)")
    ap.add_argument("--datasets", nargs="+", default=["gopro", "adobe"])
    ap.add_argument("--manifest-prefix", default="manifest_full_")
    ap.add_argument("--method", action="append", default=[], metavar="LABEL=KIND:TEMPLATE")
    ap.add_argument("--ref", default=None, help="reference method label for paired deltas")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-fvd", action="store_true")
    ap.add_argument("--skip-missing", action="store_true", help="skip clips a method lacks (default: error out)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import lpips
    lp = lpips.LPIPS(net="alex").to(DEV).eval()
    flow = sa.Flow("pm1")
    specs = {}
    for s in a.method:
        label, rest = s.split("=", 1)
        parts = rest.split(":", 2) if rest.startswith("span:") else rest.split(":", 1)
        kind, tmpl = (":".join(parts[:2]), parts[2]) if rest.startswith("span:") else (parts[0], parts[1])
        specs[label] = (kind, tmpl)
    out = {"args": vars(a), "per_clip": {}, "scenes": {}, "gt9_reblur": {}, "fvd": {}}
    t0 = time.time()
    for ds in a.datasets:
        man = json.load(open(f"{ROOT}/findings/common_testset/{a.manifest_prefix}{ds}.json"))
        clips = man["clips"][:a.limit] if a.limit else man["clips"]
        if man.get("export_root") and os.path.normpath(man["export_root"]) != os.path.normpath(a.root):
            raise SystemExit(f"--root {a.root} != manifest export_root {man['export_root']} (stale/foreign manifest?)")
        out["scenes"][ds] = {c["stem"]: c["scene"] for c in clips}
        n_sc = len({c["scene"] for c in clips})
        if not a.limit and man.get("clip_stride", 1) == 1:
            assert n_sc == EXPECT_SCENES[ds], f"{ds}: {n_sc} scenes, expected {EXPECT_SCENES[ds]} -- scene ids wrong?"
        out["per_clip"][ds] = {m: {} for m in specs}
        out["gt9_reblur"][ds] = {}
        real_f, fake_f = {}, {m: {} for m in specs}
        if not a.no_fvd:
            from cdfvd import fvd as cdfvd_mod
            # n_real/n_fake='full': the default n_fake=2048 CAPS FeatureStats, after which append() silently drops rows and
            # the last row would be returned again and again.
            o = cdfvd_mod.cdfvd(model="i3d", device=DEV, compute_feats=True, n_real="full", n_fake="full")
            last = {}

            def feat(vid, real):
                st = o.real_stats if real else o.fake_stats
                n0 = st.num_items
                (o.add_real_stats if real else o.add_fake_stats)(np.expand_dims((vid * 255).round().astype(np.uint8), 0))
                assert st.num_items == n0 + 1, "FVD feature row was not recorded (stats capped?)"
                row = st.all_features[-1][0].copy()
                prev = last.get(real)
                assert prev is None or not np.array_equal(prev, row) or np.array_equal(vid, last.get(("v", real))), \
                    "two different clips produced identical I3D features"
                last[real], last[("v", real)] = row, vid
                return row
        for ci, c in enumerate(clips):
            stem = c["stem"]
            g9 = cf.read(f"{a.root}/{ds}/gt9/gt9_{stem}.mp4")
            hw = g9.shape[1:3]
            blur = np.asarray(Image.open(f"{a.root}/{ds}/blur_inputs/{stem}.png").convert("RGB")).astype(np.float32) / 255.0
            if blur.shape[:2] != tuple(hw):
                blur = sa.resize_to(blur[None], hw)[0]
            out["gt9_reblur"][ds][stem] = float(np.abs(g9.mean(0) - blur).mean() * 255)
            for m, (kind, tmpl) in specs.items():
                path = resolve(tmpl, a.root, ds, stem)
                if not path:
                    if a.skip_missing:
                        continue
                    raise FileNotFoundError(f"{m}: no prediction for {ds}/{stem} ({tmpl})")
                native = sa.resize_to(cf.read(path), hw)
                out["per_clip"][ds][m][stem] = score_clip(kind, g9, native, blur, flow, lp)
                if not a.no_fvd and len(native) >= 9:
                    if stem not in real_f:
                        real_f[stem] = feat(g9, True)
                    fake_f[m][stem] = feat(native, False)
            if ci % 25 == 0:
                print(f"{ds} {ci + 1}/{len(clips)} t={time.time() - t0:.0f}s", flush=True)
        if not a.no_fvd:
            methods = [m for m in specs if out["per_clip"][ds][m]]
            common = sorted(set.intersection(*[set(out["per_clip"][ds][m]) for m in methods])) if methods else []
            for m in methods:
                if len(common) < 9 or any(s not in fake_f[m] for s in common):
                    continue   # method emits < 9 frames (no FVD) or the common set is too small
                sc = [out["scenes"][ds][s] for s in common]
                out["fvd"].setdefault(ds, {})[m] = fvd_ci(np.stack([real_f[s] for s in common]),
                                                           np.stack([fake_f[m][s] for s in common]), sc)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    md = summarize(out, a.ref)
    open(os.path.splitext(a.out)[0] + ".md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
