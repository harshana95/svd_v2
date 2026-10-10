"""Metrics for localize.py outputs -> DIR/loc_summary.json. python localize_eval.py --root DIR"""
import argparse
import json

import numpy as np
import torch

from extract import auroc
from probes import matched_auroc


def rank(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


def sp(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def partial_sp(a, b, z):
    ra, rb, rz = rank(a), rank(b), rank(z)
    ra = ra - np.polyval(np.polyfit(rz, ra, 1), rz); rb = rb - np.polyval(np.polyfit(rz, rb, 1), rz)
    return float(np.corrcoef(ra, rb)[0, 1])


def loc_metrics(M, D):
    """M [S,L,576] maps (higher = more blur), D [S,576] signed dist. -> per layer dict."""
    S, L, _ = M.shape; out = []
    cell = np.tile(np.arange(576), (S, 1))
    for l in range(L):
        sel = np.abs(D) >= 3
        s = torch.from_numpy(M[:, l][sel]); y = torch.from_numpy((D[sel] > 0).astype(np.int64)); c = cell[sel]
        per = [auroc(torch.from_numpy(M[i, l][np.abs(D[i]) >= 3]), torch.from_numpy(D[i][np.abs(D[i]) >= 3] > 0)) for i in range(S)]
        pg = np.mean([D[i][np.argmax(M[i, l])] >= 1 for i in range(S)])
        near = (D <= -1) & (D >= -2); far = D <= -3; ins = D >= 3
        sm = torch.from_numpy(np.concatenate([M[i, l][ins[i] | near[i]] for i in range(S)])); ym = torch.from_numpy(np.concatenate([ins[i][ins[i] | near[i]] for i in range(S)]))
        pos = np.abs(M[:, l]); mass = float(np.mean([pos[i][D[i] >= 1].sum() / pos[i].sum() for i in range(S)]))
        out.append(dict(auroc_matched=matched_auroc(s, y, c), auroc_per_sample=float(np.nanmean(per)), pointing=float(pg),
                        auroc_vs_near_outside=auroc(sm, ym.bool()), mass_inside=mass))
    return out


def profile(M, D, l):
    """mean map value by signed-distance bin, relative to its inside mean (leakage profile)."""
    bins = [(-23, -6), (-5, -3), (-2, -1), (1, 2), (3, 5), (6, 23)]
    return {f"{a}..{b}": float(np.mean(M[:, l][(D >= a) & (D <= b)])) for a, b in bins}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    res = {}
    for m in ("frozen", "ft", "random"):
        try:
            Z = np.load(f"{a.root}/loc_{m}.npz")
        except FileNotFoundError:
            continue
        D = Z["dist"].astype(int); r = res[m] = {"n": int(len(D))}
        for cond in ("p", "s"):
            for hk in ("real", "shuf"):
                r[f"attr_{cond}_{hk}"] = loc_metrics(Z[f"attr_{cond}_{hk}"], D)
                r[f"fwd_{cond}_{hk}"] = loc_metrics(Z[f"fwd_{cond}_{hk}"], D)
        r["lap_baseline"] = loc_metrics(-Z["lap_patch"][:, None], D)[0]
        r["lap_sharp_ref_baseline"] = loc_metrics((Z["lap_sharp"] - Z["lap_patch"])[:, None], D)[0]   # needs a sharp reference: upper-ish bound
        r["V"] = dict(Vp=float(Z["V_p_real"].mean()), Vseam=float(Z["V_s_real"].mean()), Vsharp=float(Z["Vs_real"].mean()),
                      Vp_shuf=float(Z["V_p_shuf"].mean()), Vsharp_shuf=float(Z["Vs_shuf"].mean()))
        dv = Z["V_p_real"] - Z["Vs_real"]
        r["completeness_ratio_per_layer"] = [float(np.median(Z["attr_p_real"][:, l].sum(1) / dv)) for l in range(24)]
        r["profile_fwd_p_real"] = {str(l): profile(Z["fwd_p_real"], D, l) for l in (2, 8, 12, 16, 24)}
        r["profile_attr_p_real"] = {str(l): profile(Z["attr_p_real"], D, l) for l in (2, 8, 12, 16, 22)}
        if "swap_in" in Z:
            den = float((Z["Vp"] - Z["Vs"]).mean())
            si, so, su = Z["swap_in"], Z["swap_out"], Z["suff_in"]
            rng = np.random.default_rng(0); S = len(si)
            def ci(x):
                bs = [x[rng.integers(0, S, S)].mean(0) for _ in range(300)]
                return np.percentile(bs, [2.5, 97.5], axis=0)
            spec = si - so
            r["swap"] = dict(den=den, comps=[f"{k}{i}" for i in range(24) for k in ("attn", "mlp")],
                             in_mean=(si.mean(0) / den).tolist(), out_mean=(so.mean(0) / den).tolist(), spec_mean=(spec.mean(0) / den).tolist(),
                             spec_lo=(ci(spec)[0] / den).tolist(), spec_hi=(ci(spec)[1] / den).tolist(), suff_in=(su.mean(0) / den).tolist(),
                             in_Vin=(Z["swap_in_Vin"].mean(0)).tolist(), grp_in=(Z["grp_in"].mean(0) / den).tolist(), grp_out=(Z["grp_out"].mean(0) / den).tolist())
            r["swap"]["V_in_only"] = dict(Vp_in=float(Z["Vp_in"].mean()), Vs_in=float(Z["Vs_in"].mean()))
            # does backprop attribution rank components like the in-mask necessity? (per-component, using block residual maps is not 1:1; use group)
        # real blur: pseudo-GT correlation
        gt, tex = Z["rb_gt"], Z["rb_tex"]; n = len(gt); r["real_blur_n"] = n
        for hk in ("real", "shuf"):
            for kind in ("attr", "fwd"):
                Mx = Z[f"rb_{kind}_{hk}"]
                r[f"rb_{kind}_{hk}_spearman_gt"] = [float(np.mean([sp(Mx[i, l], gt[i]) for i in range(n)])) for l in range(Mx.shape[1])]
                r[f"rb_{kind}_{hk}_partial_gt_given_tex"] = [float(np.mean([partial_sp(Mx[i, l], gt[i], tex[i]) for i in range(n)])) for l in range(Mx.shape[1])]
        r["rb_gt_vs_tex_spearman"] = float(np.mean([sp(gt[i], tex[i]) for i in range(n)]))
    json.dump(res, open(f"{a.root}/loc_summary.json", "w"), indent=1)
    Ls = [0, 2, 4, 6, 8, 12, 16, 20, 23]
    for m, r in res.items():
        print("==", m, r["n"], r["V"], "lap", {k: round(v, 3) for k, v in r["lap_baseline"].items()})
        for key in ("attr_p_real", "fwd_p_real", "attr_s_real", "attr_p_shuf", "fwd_p_shuf"):
            print(f"{key:12} auroc_matched", " ".join(f"{r[key][l]['auroc_matched']:.2f}" for l in Ls), "| point", " ".join(f"{r[key][l]['pointing']:.2f}" for l in Ls))
        print("rb attr gt-spearman", " ".join(f"{r['rb_attr_real_spearman_gt'][l]:+.2f}" for l in Ls), "shuf", " ".join(f"{r['rb_attr_shuf_spearman_gt'][l]:+.2f}" for l in Ls))


if __name__ == "__main__":
    main()
