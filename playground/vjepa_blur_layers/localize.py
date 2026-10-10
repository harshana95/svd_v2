"""Localisation of blur by a blur classifier trained ONLY on real whole-frame blur, + layer identification.

python localize.py --root DIR --model frozen|ft|random [--n 60] [--nreal 40] -> DIR/loc_{model}.npz
Head: linear token head on FINAL time-averaged tokens, trained on real blur (rep_blur=1 vs rep_sharp=0, same cells, train scenes, strength>=0.01).
Video logit V = mean over all 576 cells of the head logit. Never trained on spliced patches (OOD localisation test).
Per-layer heads h_l (same recipe on layer-l tokens) give forward score maps. A 'shuffled' head (permuted labels) is the null.
Per sample (spliced patch 'p', seam-only control 's', both vs the sharp repeated 'r'):
  attr[l,c]   = sum_{t,d} grad_{a_l}V * (a_l[cond] - a_l[sharp])      backprop map of block-l residual stream, l=0..23
  fwd[l,c]    = h_l(a_l[cond]) - h_l(a_l[sharp])                       forward map, l=0..24
  swap_in/out = V drop when component output is replaced by the sharp run's, at in-mask tokens only / at equal-size random
                outside tokens (|dist|>=3); per component (attn_l, mlp_l, l>=1) and per block group.
Real blur ('b'): attr/fwd maps vs pseudo-GT |blur-sharp| pooled per cell, and vs texture.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from lib import *  # noqa
from probes import fit_probe

NC_T = 48


def train_heads(root, model, shuffle=False, thr=0.01):
    t = np.load(f"{root}/glob_train/tok_{model}.npy", mmap_mode="r"); m = np.load(f"{root}/glob_train/meta.npz")
    k = m["strength"] >= thr
    heads = []
    for l in range(25):
        Tb = torch.from_numpy(np.asarray(t[k][:, 0, :, l], np.float32)).reshape(-1, 1024).cuda()
        Ts = torch.from_numpy(np.asarray(t[k][:, 1, :, l], np.float32)).reshape(-1, 1024).cuda()
        y = torch.cat([torch.ones(len(Tb)), torch.zeros(len(Ts))]).cuda()
        if shuffle:
            y = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(7)).cuda()]
        heads.append(fit_probe(torch.cat([Tb, Ts]), y, head="linear", steps=400))
    return heads


def lap_energy(img):          # [3,H,W] -> [24,24] mean |laplacian| per cell
    g = img.mean(0, keepdim=True)[None]
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=g.device).view(1, 1, 3, 3)
    return F.avg_pool2d(F.conv2d(g, k, padding=1).abs(), 16)[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--nreal", type=int, default=40)
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True; torch.manual_seed(0)
    model = {"frozen": load_frozen, "ft": load_finetuned, "random": load_random}[a.model]()
    tr = Tracer(model)
    H = {"real": train_heads(a.root, a.model), "shuf": train_heads(a.root, a.model, shuffle=True)}
    print("heads trained", flush=True)
    rng = np.random.default_rng(a.seed)
    dopatch = a.model != "random"

    def V(rec, hk):
        return H[hk][24](avg_time(rec["out"])[0]).mean()

    def maps(rp, rs, hk):
        """backprop attr [24,576] (needs graph in rp) + forward delta [25,576]."""
        res = [rp["resid"][l] for l in range(NL)]
        Vp = V(rp, hk)
        gr = torch.autograd.grad(Vp, res, retain_graph=True)
        at = torch.stack([(g * (rp["resid"][l].detach() - rs["resid"][l])).sum(-1).view(NT, G * G).sum(0) for l, g in enumerate(gr)])
        fw = []
        for l in range(25):
            zp = avg_time(rp["out"] if l == 24 else rp["resid"][l])[0].detach(); zs = avg_time(rs["out"] if l == 24 else rs["resid"][l])[0]
            fw.append(H[hk][l](zp) - H[hk][l](zs))
        return at.detach().cpu().numpy(), torch.stack(fw).detach().cpu().numpy(), float(Vp)

    out = {k: [] for k in ("dist", "strength", "scene", "lap_patch", "lap_sharp")}
    for cond in ("p", "s"):
        for hk in ("real", "shuf"):
            out[f"attr_{cond}_{hk}"] = []; out[f"fwd_{cond}_{hk}"] = []; out[f"V_{cond}_{hk}"] = []
    out["Vs_real"] = []; out["Vs_shuf"] = []
    for k in ("swap_in", "swap_out", "swap_in_Vin", "swap_out_Vin", "suff_in", "grp_in", "grp_out", "Vp_in", "Vs_in", "Vp", "Vs"):
        out[k] = []
    n = 0
    for (sc, c) in sample_list("test", a.n * 4, seed=a.seed):
        if n >= a.n:
            break
        smp = make_sample("test", sc, c, "L")
        if smp["strength"] < 0.01:
            continue
        dist = smp["dist"].reshape(-1)
        xs = to_vjepa(smp["rep_sharp"].cuda())
        rs = tr.run(xs, record=True)
        out["dist"].append(dist.astype(np.int16)); out["strength"].append(smp["strength"]); out["scene"].append(sc)
        out["lap_patch"].append(lap_energy(smp["rep_patch"][0].cuda()).reshape(-1).cpu().numpy())
        out["lap_sharp"].append(lap_energy(smp["rep_sharp"][0].cuda()).reshape(-1).cpu().numpy())
        for cond, key in (("p", "rep_patch"), ("s", "rep_seam")):
            xp = to_vjepa(smp[key].cuda()).requires_grad_(True)
            rp = tr.run(xp, record=True, grad=True)
            for hk in ("real", "shuf"):
                at, fw, v = maps(rp, rs, hk)
                out[f"attr_{cond}_{hk}"].append(at.astype(np.float32)); out[f"fwd_{cond}_{hk}"].append(fw.astype(np.float32)); out[f"V_{cond}_{hk}"].append(v)
            del rp
        for hk in ("real", "shuf"):
            out[f"Vs_{hk}"].append(float(V(rs, hk)))
        if dopatch:   # region-restricted component swaps (real head only)
            posc = np.where(dist >= 3)[0]; negc = np.where(dist <= -3)[0]
            inm = np.where(dist >= 1)[0]
            ocells = rng.choice(negc, min(len(inm), len(negc)), replace=False)
            tokm = lambda cells: torch.from_numpy(np.concatenate([t * G * G + cells for t in range(NT)])).cuda()
            Min, Mout = tokm(inm), tokm(ocells)
            vin = torch.from_numpy(posc).cuda()
            with torch.no_grad():
                rp0 = tr.run(to_vjepa(smp["rep_patch"].cuda()), record=True)
                Vf = lambda r: float(V(r, "real")); Vi = lambda r: float(H["real"][24](avg_time(r["out"])[0])[vin].mean())
                Vp, Vs = Vf(rp0), Vf(rs); out["Vp"].append(Vp); out["Vs"].append(Vs); out["Vp_in"].append(Vi(rp0)); out["Vs_in"].append(Vi(rs))
                xp = to_vjepa(smp["rep_patch"].cuda())
                def mix(name, M, base, other):
                    t = base["comps"][name].clone(); t[:, M] = other["comps"][name][:, M]; return t
                si, so, siv, sov, su = [], [], [], [], []
                for nm in COMPS[1:]:
                    r1 = tr.run(xp, replace={nm: mix(nm, Min, rp0, rs)}); r2 = tr.run(xp, replace={nm: mix(nm, Mout, rp0, rs)})
                    si.append(Vp - Vf(r1)); so.append(Vp - Vf(r2)); siv.append(Vi(rp0) - Vi(r1)); sov.append(Vi(rp0) - Vi(r2))
                    r3 = tr.run(xs, replace={nm: mix(nm, Min, rs, rp0)}); su.append(Vf(r3) - Vs)
                gi, go = [], []
                for g in range(6):
                    names = [f"{k}{i}" for i in range(4 * g, 4 * g + 4) for k in ("attn", "mlp") if i >= 1]
                    gi.append(Vp - Vf(tr.run(xp, replace={nm: mix(nm, Min, rp0, rs) for nm in names})))
                    go.append(Vp - Vf(tr.run(xp, replace={nm: mix(nm, Mout, rp0, rs) for nm in names})))
                for k, v in (("swap_in", si), ("swap_out", so), ("swap_in_Vin", siv), ("swap_out_Vin", sov), ("suff_in", su), ("grp_in", gi), ("grp_out", go)):
                    out[k].append(v)
            del rp0
        n += 1
        if n % 5 == 0:
            print(a.model, "patch", n, flush=True)
    # ---- real whole-frame blur with pseudo-GT local blur magnitude
    rb = {k: [] for k in ("gt", "tex", "attr_real", "attr_shuf", "fwd_real", "fwd_shuf", "strength")}
    nb = 0
    for (sc, c) in sample_list("test", a.nreal * 4, seed=a.seed + 1):
        if nb >= a.nreal:
            break
        smp = make_sample("test", sc, c, "R")
        if smp["strength"] < 0.0:
            continue
        gt = F.avg_pool2d((smp["rep_blur"][0] - smp["rep_sharp"][0]).abs().mean(0)[None, None], 16)[0, 0].reshape(-1).numpy()
        if float(np.abs(smp["rep_blur"][0] - smp["rep_sharp"][0]).mean()) < 0.01:
            continue
        rs = tr.run(to_vjepa(smp["rep_sharp"].cuda()), record=True)
        rp = tr.run(to_vjepa(smp["rep_blur"].cuda()).requires_grad_(True), record=True, grad=True)
        for hk in ("real", "shuf"):
            at, fw, _ = maps(rp, rs, hk); rb[f"attr_{hk}"].append(at.astype(np.float32)); rb[f"fwd_{hk}"].append(fw.astype(np.float32))
        rb["gt"].append(gt); rb["tex"].append(lap_energy(smp["rep_sharp"][0].cuda()).reshape(-1).cpu().numpy())
        rb["strength"].append(float(np.abs(smp["rep_blur"][0] - smp["rep_sharp"][0]).mean()))
        del rp; nb += 1
        if nb % 5 == 0:
            print(a.model, "real", nb, flush=True)
    arr = {k: np.array(v) for k, v in out.items() if len(v)}
    arr.update({f"rb_{k}": np.array(v) for k, v in rb.items()})
    np.savez(f"{a.root}/loc_{a.model}.npz", **arr)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
