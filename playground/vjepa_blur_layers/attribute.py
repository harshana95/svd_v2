"""Analyses C + D: blur classifier on final V-JEPA tokens -> backprop attribution per residual writer, validated by
activation patching (necessity / sufficiency) and top-k / random-k / bottom-k cumulative patching.

python attribute.py --root DIR --model frozen|ft --n 80 [--head linear|mlp]

Classifier: per-cell head on the time-averaged FINAL normed tokens (trained on stored train-split features, |dist|>=3).
Target T(x) = mean head logit over the positive (inside-patch, dist>=3) cells.
Components: embed, attn{i}, mlp{i}  (the residual writers; 49).
  total_c  = sum_tokens grad_{a_c}(T) . (a_c[patch] - a_c[sharp])        (first-order total effect, incl. downstream recompute)
  direct_c = sum_tokens grad_{x_final}(T) . (a_c[patch] - a_c[sharp])    (direct write to the final stream; ~additive)
  nec_c    = T[patch] - T[patch with a_c <- a_c[sharp]]                  (ground truth necessity)
  suff_c   = T[sharp with a_c <- a_c[patch]] - T[sharp]                  (ground truth sufficiency)
"""
import argparse
import json
import os

import numpy as np
import torch

from lib import *  # noqa
from extract import auroc
from probes import load


class Head(torch.nn.Module):
    def __init__(self, mu, sd, kind):
        super().__init__()
        self.register_buffer("mu", mu); self.register_buffer("sd", sd)
        d = mu.shape[-1]
        self.net = torch.nn.Linear(d, 1) if kind == "linear" else torch.nn.Sequential(
            torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Linear(256, 1))

    def forward(self, z):
        return self.net((z - self.mu) / self.sd).squeeze(-1)


def train_head(root, model, kind, steps=600, wd=1e-3):
    m, f = load(root, "train", model)
    ii, ss = np.where((m["slot_kind"] == 0) & m["slot_valid"] & (np.abs(m["slot_dist"]) >= 3) & (m["strength"][:, None] >= 0.01))
    X = torch.from_numpy(np.stack([f[i, s, 24] for i, s in zip(ii, ss)]).astype(np.float32)).cuda()
    y = torch.from_numpy(m["slot_lab"][ii, ss]).float().cuda()
    h = Head(X.mean(0), X.std(0).clamp_min(1e-6), kind).cuda()
    opt = torch.optim.Adam(h.parameters(), lr=0.01 if kind == "linear" else 0.003, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(); torch.nn.functional.binary_cross_entropy_with_logits(h(X), y).backward(); opt.step()
    return h.eval().requires_grad_(False)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--model", default="frozen")
    ap.add_argument("--head", default="linear"); ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_frozen() if a.model == "frozen" else load_finetuned()
    tr = Tracer(model)
    head = train_head(a.root, a.model, a.head)
    out_dir = os.path.join(a.root, f"attr_{a.model}_{a.head}"); os.makedirs(out_dir, exist_ok=True)

    def T(rec, posm):
        z = rec["out"].view(NT, G * G, -1).mean(0)            # [576,D] time-averaged final tokens
        return head(z)[posm].mean(), head(z)

    samples = sample_list("test", a.n * 3, seed=a.seed)
    C = len(COMPS)
    R = {k: [] for k in ("total", "direct", "total_inpos", "gnorm", "dnorm", "xnorm", "nec", "suff", "Tp", "Ts", "strength",
                         "grp_nec", "grp_suff", "complete_ratio", "full_patch_err", "noop_err", "head_auroc")}
    groups = [list(range(g * 4, g * 4 + 4)) for g in range(6)]
    gcomps = [[c for c in COMPS if c != "embed" and int(c.lstrip("attnmlp")) in g] for g in groups]
    kept = []
    n_done = 0
    for (sc, c) in samples:
        if n_done >= a.n:
            break
        smp = make_sample("test", sc, c, "t")
        if smp["strength"] < 0.01:      # skip near-static patches (label noise); strength logged for the rest
            continue
        dist = smp["dist"].reshape(-1)
        posm = torch.from_numpy(dist >= 3).cuda(); negm = torch.from_numpy(dist <= -3).cuda()
        xs, xp = to_vjepa(smp["rep_sharp"].cuda()), to_vjepa(smp["rep_patch"].cuda())
        rs = tr.run(xs, record=True)
        rp = tr.run(xp.clone().requires_grad_(True), record=True, grad=True)  # input must require grad: params are frozen
        Tp, logit_p = T(rp, posm); Ts, _ = T(rs, posm)
        names = COMPS
        cp = [rp["comps"][n] for n in names]
        gr = torch.autograd.grad(Tp, cp + [rp["xfinal"]])
        gx = gr[-1]
        delta = [rp["comps"][n].detach() - rs["comps"][n] for n in names]
        inpos = torch.from_numpy(np.tile(dist >= 1, NT)).cuda()   # tokens inside the patch (t,h,w order)
        tot = np.array([float((g_ * d_).sum()) for g_, d_ in zip(gr[:-1], delta)])
        tin = np.array([float((g_ * d_)[0][inpos].sum()) for g_, d_ in zip(gr[:-1], delta)])
        dire = np.array([float((gx * d_).sum()) for d_ in delta])
        R["total"].append(tot); R["direct"].append(dire); R["total_inpos"].append(tin)
        R["gnorm"].append([float(g_.norm()) for g_ in gr[:-1]]); R["dnorm"].append([float(d_.norm()) for d_ in delta])
        R["xnorm"].append([float(rp["comps"][n].norm()) for n in names])
        actual = float(Tp - Ts)
        R["complete_ratio"].append(dire.sum() / actual if abs(actual) > 1e-6 else np.nan)
        # patching ground truth
        cs, cpd = {n: rs["comps"][n] for n in names}, {n: rp["comps"][n].detach() for n in names}
        with torch.no_grad():
            nec, suff = [], []
            for n in names:
                nec.append(float(Tp - T(tr.run(xp, replace={n: cs[n]}), posm)[0]))
                suff.append(float(T(tr.run(xs, replace={n: cpd[n]}), posm)[0] - Ts))
            R["nec"].append(nec); R["suff"].append(suff)
            gn, gsu = [], []
            for gc in gcomps:
                gn.append(float(Tp - T(tr.run(xp, replace={n: cs[n] for n in gc}), posm)[0]))
                gsu.append(float(T(tr.run(xs, replace={n: cpd[n] for n in gc}), posm)[0] - Ts))
            R["grp_nec"].append(gn); R["grp_suff"].append(gsu)
            R["full_patch_err"].append(float((T(tr.run(xp, replace=dict(cs)), posm)[0] - Ts).abs()))
            R["noop_err"].append(float((T(tr.run(xp, replace={}), posm)[0] - Tp).abs()))
        sel = posm | negm
        R["head_auroc"].append(auroc(logit_p[sel].detach(), posm[sel]))
        R["Tp"].append(float(Tp)); R["Ts"].append(float(Ts)); R["strength"].append(smp["strength"])
        kept.append((sc, c)); n_done += 1
        if n_done % 5 == 0:
            print(a.model, n_done, "Tp-Ts", round(actual, 3), "spearman(total,nec)",
                  round(spearman(np.mean(R["total"], 0), np.mean(R["nec"], 0)), 3), flush=True)
        del rs, rp, gr
    A = {k: np.array(v) for k, v in R.items()}
    np.savez(os.path.join(out_dir, "attr.npz"), **A, comps=np.array(COMPS), samples=np.array([f"{s}_{c}" for s, c in kept]))

    # --- cumulative top-k / random-k / bottom-k patching, ranking chosen on first half, evaluated on the second half
    scn = sorted({s for s, _ in kept}); rk = np.array([s in scn[::2] for s, _ in kept])   # scene-disjoint rank/eval split
    order = np.argsort(-A["total"][rk].mean(0))
    rng = np.random.default_rng(0)
    cum = {"k": [], "top": [], "rand": [], "bottom": []}
    ks = [1, 2, 4, 8, 16, 24]
    for k in ks:
        top, rnd, bot = [], [], []
        for (sc, c) in [kept[i] for i in np.where(~rk)[0]]:
            smp = make_sample("test", sc, c, "t")
            dist = smp["dist"].reshape(-1); posm = torch.from_numpy(dist >= 3).cuda()
            xs, xp = to_vjepa(smp["rep_sharp"].cuda()), to_vjepa(smp["rep_patch"].cuda())
            with torch.no_grad():
                rs = tr.run(xs, record=True); Tp = T(tr.run(xp), posm)[0]
                def drop(idx):
                    return float(Tp - T(tr.run(xp, replace={COMPS[j]: rs["comps"][COMPS[j]] for j in idx}), posm)[0])
                top.append(drop(order[:k])); bot.append(drop(order[::-1][:k]))
                rnd.append(np.mean([drop(rng.choice(C, k, replace=False)) for _ in range(3)]))
        cum["k"].append(k); cum["top"].append(float(np.mean(top))); cum["rand"].append(float(np.mean(rnd))); cum["bottom"].append(float(np.mean(bot)))
        print("cum", k, cum["top"][-1], cum["rand"][-1], cum["bottom"][-1], flush=True)
    summ = dict(
        n=n_done, mean_Tp_minus_Ts=float(np.mean(A["Tp"] - A["Ts"])),
        head_auroc=float(np.nanmean(A["head_auroc"])),
        spearman_total_vs_nec=spearman(A["total"].mean(0), A["nec"].mean(0)),
        spearman_direct_vs_nec=spearman(A["direct"].mean(0), A["nec"].mean(0)),
        spearman_total_vs_suff=spearman(A["total"].mean(0), A["suff"].mean(0)),
        splithalf_total=spearman(A["total"][rk].mean(0), A["total"][~rk].mean(0)),
        splithalf_nec=spearman(A["nec"][rk].mean(0), A["nec"][~rk].mean(0)),
        complete_ratio_median=float(np.nanmedian(A["complete_ratio"])),
        full_patch_err_max=float(A["full_patch_err"].max()), noop_err_max=float(A["noop_err"].max()),
        cumulative=cum)
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
