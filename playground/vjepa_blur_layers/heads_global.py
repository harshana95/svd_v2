"""Per-block classifiers on REAL whole-frame blur, linear and MLP heads, scene-disjoint train(GoPro train)/test.

python heads_global.py --root DIR -> DIR/heads_global.json
 video-level: pooled (mean|max) block features -> blurry video?  positives = rep_blur; negatives = rep_sharp (same pixels but sharp)
              and gt_vid (real sharp clip with motion). Reports AUROC blur-vs-sharp, blur-vs-gt, by strength tercile, shuffled-label null.
 token-level: time-averaged cell tokens, rep_blur vs rep_sharp at the same cell (position is balanced by construction).
Train filter: global blur strength >= 0.01 (below that the blurred and sharp frames are almost identical).
"""
import argparse
import json

import numpy as np
import torch

from extract import auroc
from probes import fit_probe

THR = 0.01


def ld(root, split, model, kind):
    return np.load(f"{root}/glob_{split}/{kind}_{model}.npy", mmap_mode="r")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True)
    ap.add_argument("--models", nargs="+", default=["frozen", "ft", "random"])
    a = ap.parse_args(); dev = "cuda"
    mt, me = np.load(f"{a.root}/glob_train/meta.npz"), np.load(f"{a.root}/glob_test/meta.npz")
    ktr, kte = mt["strength"] >= THR, me["strength"] >= THR
    q = np.quantile(me["strength"][kte], [1 / 3, 2 / 3])
    ter = {"weak": me["strength"] <= q[0], "mid": (me["strength"] > q[0]) & (me["strength"] <= q[1]), "strong": me["strength"] > q[1]}
    res = {}
    for model in a.models:
        vt, ve = ld(a.root, "train", model, "vid"), ld(a.root, "test", model, "vid")
        tt, te = ld(a.root, "train", model, "tok"), ld(a.root, "test", model, "tok")
        res[model] = {}
        for head in ("linear", "mlp"):
            res[model][head] = {}
            for layer in range(25):
                r = {}
                for pool, pn in ((0, "mean"), (1, "max")):
                    Xb, Xs, Xg = (torch.from_numpy(np.asarray(vt[ktr, c, pool, layer], np.float32)).to(dev) for c in range(3))
                    X = torch.cat([Xb, Xs, Xg]); y = torch.cat([torch.ones(len(Xb)), torch.zeros(len(Xs) + len(Xg))]).to(dev)
                    # class-balance by duplicating positives x2 (negatives are two conditions)
                    X, y = torch.cat([X, Xb]), torch.cat([y, torch.ones(len(Xb), device=dev)])
                    P = fit_probe(X, y, head=head, steps=400)
                    Pn = fit_probe(X, y[torch.randperm(len(y), generator=torch.Generator().manual_seed(1)).to(dev)], head=head, steps=400)
                    Eb, Es, Eg = (torch.from_numpy(np.asarray(ve[kte, c, pool, layer], np.float32)).to(dev) for c in range(3))
                    sb, ss, sg = P(Eb), P(Es), P(Eg)
                    lab = lambda n0, n1: torch.cat([torch.ones(n0), torch.zeros(n1)]).bool().to(dev)
                    r[f"{pn}_auroc_vs_sharp"] = auroc(torch.cat([sb, ss]), lab(len(sb), len(ss)))
                    r[f"{pn}_auroc_vs_gt"] = auroc(torch.cat([sb, sg]), lab(len(sb), len(sg)))
                    r[f"{pn}_auroc_vs_both"] = auroc(torch.cat([sb, ss, sg]), lab(len(sb), len(ss) + len(sg)))
                    nb = Pn(Eb); r[f"{pn}_null"] = auroc(torch.cat([nb, Pn(Es), Pn(Eg)]), lab(len(nb), len(ss) + len(sg)))
                    tk = ter_k = None
                    for tn, m in ter.items():
                        mm = torch.from_numpy(m[kte]).to(dev)
                        r[f"{pn}_{tn}_vs_sharp"] = auroc(torch.cat([sb[mm], ss[mm]]), lab(int(mm.sum()), int(mm.sum())))
                # token-level (position balanced: same cells in both conditions)
                Tb = torch.from_numpy(np.asarray(tt[ktr][:, 0, :, layer], np.float32)).reshape(-1, 1024).to(dev)
                Ts = torch.from_numpy(np.asarray(tt[ktr][:, 1, :, layer], np.float32)).reshape(-1, 1024).to(dev)
                P = fit_probe(torch.cat([Tb, Ts]), torch.cat([torch.ones(len(Tb)), torch.zeros(len(Ts))]).to(dev), head=head, steps=400)
                Eb = torch.from_numpy(np.asarray(te[kte][:, 0, :, layer], np.float32)).reshape(-1, 1024).to(dev)
                Es = torch.from_numpy(np.asarray(te[kte][:, 1, :, layer], np.float32)).reshape(-1, 1024).to(dev)
                sb, ss = P(Eb), P(Es)
                r["tok_auroc"] = auroc(torch.cat([sb, ss]), torch.cat([torch.ones(len(sb)), torch.zeros(len(ss))]).bool().to(dev))
                # paired: within a sample/cell, is blur token scored above sharp token?
                r["tok_paired_acc"] = float((sb > ss).float().mean())
                res[model][head][layer] = r
                print(model, head, layer, {k: round(v, 3) for k, v in r.items() if k in ("mean_auroc_vs_sharp", "mean_auroc_vs_gt", "max_auroc_vs_sharp", "tok_auroc")}, flush=True)
    json.dump(res, open(f"{a.root}/heads_global.json", "w"), indent=1)


if __name__ == "__main__":
    main()
