"""Analysis B: per-block linear probes (token -> inside blur patch), scene-disjoint train(GoPro train)/test(GoPro test).

python probes.py --root DIR [--models frozen ft random] -> DIR/probes.json
Train tokens: patch slots with |dist|>=3 (2-cell seam margin). Test reports: AUROC for |dist|>=3, by distance bin,
shuffled-label null, seam-only control (blur probe on sharp-vs-other-frame splice, label 0 -> AUROC vs patch negatives),
global blur vs sharp (generalisation to un-spliced blur), and by blur-strength tercile.
"""
import argparse
import json
import os

import numpy as np
import torch

from extract import auroc


def coarse(c):
    c = np.asarray(c); return (c // 24) // 6 * 4 + (c % 24) // 6   # 4x4 coarse position bins


def matched_auroc(s, y, c, min_n=10):
    """Position-matched AUROC: mean over coarse position bins that hold >=min_n of each class."""
    b = torch.from_numpy(coarse(c)).to(s.device); out = []
    for k in range(16):
        m = b == k
        if int((m & (y == 1)).sum()) >= min_n and int((m & (y == 0)).sum()) >= min_n:
            out.append(auroc(s[m], y[m].bool()))
    return float(np.mean(out)) if out else float("nan")


def balance_idx(y, c, seed=0):
    """Per coarse bin keep an equal number of positives and negatives."""
    rng = np.random.default_rng(seed); b = coarse(c); keep = []
    for k in range(16):
        p, n = np.where((b == k) & (y == 1))[0], np.where((b == k) & (y == 0))[0]
        m = min(len(p), len(n))
        if m:
            keep += list(rng.choice(p, m, replace=False)) + list(rng.choice(n, m, replace=False))
    return np.array(sorted(keep))


def fit_probe(X, y, steps=300, wd=1e-3, seed=0):
    """Standardised-feature logistic regression (full-batch LBFGS-ish via Adam), fixed reg -> same capacity at every layer."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)
    Xn = (X - mu) / sd
    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True); b = torch.zeros(1, device=X.device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.02)
    yf = y.float()
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xn @ w + b, yf) + wd * (w ** 2).sum()
        loss.backward(); opt.step()
    return lambda Z: ((Z - mu) / sd) @ w.detach() + b.detach()


def load(root, split, model):
    d = os.path.join(root, split)
    m = np.load(os.path.join(d, "meta.npz"))
    f = np.load(os.path.join(d, f"feats_{model}.npy"), mmap_mode="r")
    return m, f


def gather(m, f, layer, kinds, cond):
    """-> X [n,D] float32, plus index arrays (sample id, slot) for the tokens selected by cond(meta slots)."""
    idx_i, idx_s = np.where(np.isin(m["slot_kind"], kinds) & m["slot_valid"] & cond)
    X = np.stack([f[i, s, layer] for i, s in zip(idx_i, idx_s)]) if len(idx_i) else np.zeros((0, f.shape[-1]), np.float16)
    return torch.from_numpy(X.astype(np.float32)), idx_i, idx_s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True)
    ap.add_argument("--models", nargs="+", default=["frozen", "ft", "random"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for model in a.models:
        mt, ft = load(a.root, "train", model); me, fe = load(a.root, "test", model)
        dtr, dte = np.abs(mt["slot_dist"]), np.abs(me["slot_dist"])
        res[model] = {}
        for layer in range(25):
            Xtr, ii, ss = gather(mt, ft, layer, [0], (dtr >= 3) & (mt['strength'][:, None] >= 0.01)); ytr = torch.from_numpy(mt["slot_lab"][ii, ss])
            kp = balance_idx(ytr.numpy(), mt['slot_cell'][ii, ss]); Xtr, ytr = Xtr[kp].to(dev), ytr[kp].to(dev)   # position-balanced training
            P = fit_probe(Xtr, ytr)
            Pn = fit_probe(Xtr, ytr[torch.randperm(len(ytr), generator=torch.Generator().manual_seed(1)).to(dev)])  # shuffled-label null
            r = {}
            Xte, ji, js = gather(me, fe, layer, [0], dte >= 1); Xte = Xte.to(dev)
            yte = torch.from_numpy(me["slot_lab"][ji, js]).to(dev); dd = dte[ji, js]; s = P(Xte)
            cte = me['slot_cell'][ji, js]
            m3 = torch.from_numpy(dd >= 3).to(dev)
            r["auroc_d>=3"] = matched_auroc(s[m3], yte[m3], cte[dd >= 3])          # PRIMARY: position-matched
            r["auroc_d>=3_unmatched"] = auroc(s[m3], yte[m3].bool())
            for d0 in (1, 2):
                msk = torch.from_numpy(dd == d0).to(dev)
                r[f"auroc_d={d0}"] = matched_auroc(s[msk], yte[msk], cte[(dd == d0)])
            r["auroc_null_shuffled"] = auroc(Pn(Xte)[torch.from_numpy(dd >= 3).to(dev)], yte[torch.from_numpy(dd >= 3).to(dev)].bool())
            # strength terciles (per sample)
            st = me["strength"][ji]; q = np.quantile(me["strength"], [1 / 3, 2 / 3])
            for name, msk in (("weak", st <= q[0]), ("mid", (st > q[0]) & (st <= q[1])), ("strong", st > q[1])):
                mk = torch.from_numpy(msk & (dd >= 3)).to(dev)
                r[f"auroc_{name}"] = matched_auroc(s[mk], yte[mk], cte[(msk & (dd >= 3))])
            # seam-only control: sharp-splice tokens (label 0) vs patch positives (label 1) -> should stay high if probe is blur-specific
            Xs, si, ssl = gather(me, fe, layer, [1], dte >= 3); Xs = Xs.to(dev)
            pos_te = torch.from_numpy((dd >= 3)).to(dev) & (yte == 1)
            if len(Xs):
                sc = torch.cat([s[pos_te], P(Xs)]); lb = torch.cat([torch.ones(int(pos_te.sum())), torch.zeros(len(Xs))]).bool().to(dev)
                r["auroc_blur_vs_seamonly"] = auroc(sc, lb)
                negm = torch.from_numpy((dd >= 3)).to(dev) & (yte == 0)
                r["seam_score_minus_negscore"] = float(P(Xs).mean() - s[negm].mean())
            # global blur vs global sharp (un-spliced blur)
            Xg, gi, gs = gather(me, fe, layer, [2, 3], np.ones_like(dte, bool)); Xg = Xg.to(dev)
            yg = torch.from_numpy(me["slot_lab"][gi, gs]).to(dev).bool()
            r["auroc_global_blur_vs_sharp"] = auroc(P(Xg), yg)
            if layer == 0:   # position-only control (row/col polynomial features), independent of the model
                def pf(mm, jj, ss_):
                    c = mm['slot_cell'][jj, ss_]; rr, cc = (c // 24 - 11.5) / 12, (c % 24 - 11.5) / 12
                    return torch.from_numpy(np.stack([rr, cc, rr * rr, cc * cc, rr * cc], 1)).float().to(dev)
                Pp = fit_probe(pf(mt, ii, ss)[torch.from_numpy(kp).to(dev)], ytr)
                r["auroc_position_only_unmatched"] = auroc(Pp(pf(me, ji, js))[m3], yte[m3].bool())
                r["auroc_position_only"] = matched_auroc(Pp(pf(me, ji, js))[m3], yte[m3], cte[dd >= 3])
            res[model][layer] = r
            print(model, layer, {k: round(v, 3) for k, v in r.items() if k.startswith("auroc")}, flush=True)
    json.dump(res, open(os.path.join(a.root, "probes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
