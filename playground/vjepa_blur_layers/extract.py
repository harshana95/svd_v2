"""Per-block token differences (analysis A/E) + token feature dump for per-block probes (analysis B).

python extract.py --split train --n 400 --out DIR     (run once per split; scene-disjoint GoPro train/test)
Writes DIR/{split}/: meta.npz, diff_{frozen,ft}.npy [N,5,25,8], move.npy [N,6,25,8], feats_{frozen,ft,random}.npy
(layer axis: 0..23 = residual stream after block i, 24 = final normed output).
"""
import argparse
import os
import time

import numpy as np
import torch

from lib import *  # noqa

FIELDS = ["cd_all", "rl2_all", "cd_pos", "cd_neg", "rl2_pos", "rl2_neg", "auroc_cd", "auroc_rl2"]
PAIRS = [("rep_blur", "rep_sharp"), ("rep_sharp", "gt_vid"), ("rep_blur", "gt_vid"),
         ("rep_patch", "rep_sharp"), ("rep_seam", "rep_sharp")]
MOVE = [("rep_blur", "rep_blur"), ("rep_sharp", "rep_sharp"), ("rep_patch", "rep_patch"), ("gt_vid", "gt_vid"),
        ("rep_blur", "gt_vid"), ("__fr_blur", "gt_vid")]   # (ft cond, frozen cond); last = frozen-vs-frozen baseline
NPOS, NNEG, NSEAM, NGB = 64, 64, 32, 32


def auroc(score, label):
    """score [n], label bool [n] -> P(score_pos > score_neg) via ranks (ties ~ average-ish, fine for floats)."""
    n1 = int(label.sum()); n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = torch.empty(len(score), device=score.device)
    r[torch.argsort(score)] = torch.arange(1, len(score) + 1, device=score.device, dtype=score.dtype)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def layers_of(rec):
    return rec["resid"] + [rec["out"]]   # 25 x [1,N,D]


def metrics(za, zb, pos, neg):
    """za, zb [N,D] (N = NT*576). pos/neg: bool [576]."""
    cd = 1 - torch.nn.functional.cosine_similarity(za, zb, dim=-1)
    rl = (za - zb).norm(dim=-1) / zb.norm(dim=-1).clamp_min(1e-6)
    cd_c, rl_c = cd.view(NT, -1).mean(0), rl.view(NT, -1).mean(0)
    sel = pos | neg
    lab = pos[sel]
    return [cd.mean(), rl.mean(), cd_c[pos].mean(), cd_c[neg].mean(), rl_c[pos].mean(), rl_c[neg].mean(),
            auroc(cd_c[sel], lab), auroc(rl_c[sel], lab)]


@torch.no_grad()
def calib(model, tr, samples, n=24):
    s1 = s2 = cnt = 0
    for (sc, c) in samples[:n]:
        smp = make_sample(args.split, sc, c, "c")
        L = layers_of(tr.run(to_vjepa(smp["rep_sharp"].cuda()), record=True))
        a = torch.stack([l[0] for l in L]).double()   # [25,N,D] fp64: avoids E[x^2]-mu^2 cancellation
        s1 = s1 + a.sum(1); s2 = s2 + (a ** 2).sum(1); cnt += a.shape[1]
    mu = s1 / cnt
    sd = (s2 / cnt - mu ** 2).clamp_min(1e-8).sqrt()
    return mu.float(), sd.float()                       # [25,D]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_random", action="store_true")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out = os.path.join(args.out, args.split); os.makedirs(out, exist_ok=True)
    samples = sample_list(args.split, args.n, seed=args.seed)
    models = {"frozen": load_frozen(), "ft": load_finetuned()}
    if not args.no_random:
        models["random"] = load_random()
    trs = {k: Tracer(m) for k, m in models.items()}
    stats = {k: calib(models[k], trs[k], samples) for k in models}
    torch.save(stats, os.path.join(out, "stats.pt"))
    print("calibrated", flush=True)

    N = len(samples)
    T = NPOS + NNEG + NSEAM + 2 * NGB
    feats = {k: np.lib.format.open_memmap(os.path.join(out, f"feats_{k}.npy"), mode="w+", dtype=np.float16,
                                          shape=(N, T, 25, 1024)) for k in models}
    diff = {k: np.full((N, len(PAIRS), 25, len(FIELDS)), np.nan, np.float32) for k in ("frozen", "ft")}
    move = np.full((N, len(MOVE), 25, len(FIELDS)), np.nan, np.float32)
    slot_lab = np.zeros((N, T), np.int8); slot_dist = np.zeros((N, T), np.int16); slot_cell = np.zeros((N, T), np.int16)
    slot_valid = np.zeros((N, T), bool); slot_kind = np.zeros((N, T), np.int8)  # 0 patch,1 seam,2 gblur,3 gsharp
    meta = dict(scene=[], center=[], strength=[], geom=[])
    rng = np.random.default_rng(args.seed + 1)
    t_start = time.time()
    for i, (sc, c) in enumerate(samples):
        smp = make_sample(args.split, sc, c, "s")
        meta["scene"].append(sc); meta["center"].append(c); meta["strength"].append(smp["strength"]); meta["geom"].append(smp["geom"])
        dist = smp["dist"].reshape(-1)
        pos_all, neg_all = np.where(dist >= 1)[0], np.where(dist <= -1)[0]
        pos = torch.from_numpy(dist >= 3).cuda(); neg = torch.from_numpy(dist <= -3).cuda()
        # slots
        cells = []
        def pick(pool, k, lab, kind):
            sel = rng.choice(pool, size=min(k, len(pool)), replace=False)
            cells.append((sel, lab, kind, k))
        pick(pos_all, NPOS, 1, 0); pick(neg_all, NNEG, 0, 0); pick(pos_all, NSEAM, 0, 1)
        pick(np.arange(G * G), NGB, 1, 2); pick(np.arange(G * G), NGB, 0, 3)
        off = 0
        for sel, lab, kind, k in cells:
            n = len(sel)
            slot_lab[i, off:off + n] = lab; slot_cell[i, off:off + n] = sel; slot_dist[i, off:off + n] = dist[sel]
            slot_valid[i, off:off + n] = True; slot_kind[i, off:off + n] = kind
            off += k

        x = {k: to_vjepa(v.cuda()) for k, v in smp.items() if k in ("gt_vid", "rep_sharp", "rep_blur", "rep_patch", "rep_seam")}
        recs = {}
        with torch.no_grad():
            for mk in models:
                conds = ["rep_patch", "rep_seam", "rep_blur", "rep_sharp"] if mk == "random" else list(x)
                recs[mk] = {cn: torch.stack([l[0] for l in layers_of(trs[mk].run(x[cn], record=True))]) for cn in conds}  # [25,N,D]
            for mk in ("frozen", "ft"):
                mu, sd = stats[mk]
                for pi, (a, b) in enumerate(PAIRS):
                    za = (recs[mk][a] - mu[:, None]) / sd[:, None]; zb = (recs[mk][b] - mu[:, None]) / sd[:, None]
                    for l in range(25):
                        diff[mk][i, pi, l] = [float(v) for v in metrics(za[l], zb[l], pos, neg)]
            mu, sd = stats["frozen"]
            for mi, (a, b) in enumerate(MOVE):
                ra = recs["frozen"]["rep_blur"] if a == "__fr_blur" else recs["ft"][a]
                za = (ra - mu[:, None]) / sd[:, None]; zb = (recs["frozen"][b] - mu[:, None]) / sd[:, None]
                for l in range(25):
                    move[i, mi, l] = [float(v) for v in metrics(za[l], zb[l], pos, neg)]
            # probe features (time-averaged cell tokens)
            for mk in models:
                R = recs[mk]
                def cellfeat(cn, sel):  # [25,N,D] -> [len(sel),25,D]
                    t = R[cn].view(25, NT, G * G, 1024).mean(1)
                    return t[:, sel].permute(1, 0, 2)
                parts = [cellfeat("rep_patch", torch.from_numpy(cells[0][0]).cuda()), cellfeat("rep_patch", torch.from_numpy(cells[1][0]).cuda()),
                         cellfeat("rep_seam", torch.from_numpy(cells[2][0]).cuda()),
                         cellfeat("rep_blur", torch.from_numpy(cells[3][0]).cuda()), cellfeat("rep_sharp", torch.from_numpy(cells[4][0]).cuda())]
                buf = torch.zeros(T, 25, 1024, device="cuda", dtype=torch.float16)
                off = 0
                for p, (_, _, _, k) in zip(parts, cells):
                    buf[off:off + p.shape[0]] = p.half(); off += k
                feats[mk][i] = buf.cpu().numpy()
        del recs
        if i % 10 == 0:
            print(f"{i}/{N} {time.time() - t_start:.0f}s", flush=True)
    for mk in feats:
        feats[mk].flush()
    for mk in diff:
        np.save(os.path.join(out, f"diff_{mk}.npy"), diff[mk])
    np.save(os.path.join(out, "move.npy"), move)
    np.savez(os.path.join(out, "meta.npz"), scene=np.array(meta["scene"]), center=np.array(meta["center"]),
             strength=np.array(meta["strength"]), geom=np.array(meta["geom"]), slot_lab=slot_lab, slot_dist=slot_dist,
             slot_cell=slot_cell, slot_valid=slot_valid, slot_kind=slot_kind, fields=np.array(FIELDS),
             pairs=np.array([f"{a}|{b}" for a, b in PAIRS]), moves=np.array([f"ft:{a}|fr:{b}" for a, b in MOVE]))
    print("done", flush=True)
