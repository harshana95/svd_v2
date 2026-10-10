"""Follow-ups to attribute.py (embed / mlp0 trivially dominate: the blur signal ENTERS through them).

--part patch --model M : (a) top/random/bottom-k cumulative patching over blocks>=1 only (ranking by first-half mean total
   attribution, evaluated on held-out samples); (b) prefix patching (sharp values for embed+blocks[0,b)) and suffix patching
   (blocks[b,24) only) -> where is blur-specific writing needed / sufficient.
--part movement : LoRA ablation by block group: final-token closeness to the frozen model's GT-video tokens (the Stage-B
   training target) when LoRA is kept only in group g / removed only from group g, plus per-block LoRA weight-delta norms.
"""
import argparse
import json
import os

import numpy as np
import torch

from lib import *  # noqa
from attribute import train_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--part", required=True)
    ap.add_argument("--model", default="frozen"); ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    if a.part == "movement":
        return movement(a)
    model = load_frozen() if a.model == "frozen" else load_finetuned()
    tr = Tracer(model)
    head = train_head(a.root, a.model, "linear")
    A = np.load(os.path.join(a.root, f"attr_{a.model}_linear", "attr.npz"))
    names = [str(s).rsplit("_", 1)[0] for s in A["samples"]]
    scn = sorted(set(names)); rank_scenes = set(scn[::2])
    rk = np.array([s in rank_scenes for s in names])
    tot = A["total"][rk].mean(0)
    cand = np.arange(3, len(COMPS))                 # drop embed, attn0, mlp0?  keep attn0 out too: block 0 = indices 0..2
    cand = np.array([j for j in range(len(COMPS)) if j >= 3])
    order = cand[np.argsort(-tot[cand])]
    rng = np.random.default_rng(0)

    def T(rec, posm):
        z = rec["out"].view(NT, G * G, -1).mean(0)
        return head(z)[posm].mean()

    res = {"topk": {}, "randk": {}, "bottomk": {}, "prefix": {}, "suffix": {}, "Tp_minus_Ts": []}
    ks = [2, 4, 8, 16, 32]
    bs = [1, 2, 4, 8, 12, 16, 20, 24]
    pre = {b: ["embed"] + [c for c in COMPS if c != "embed" and int(c.lstrip("attnmlp")) < b] for b in bs}
    suf = {b: [c for c in COMPS if c != "embed" and int(c.lstrip("attnmlp")) >= b] for b in bs}
    acc = {k: {m: [] for m in ("topk", "randk", "bottomk")} for k in ks}
    accp = {b: [] for b in bs}; accs = {b: [] for b in bs}; dts = []
    samples = [s for s in sample_list("test", a.n * 8, seed=11) if s[0] not in rank_scenes]   # eval scenes disjoint from ranking scenes
    n = 0
    for (sc, c) in samples:
        if n >= a.n:
            break
        smp = make_sample("test", sc, c, "f")
        if smp["strength"] < 0.01:
            continue
        posm = torch.from_numpy(smp["dist"].reshape(-1) >= 3).cuda()
        xs, xp = to_vjepa(smp["rep_sharp"].cuda()), to_vjepa(smp["rep_patch"].cuda())
        with torch.no_grad():
            rs = tr.run(xs, record=True); Ts = T(rs, posm); Tp = T(tr.run(xp), posm)
            dts.append(float(Tp - Ts))
            drop = lambda names: float(Tp - T(tr.run(xp, replace={m: rs["comps"][m] for m in names}), posm))
            for k in ks:
                acc[k]["topk"].append(drop([COMPS[j] for j in order[:k]]))
                acc[k]["bottomk"].append(drop([COMPS[j] for j in order[::-1][:k]]))
                acc[k]["randk"].append(np.mean([drop([COMPS[j] for j in rng.choice(cand, k, replace=False)]) for _ in range(3)]))
            for b in bs:
                accp[b].append(drop(pre[b])); accs[b].append(drop(suf[b]))
        n += 1
        if n % 8 == 0:
            print(a.model, n, flush=True)
    res["n"] = n; res["Tp_minus_Ts"] = float(np.mean(dts))
    res["topk"] = {k: float(np.mean(acc[k]["topk"])) for k in ks}
    res["randk"] = {k: float(np.mean(acc[k]["randk"])) for k in ks}
    res["bottomk"] = {k: float(np.mean(acc[k]["bottomk"])) for k in ks}
    res["prefix_necessity"] = {b: float(np.mean(accp[b])) for b in bs}     # patch embed + blocks [0,b)
    res["suffix_necessity"] = {b: float(np.mean(accs[b])) for b in bs}     # patch blocks [b,24)
    res["top_order_names"] = [COMPS[j] for j in order[:12]]
    json.dump(res, open(os.path.join(a.root, f"followup_patch_{a.model}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


@torch.no_grad()
def movement(a):
    fr, ft = load_frozen(), load_finetuned()
    out = {"lora_delta": {f"{i}.{k}": v for (i, k), v in lora_delta_norms(fr, ft).items()}}
    groups = [list(range(g * 4, g * 4 + 4)) for g in range(6)]
    variants = {"frozen": fr, "ft_full": ft}
    for g, bl in enumerate(groups):
        variants[f"only_g{g}"] = ft_subset(fr, ft, set(bl))
        variants[f"except_g{g}"] = ft_subset(fr, ft, set(range(NL)) - set(bl))
    trs = {k: Tracer(m) for k, m in variants.items()}
    acc = {k: [] for k in variants}
    samples = sample_list("test", 40, seed=21)
    for (sc, c) in samples:
        smp = make_sample("test", sc, c, "m")
        tgt = trs["frozen"].run(to_vjepa(smp["gt_vid"].cuda()))["out"][0]
        xb = to_vjepa(smp["rep_blur"].cuda())
        for k, t in trs.items():
            o = t.run(xb)["out"][0]
            acc[k].append(float(((o - tgt).norm(dim=-1) / tgt.norm(dim=-1)).mean()))
    out["rl2_to_frozen_gt_tokens"] = {k: float(np.mean(v)) for k, v in acc.items()}
    json.dump(out, open(os.path.join(a.root, "followup_movement.json"), "w"), indent=1)
    print(json.dumps(out["rl2_to_frozen_gt_tokens"], indent=1))


if __name__ == "__main__":
    main()
