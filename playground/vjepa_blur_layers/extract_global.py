"""Real (whole-frame) blur features for video-level and token-level per-block classifiers.

python extract_global.py --split train|test --n N --out DIR  -> DIR/glob_{split}/
 vid_{model}.npy [N,3,2,25,1024] fp16: conditions (rep_blur, rep_sharp, gt_vid) x pooling (mean, max over time-avg cell tokens) x layer
 tok_{model}.npy [N,2,NC,25,1024] fp16: NC random cells, time-averaged tokens of (rep_blur, rep_sharp) at the SAME cells
 meta.npz: scene, center, strength (mean|blur-sharp| over the whole frame), cells
Layer axis: 0..23 residual stream after block i, 24 final normed output. Labels are real GoPro blur (mean of 11 frames), no splicing.
"""
import argparse
import os

import numpy as np
import torch

from lib import *  # noqa

NC = 48

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    out = os.path.join(args.out, f"glob_{args.split}"); os.makedirs(out, exist_ok=True)
    samples = sample_list(args.split, args.n, seed=args.seed)
    models = {"frozen": load_frozen(), "ft": load_finetuned(), "random": load_random()}
    trs = {k: Tracer(m) for k, m in models.items()}
    N = len(samples)
    vid = {k: np.lib.format.open_memmap(f"{out}/vid_{k}.npy", mode="w+", dtype=np.float16, shape=(N, 3, 2, 25, 1024)) for k in models}
    tok = {k: np.lib.format.open_memmap(f"{out}/tok_{k}.npy", mode="w+", dtype=np.float16, shape=(N, 2, NC, 25, 1024)) for k in models}
    rng = np.random.default_rng(args.seed + 1)
    meta = dict(scene=[], center=[], strength=[]); cells = np.zeros((N, NC), np.int16)
    for i, (sc, c) in enumerate(samples):
        smp = make_sample(args.split, sc, c, "g")
        meta["scene"].append(sc); meta["center"].append(c)
        meta["strength"].append(float((smp["rep_blur"][0] - smp["rep_sharp"][0]).abs().mean()))
        cells[i] = rng.choice(G * G, NC, replace=False)
        sel = torch.from_numpy(cells[i].astype(np.int64)).cuda()
        x = {k: to_vjepa(smp[k].cuda()) for k in ("rep_blur", "rep_sharp", "gt_vid")}
        with torch.no_grad():
            for mk, tr in trs.items():
                for ci, cn in enumerate(("rep_blur", "rep_sharp", "gt_vid")):
                    rec = tr.run(x[cn], record=True)
                    L = torch.stack([l[0] for l in rec["resid"]] + [rec["out"][0]])        # [25,N,D]
                    t = L.view(25, NT, G * G, 1024).mean(1)                                 # [25,576,D] time-averaged
                    vid[mk][i, ci, 0] = t.mean(1).half().cpu().numpy(); vid[mk][i, ci, 1] = t.max(1).values.half().cpu().numpy()
                    if ci < 2:
                        tok[mk][i, ci] = t[:, sel].permute(1, 0, 2).half().cpu().numpy()
        if i % 20 == 0:
            print(f"{i}/{N}", flush=True)
    for d in (vid, tok):
        for v in d.values():
            v.flush()
    np.savez(f"{out}/meta.npz", scene=np.array(meta["scene"]), center=np.array(meta["center"]), strength=np.array(meta["strength"]), cells=cells)
    print("done", flush=True)
