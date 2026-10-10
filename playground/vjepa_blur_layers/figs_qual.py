"""Qualitative token maps. python figs_qual.py ROOT OUT   (GPU, small footprint)"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from lib import *  # noqa
from extract import layers_of
from probes import gather, fit_probe, load

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
models = {"frozen": load_frozen(), "ft": load_finetuned()}
trs = {k: Tracer(m) for k, m in models.items()}
stats = torch.load(f"{R}/train/stats.pt", map_location=dev)
PL = [2, 8, 24]
probes = {}
for k in ("frozen", "ft"):
    mt, ft = load(R, "train", k)
    for l in PL:
        X, ii, ss = gather(mt, ft, l, [0], (np.abs(mt["slot_dist"]) >= 3) & (mt["strength"][:, None] >= 0.01))
        probes[(k, l)] = fit_probe(X.to(dev), torch.from_numpy(mt["slot_lab"][ii, ss]).to(dev))


def feats(k, x):
    with torch.no_grad():
        return torch.stack([l[0] for l in layers_of(trs[k].run(x, record=True))])   # [25,N,D]


def dmap(k, A, B, l):
    mu, sd = stats[k][0][l], stats[k][1][l]
    za, zb = (A[l] - mu) / sd, (B[l] - mu) / sd
    return ((za - zb).norm(dim=-1) / zb.norm(dim=-1)).view(NT, G * G).mean(0).view(G, G).cpu().numpy()


def pmap(k, A, l):
    t = A[l].view(NT, G * G, -1).mean(0)
    return torch.sigmoid(probes[(k, l)](t)).view(G, G).cpu().numpy()


def img(t):
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


# pick samples: 4 with decent blur strength, spread of strengths
cand = []
for sc, c in sample_list("test", 60, seed=5):
    s = make_sample("test", sc, c, "q")
    if s["strength"] >= 0.02:
        cand.append(s)
cand = sorted(cand, key=lambda s: s["strength"])
pick = [cand[int(i)] for i in np.linspace(len(cand) * 0.15, len(cand) - 1, 4)]
print("strengths", [round(p["strength"], 3) for p in pick])

# ---------- Fig 7: spliced patch
cols = 1 + 4 + len(PL) + 1
fig, axs = plt.subplots(4, cols, figsize=(2.1 * cols, 2.2 * 4))
DL = [0, 8, 16, 24]
for r, s in enumerate(pick):
    xs, xp = to_vjepa(s["rep_sharp"].cuda()), to_vjepa(s["rep_patch"].cuda())
    Fs, Fp = feats("frozen", xs), feats("frozen", xp)
    Gs, Gp = feats("ft", xs), feats("ft", xp)
    t0, l0, bh, bw = s["geom"]
    ax = axs[r, 0]; ax.imshow(img(s["rep_patch"][0])); ax.add_patch(Rectangle((l0 * 16, t0 * 16), bw * 16, bh * 16, fill=False, ec="#ffd400", lw=1.5))
    ax.set_title(f"input (blur patch boxed)\nstrength {s['strength']:.3f}", fontsize=7)
    for j, l in enumerate(DL):
        ax = axs[r, 1 + j]; ax.imshow(dmap("frozen", Fp, Fs, l), vmin=0, vmax=1.0, cmap="magma"); ax.set_title(f"token diff vs sharp, block {l if l < 24 else 'final'}", fontsize=7)
    for j, l in enumerate(PL):
        ax = axs[r, 5 + j]; ax.imshow(pmap("frozen", Fp, l), vmin=0, vmax=1, cmap="viridis"); ax.set_title(f"frozen probe P(blur), block {l if l < 24 else 'final'}", fontsize=7)
    ax = axs[r, cols - 1]; ax.imshow(pmap("ft", Gp, 24), vmin=0, vmax=1, cmap="viridis"); ax.set_title("finetuned probe P(blur), final", fontsize=7)
    for a in axs[r]:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
fig.suptitle("Spliced blur patch: token-difference maps (frozen) and probe maps. Difference is local at every depth; probes localise from block 8 on", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig7_patch_maps.png", dpi=130); plt.close(fig)

# ---------- Fig 8: whole-frame blur (the requested repeated-blur vs GT-video comparison)
cols = 8
fig, axs = plt.subplots(4, cols, figsize=(2.1 * cols, 2.2 * 4))
for r, s in enumerate(pick):
    xb, xs_, xg = (to_vjepa(s[k].cuda()) for k in ("rep_blur", "rep_sharp", "gt_vid"))
    Fb, Fs, Fg = (feats("frozen", x) for x in (xb, xs_, xg))
    Gb = feats("ft", xb)
    pix = (s["rep_blur"][0] - s["rep_sharp"][0]).abs().mean(0)
    pix = torch.nn.functional.adaptive_avg_pool2d(pix[None, None], G)[0, 0].numpy()
    panels = [("blurred input (repeated)", None, img(s["rep_blur"][0])), ("|blur - sharp| (pixel)", pix, None),
              ("blur-rep vs GT video, block 4", dmap("frozen", Fb, Fg, 4), None), ("blur-rep vs GT video, block 12", dmap("frozen", Fb, Fg, 12), None),
              ("sharp-rep vs GT video, block 12\n(static clip only)", dmap("frozen", Fs, Fg, 12), None),
              ("blur-rep vs sharp-rep, block 12\n(blur only)", dmap("frozen", Fb, Fs, 12), None),
              ("frozen probe P(blur), block 8", pmap("frozen", Fb, 8), None),
              ("finetuned(blur-rep) vs frozen GT,\nfinal block", dmap("frozen", Gb, Fg, 24), None)]
    for j, (ttl, m, im) in enumerate(panels):
        ax = axs[r, j]
        if im is not None: ax.imshow(im)
        else: ax.imshow(m, cmap="magma" if j != 6 else "viridis", vmin=0, vmax=(1.2 if j in (2, 3, 4, 5, 7) else (1 if j == 6 else None)))
        ax.set_title(ttl, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
fig.suptitle("Whole-frame blur: where do repeated-blur tokens differ from GT-video tokens? Static-clip and blur terms look alike spatially; the finetune pulls the final tokens toward the GT target", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig8_wholeframe_maps.png", dpi=130); plt.close(fig)
print("done")
