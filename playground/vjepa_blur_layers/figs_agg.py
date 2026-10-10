"""Aggregate figures from the saved results (CPU only). python figs_agg.py ROOT OUT"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R, OUT = sys.argv[1], sys.argv[2]
import os
os.makedirs(OUT, exist_ok=True)
C = {"frozen": "#0072B2", "ft": "#D55E00", "random": "#7a7f87"}
NAME = {"frozen": "frozen V-JEPA", "ft": "finetuned (Stage-B LoRA)", "random": "random-init"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.6, "lines.linewidth": 1.8, "figure.dpi": 140,
                     "axes.facecolor": "white", "figure.facecolor": "white"})
X = np.arange(25)
xt = [0, 4, 8, 12, 16, 20, 24]
meta = np.load(f"{R}/test/meta.npz")


def label_x(ax, final=True):
    ax.set_xticks(xt); ax.set_xticklabels([str(i) if i < 24 else "final" for i in xt]); ax.set_xlabel("V-JEPA block (24 = final normed output)")


# ---- Fig 1: token difference per block, three pairs
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
sty = {0: ("blur-rep vs sharp-rep  (blur only)", "-", 1), 1: ("sharp-rep vs GT video  (static clip / motion)", "--", 1), 2: ("blur-rep vs GT video  (requested)", "-", 2.6)}
for ax, k in zip(axs, ("frozen", "ft")):
    d = np.load(f"{R}/test/diff_{k}.npy")
    for pi, (lab, ls, lw) in sty.items():
        y = np.nanmean(d[:, pi, :, 1], 0)
        ax.plot(X, y, ls, lw=lw, color=C[k], alpha=1 if pi == 2 else 0.75, label=lab)
    ax.set_title(NAME[k]); label_x(ax); ax.legend(frameon=False, fontsize=8, loc="lower center")
axs[0].set_ylabel("standardised relative L2 of tokens"); axs[0].set_ylim(0.4, 0.9)
fig.suptitle("Where do the repeated-blur and GT-video tokens differ? Most of the gap to GT video is the static clip, not blur", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_token_difference.png"); plt.close(fig)

# ---- Fig 2: inside vs outside the spliced patch
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, k in zip(axs, ("frozen", "ft")):
    d = np.load(f"{R}/test/diff_{k}.npy")
    ax.plot(X, np.nanmean(d[:, 3, :, 4], 0), color=C[k], label="inside patch"); ax.plot(X, np.nanmean(d[:, 3, :, 5], 0), "--", color=C[k], label="outside patch")
    ax.plot(X, np.nanmean(d[:, 4, :, 0 + 1], 0), ":", color="#444", label="sharp-splice seam control (all tokens)")
    ax.set_title(NAME[k]); label_x(ax); ax.legend(frameon=False, fontsize=8)
axs[0].set_ylabel("relative L2 vs unspliced sharp")
fig.suptitle("The blur effect stays inside the blurred region; leakage outside grows with depth", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_inside_outside.png"); plt.close(fig)

# ---- Fig 3: probes
P = json.load(open(f"{R}/probes.json"))
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, (key, ttl) in zip(axs, (("auroc_d>=3", "interior blur tokens (position-matched)"), ("auroc_blur_vs_seamonly", "blur vs sharp-splice seam control"),
                                  ("auroc_global_blur_vs_sharp", "transfer to whole-frame blur"))):
    for m in ("frozen", "ft", "random"):
        ax.plot(X, [P[m][str(l)][key] for l in X], color=C[m], label=NAME[m], lw=2.2 if m != "random" else 1.4)
    ax.axhline(0.5, color="#999", lw=0.8); ax.set_title(ttl, fontsize=9); label_x(ax)
    if key == "auroc_d>=3":
        ax.axhline(P["frozen"]["0"]["auroc_position_only"], color="#222", ls=":", lw=1.2); ax.text(24, P["frozen"]["0"]["auroc_position_only"] - 0.035, "position-only probe", ha="right", fontsize=8)
axs[0].set_ylabel("probe AUROC (test scenes)"); axs[0].set_ylim(0.45, 1.01); axs[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("Linear probe for 'blurred token' per block: decodable from block 2, saturates at blocks 6-8", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_probes.png"); plt.close(fig)

# ---- Fig 4: attribution vs necessity per component
fig, axs = plt.subplots(2, 2, figsize=(11, 6))
for j, k in enumerate(("frozen", "ft")):
    a = np.load(f"{R}/attr_{k}_linear/attr.npz")
    for i, (nm, ttl) in enumerate((("total", "backprop attribution"), ("nec", "patching: logit drop when swapped to sharp"))):
        ax = axs[i, j]; m = a[nm].mean(0)
        at = np.array([m[1 + 2 * b] for b in range(24)]); ml = np.array([m[2 + 2 * b] for b in range(24)])
        b = np.arange(24)
        ax.bar(b - 0.2, at, 0.4, color=C[k], label="attention"); ax.bar(b + 0.2, ml, 0.4, color=C[k], alpha=0.45, label="MLP")
        ax.set_ylim(-3, 3.2); ax.set_title(f"{NAME[k]}: {ttl}", fontsize=9)
        ax.text(0.3, 2.6, f"mlp0 = {ml[0]:.1f} (off scale)", fontsize=8)
        ax.set_xlabel("block"); ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("Per-component evidence (clipped to +-3; embed and mlp0 dwarf everything)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_components.png"); plt.close(fig)

# ---- Fig 5: agreement scatter, top-k, suffix, group
fig, axs = plt.subplots(1, 4, figsize=(15, 3.5))
for k in ("frozen", "ft"):
    a = np.load(f"{R}/attr_{k}_linear/attr.npz"); keep = np.arange(1, 49)
    axs[0].scatter(a["total"].mean(0)[keep], a["nec"].mean(0)[keep], s=14, color=C[k], alpha=0.8, label=NAME[k])
    f = json.load(open(f"{R}/followup_patch_{k}.json"))
    ks = sorted(map(int, f["topk"]))
    axs[1].plot(ks, [f["topk"][str(q)] for q in ks], "-o", ms=4, color=C[k], label=f"{NAME[k]}: top-k")
    axs[1].plot(ks, [f["randk"][str(q)] for q in ks], "--", color=C[k], alpha=0.7, label="random-k")
    bs = sorted(map(int, f["suffix_necessity"]))
    axs[2].plot(bs, [f["suffix_necessity"][str(q)] / f["Tp_minus_Ts"] for q in bs], "-o", ms=4, color=C[k], label=NAME[k])
    g = a["grp_nec"].mean(0) / f["Tp_minus_Ts"]
    axs[3].bar(np.arange(6) + (-0.2 if k == "frozen" else 0.2), g, 0.4, color=C[k], label=NAME[k])
axs[0].set_xscale("symlog", linthresh=0.3); axs[0].set_yscale("symlog", linthresh=0.3)
axs[0].set_xlabel("backprop attribution"); axs[0].set_ylabel("patching necessity"); axs[0].set_title("attribution vs patching (excl. embed)", fontsize=9)
axs[1].set_xlabel("k components patched (blocks >= 1)"); axs[1].set_ylabel("logit drop"); axs[1].set_title("held-out top-k vs random-k", fontsize=9); axs[1].legend(frameon=False, fontsize=7)
axs[2].set_xlabel("patch all writers in blocks >= b"); axs[2].set_ylabel("fraction of effect removed"); axs[2].set_title("suffix patching", fontsize=9)
axs[3].set_xticks(range(6)); axs[3].set_xticklabels([f"{4*i}-{4*i+3}" for i in range(6)]); axs[3].set_xlabel("block group patched"); axs[3].set_ylabel("fraction of effect removed"); axs[3].set_title("group necessity", fontsize=9)
axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_patching.png"); plt.close(fig)

# ---- Fig 6: LoRA
mv = json.load(open(f"{R}/followup_movement.json"))
fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
for nm, ls in (("attn.qkv", "-"), ("attn.proj", "--"), ("mlp.fc1", "-."), ("mlp.fc2", ":")):
    axs[0].plot(range(24), [mv["lora_delta"][f"{i}.{nm}"] for i in range(24)], ls, color=C["ft"], label=nm)
axs[0].set_xlabel("block"); axs[0].set_ylabel("relative weight change |dW|/|W|"); axs[0].set_title("LoRA weight change (merged)", fontsize=9); axs[0].legend(frameon=False, fontsize=8)
r = mv["rl2_to_frozen_gt_tokens"]; g = np.arange(6)
axs[1].bar(g - 0.2, [r[f"only_g{i}"] for i in g], 0.4, color=C["ft"], label="LoRA only in this group")
axs[1].bar(g + 0.2, [r[f"except_g{i}"] for i in g], 0.4, color=C["ft"], alpha=0.4, label="LoRA everywhere except this group")
axs[1].axhline(r["frozen"], color=C["frozen"], lw=1.4); axs[1].text(5.4, r["frozen"] + 0.005, "frozen", color=C["frozen"], ha="right", fontsize=8)
axs[1].axhline(r["ft_full"], color="#222", lw=1.2, ls=":"); axs[1].text(5.4, r["ft_full"] - 0.02, "full LoRA", ha="right", fontsize=8)
axs[1].set_xticks(g); axs[1].set_xticklabels([f"{4*i}-{4*i+3}" for i in g]); axs[1].set_ylim(0.38, 0.58); axs[1].set_ylabel("rel-L2 of final tokens to frozen GT-video tokens")
axs[1].set_title("which LoRA blocks matter for the Stage-B objective", fontsize=9); axs[1].legend(frameon=False, fontsize=7, loc="upper center")
mvv = np.load(f"{R}/test/move.npy")
for i, (lab, ls) in ((0, ("on blurred input", "-")), (1, ("on sharp input", "--"))):
    axs[2].plot(X, np.nanmean(mvv[:, i, :, 1], 0), ls, color=C["ft"], label=lab)
axs[2].set_xlabel("block"); axs[2].set_ylabel("rel-L2 finetuned vs frozen tokens"); axs[2].set_title("how far finetuning moves each block", fontsize=9); axs[2].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_lora.png"); plt.close(fig)
print("saved to", OUT)
