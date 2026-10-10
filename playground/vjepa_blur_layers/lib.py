"""Shared helpers for the V-JEPA blur-layer study (see findings/VJEPA_blur_layer_analysis.md).

Models: frozen V-JEPA2.1 ViT-L/384, the Stage-B LoRA-finetuned one (merged), random-init.
Tracer: graph-keeping hooks (NO detach / inference_mode) that record or replace the three kinds of residual writers:
  'embed' (input of block 0), 'attn{i}', 'mlp{i}'; block-output residual streams; final pre-norm stream.
"""
import os
import sys
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from models.LucidFlux.jepa_helpers import load_vjepa2_model  # noqa: E402
from utils.patched_blur_protocol import sample_patch_geometry  # noqa: E402

HUB = "vjepa2_1_vit_large_384"
FT_CKPT = ("/scratch/gilbreth/wweligam/experiments/Motion/BlurJepaViT_stageB_vjepalora_v2_2026_09_10_14_08_22/"
           "checkpoint-50000/PeftModel_1")
GOPRO = "/scratch/gilbreth/wweligam/data/GOPRO_Large_all"
SIZE, G, NT, NF, WIN = 384, 24, 5, 10, 11   # image size, grid, tubelets, vjepa frames, blur window
NL = 24
COMPS = ["embed"] + [f"{k}{i}" for i in range(NL) for k in ("attn", "mlp")]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ------------------------------------------------------------------ models
def load_frozen(dev="cuda", pretrained=True):
    m = load_vjepa2_model(HUB, dev, torch.float32, pretrained=pretrained)
    return m.eval().requires_grad_(False)


def load_finetuned(dev="cuda", ckpt=FT_CKPT):
    from peft import PeftModel
    base = load_vjepa2_model(HUB, dev, torch.float32, pretrained=True)
    m = PeftModel.from_pretrained(base, ckpt).merge_and_unload()
    return m.eval().requires_grad_(False)


def load_random(dev="cuda"):
    torch.manual_seed(0)
    return load_frozen(dev, pretrained=False)


def ft_subset(frozen, ft, blocks_on):
    """Frozen weights everywhere except `blocks_on`, which take the finetuned weights (= LoRA on only those blocks)."""
    import copy
    m = copy.deepcopy(frozen)
    fsd = ft.state_dict()
    sd = m.state_dict()
    for k in sd:
        if k.startswith("blocks."):
            if int(k.split(".")[1]) in blocks_on:
                sd[k] = fsd[k].clone()
    m.load_state_dict(sd)
    return m.eval().requires_grad_(False)


def lora_delta_norms(frozen, ft):
    """Per-block relative weight change of the merged model, by submodule."""
    fs, gs = frozen.state_dict(), ft.state_dict()
    out = {}
    for k in fs:
        if k.startswith("blocks.") and k.endswith("weight") and fs[k].ndim == 2:
            i = int(k.split(".")[1])
            name = k.split(".", 2)[2].replace(".weight", "")
            out[(i, name)] = float((gs[k] - fs[k]).norm() / fs[k].norm())
    return out


# ------------------------------------------------------------------ tracer
class Tracer:
    def __init__(self, model):
        self.m = model

    @staticmethod
    def _tok(o):
        return o[0] if isinstance(o, tuple) else o

    def run(self, x, record=False, replace=None, grad=False, keep_input_grad=False):
        """x [B,3,NF,H,W] ImageNet-normalised. replace: {comp: tensor [B,N,D]} (swapped in, graph-free constant).
        Returns dict(out=[B,N,D] final normed tokens, resid=[24 x [B,N,D]], comps={name: tensor}, xfinal=pre-norm stream)."""
        replace = replace or {}
        rec = {"comps": {}, "resid": [None] * NL}
        hs = []
        blocks = self.m.blocks

        def pre0(mod, args):
            a = args[0]
            if "embed" in replace:
                a = replace["embed"].to(a.dtype)
            if record:
                rec["comps"]["embed"] = a
            return (a,) + tuple(args[1:])
        hs.append(blocks[0].register_forward_pre_hook(pre0))

        for i, blk in enumerate(blocks):
            def mk(name, tuple_out):
                def h(mod, args, out):
                    t = self._tok(out)
                    if name in replace:
                        t = replace[name].to(t.dtype)
                    if record:
                        rec["comps"][name] = t
                    return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
                return h
            hs.append(blk.attn.register_forward_hook(mk(f"attn{i}", True)))
            hs.append(blk.mlp.register_forward_hook(mk(f"mlp{i}", False)))

            def hb(mod, args, out, i=i):
                if record:
                    rec["resid"][i] = self._tok(out)
            hs.append(blk.register_forward_hook(hb))

        def hn(mod, args):
            if record:
                rec["xfinal"] = args[0]
        hs.append(self.m.norms_block[-1].register_forward_pre_hook(hn))
        try:
            with torch.set_grad_enabled(grad):
                out = self.m(x)
        finally:
            for h in hs:
                h.remove()
        rec["out"] = out
        return rec


# ------------------------------------------------------------------ data
def list_scenes(split):
    r = os.path.join(GOPRO, split)
    return sorted(d for d in os.listdir(r) if os.path.isdir(os.path.join(r, d)))


def _load(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)


def _rs(x):
    return F.interpolate(x, size=(SIZE, SIZE), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)


def to_vjepa(frames01):
    """[T,3,H,W] in [0,1] -> [1,3,T,H,W] normalised."""
    x = (frames01 - MEAN.to(frames01.device)) / STD.to(frames01.device)
    return x.permute(1, 0, 2, 3).unsqueeze(0)


def snap_geom(name, seed=42):
    g = sample_patch_geometry(SIZE, SIZE, name, seed=seed)
    t0, l0 = round(g["top"] / 16), round(g["left"] / 16)
    bh, bw = max(4, round(g["bh"] / 16)), max(4, round(g["bw"] / 16))
    t0, l0 = min(t0, G - bh), min(l0, G - bw)
    return t0, l0, bh, bw   # in cells


def cell_dist(geom):
    """Signed Chebyshev distance (cells) to the patch boundary: +d inside (d>=1), -d outside (d>=1)."""
    t0, l0, bh, bw = geom
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    inside = (ii >= t0) & (ii < t0 + bh) & (jj >= l0) & (jj < l0 + bw)
    din = np.minimum.reduce([ii - t0, t0 + bh - 1 - ii, jj - l0, l0 + bw - 1 - jj]) + 1
    dout = np.maximum.reduce([t0 - ii, ii - (t0 + bh - 1), l0 - jj, jj - (l0 + bw - 1), np.zeros_like(ii)])
    return np.where(inside, din, -dout).astype(np.int16)


def make_sample(split, scene, center, tag, seed=42):
    """Build the conditions for one (scene, center). All tensors [T,3,384,384] in [0,1] on CPU."""
    d = os.path.join(GOPRO, split, scene)
    names = sorted(os.listdir(d))
    fr = torch.stack([_load(os.path.join(d, names[center + k])) for k in range(-(WIN // 2), WIN // 2 + 1)])  # 11 frames
    H, W = fr.shape[-2:]
    rng = np.random.default_rng(zlib.crc32(f'{scene}_{center}_{seed}'.encode()))
    x0 = int(rng.integers(0, W - H + 1))
    fr = fr[..., :, x0:x0 + H]                       # square H x H crop
    blur = _rs(fr.mean(0, keepdim=True))[0]
    frs = _rs(fr)
    sharp = frs[WIN // 2]
    gt_vid = frs[:NF]
    other = frs[-1]                                  # a different sharp frame (seam-only control source)
    geom = snap_geom(f"{scene}_{center}", seed)
    t0, l0, bh, bw = geom
    ys, xs = slice(t0 * 16, (t0 + bh) * 16), slice(l0 * 16, (l0 + bw) * 16)
    patch = sharp.clone(); patch[:, ys, xs] = blur[:, ys, xs]
    seam = sharp.clone(); seam[:, ys, xs] = other[:, ys, xs]
    strength = float((blur[:, ys, xs] - sharp[:, ys, xs]).abs().mean())
    rep = lambda im: im.unsqueeze(0).repeat(NF, 1, 1, 1)
    return dict(gt_vid=gt_vid, rep_sharp=rep(sharp), rep_blur=rep(blur), rep_patch=rep(patch), rep_seam=rep(seam),
                geom=geom, strength=strength, dist=cell_dist(geom), scene=scene, center=center)


def sample_list(split, n, seed=0, margin=8):
    """n (scene, center) pairs, round-robin over scenes, random centers (frames 1100/scene)."""
    rng = np.random.default_rng(seed)
    scenes = list_scenes(split)
    out = []
    for k in range(n):
        s = scenes[k % len(scenes)]
        nf = len(os.listdir(os.path.join(GOPRO, split, s)))
        out.append((s, int(rng.integers(margin, nf - margin))))
    return out


def avg_time(tok):
    """[B,N,D] with N = NT*G*G (t,h,w order) -> [B,G*G,D] mean over the NT tubelets."""
    b, n, d = tok.shape
    return tok.reshape(b, NT, G * G, d).mean(1)
