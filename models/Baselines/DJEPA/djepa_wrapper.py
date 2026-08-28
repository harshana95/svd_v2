"""Framework-agnostic wrapper around the vendored D-JEPA generator.

D-JEPA (https://arxiv.org/abs/2410.03755) is an ImageNet class-/unconditional
*generator*: its released checkpoints were never trained to take a degraded
image and restore it. To use it as a restoration/deblurring baseline anyway,
this module adds an SDEdit-style "noise-and-denoise" adaptation on top of the
unmodified vendored model (see `sdedit_sample_tokens` below) -- nothing in
`djepa_lib/` is changed to make this work.

Everything here is independent of this repo's `BaseModel`/accelerate
machinery so it can be unit-tested or reused on its own; `DJEPA_model.py`
is the thin adapter that plugs it into the training/eval framework.
"""
import logging
import math
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .djepa_lib.djepa import DJEPA, djepa_base, djepa_huge, djepa_large, mask_by_order
from .djepa_lib.vae import AutoencoderKL

# KL-16 VAE checkpoint pretrained by MAR (https://github.com/LTH14/mar); D-JEPA
# reuses it as-is. Hosted on Dropbox -- there is no HF/hub mirror, so it can't be
# fetched with `huggingface_hub`.
KL16_VAE_URL = (
    "https://www.dropbox.com/scl/fi/hhmuvaiacrarfg28qxhwz/kl16.ckpt"
    "?rlkey=l44xipsezc8atcffdp4q7mwmh&dl=1"
)
KL16_VAE_KWARGS = dict(embed_dim=16, ch_mult=(1, 1, 2, 2, 4))
KL16_SCALING_FACTOR = 0.2325
KL16_SHIFT = 0.0
KL16_VAE_STRIDE = 16

DJEPA_HF_REPO_ID = "densechen/d-jepa-imagenet"
DJEPA_VARIANTS = {"base": djepa_base, "large": djepa_large, "huge": djepa_huge}

# Structural kwargs the released checkpoints were trained with. These are NOT
# free hyperparameters -- changing qk_norm/jepaloss/diffloss would make the
# saved state_dict fail to load (see checkpoints/Baselines/DJEPA*.yml for the
# non-structural knobs that *are* safe to tune at inference time).
_CHECKPOINT_STRUCTURE_KWARGS = dict(
    vae_stride=KL16_VAE_STRIDE,
    patch_size=1,
    vae_embed_dim=KL16_VAE_KWARGS["embed_dim"],
    class_num=1000,
    buffer_size=64,
    diffloss="gauss",
    jepaloss="jepa",
    qk_norm=True,
)


def default_cache_dir() -> str:
    return os.path.expanduser(os.environ.get("DJEPA_CACHE_DIR", "~/.cache/djepa"))


def resolve_kl16_vae_checkpoint(path_hint: str | None, cache_dir: str | None, logger=None) -> str:
    """Return a local path to MAR's kl16.ckpt, downloading it if necessary."""
    logger = logger or logging.getLogger(__name__)
    if path_hint and os.path.exists(path_hint):
        return path_hint

    cache_dir = os.path.expanduser(cache_dir or default_cache_dir())
    os.makedirs(cache_dir, exist_ok=True)
    cached_path = os.path.join(cache_dir, "kl16.ckpt")
    if os.path.exists(cached_path):
        return cached_path

    logger.info(f"KL16-VAE checkpoint not found; downloading to {cached_path} ...")
    try:
        urllib.request.urlretrieve(KL16_VAE_URL, cached_path)
    except Exception as e:
        raise FileNotFoundError(
            "D-JEPA requires the KL16-VAE checkpoint pretrained by MAR, and automatic "
            f"download failed ({e}). Gilbreth compute nodes typically have no internet "
            f"access -- download it on a login node from {KL16_VAE_URL.split('?')[0]} "
            f"and either set `djepa_vae_ckpt_path` in the config or place it at {cached_path}."
        ) from e
    return cached_path


def resolve_djepa_checkpoint(path_hint: str | None, variant: str, logger=None) -> str:
    """Return a local path to the D-JEPA-{variant} checkpoint, downloading from HF hub if needed."""
    logger = logger or logging.getLogger(__name__)
    if path_hint and os.path.exists(path_hint):
        return path_hint

    from huggingface_hub import hf_hub_download

    filename = f"djepa_{variant}.pth"
    logger.info(f"Downloading D-JEPA-{variant} checkpoint from {DJEPA_HF_REPO_ID}/{filename} ...")
    try:
        return hf_hub_download(repo_id=DJEPA_HF_REPO_ID, filename=filename)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not download {filename} from {DJEPA_HF_REPO_ID} ({e}). If the compute "
            "node has no internet access, download it on a login node (it will land in the "
            "shared HF cache, HF_HOME) or set `djepa_checkpoint_path` to a local copy."
        ) from e


def build_djepa(variant: str, img_size: int, num_sampling_steps: str, grad_checkpointing: bool = False) -> DJEPA:
    if variant not in DJEPA_VARIANTS:
        raise ValueError(f"Unknown D-JEPA variant '{variant}'; choose from {list(DJEPA_VARIANTS)}")
    return DJEPA_VARIANTS[variant](
        img_size=img_size,
        num_sampling_steps=num_sampling_steps,
        grad_checkpointing=grad_checkpointing,
        **_CHECKPOINT_STRUCTURE_KWARGS,
    )


def load_djepa_weights(model: DJEPA, checkpoint_path: str, use_ema: bool = True, logger=None) -> None:
    logger = logger or logging.getLogger(__name__)
    state = torch.load(checkpoint_path, map_location="cpu")
    key = "model_ema" if use_ema and "model_ema" in state else "model"
    if use_ema and key != "model_ema":
        logger.warning(f"'{checkpoint_path}' has no 'model_ema' entry; falling back to raw 'model' weights.")
    state_dict = state[key]
    # The released checkpoints were saved from a `torch.compile`-wrapped model
    # (the D-JEPA training scripts pass `--compile`), so every key carries the
    # OptimizedModule's "_orig_mod." prefix. Strip it -- our (uncompiled)
    # DJEPA instance has plain parameter names.
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded D-JEPA weights from {checkpoint_path} (key='{key}').")


@torch.no_grad()
def sdedit_sample_tokens(
    model: DJEPA,
    lq_tokens: torch.Tensor,
    strength: float = 0.6,
    num_iter: int = 64,
    cfg_scale: float = 1.0,
    cfg_schedule: str = "linear",
    labels: torch.Tensor | None = None,
    temperature: float = 1.0,
    time_shifting_factor: float = 1.0,
    progress: bool = False,
) -> torch.Tensor:
    """SDEdit-style restoration sampling for D-JEPA.

    D-JEPA generates images by progressively unmasking latent tokens
    (MaskGIT-style), not by denoising Gaussian noise over the whole image, so
    there's no direct equivalent of "add noise, then denoise from t=strength".
    This is the token-masking analogue: instead of starting `sample_tokens`
    from an all-masked canvas, we start partway through its trajectory with
    the *known* tokens of the degraded input already filled in at the
    positions the schedule would have "decided" by that point. Those
    positions are held fixed; the remaining (still-masked) positions are
    generated by the model as usual, informed by the fixed LQ context through
    attention. `strength` plays the same role as in image diffusion SDEdit:
    ~0 keeps the input almost unchanged, 1.0 discards it and generates from
    scratch (equivalent to `DJEPA.sample_tokens`).

    Args:
        lq_tokens: [B, L, token_embed_dim] patchified, VAE-latent-space,
            scaling-normalized tokens of the degraded input (see
            `DJEPARestorer.encode`).
    """
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"strength must be in (0, 1], got {strength}")

    bsz, seq_len, _ = lq_tokens.shape
    device = lq_tokens.device

    start_step = int(round((1.0 - strength) * num_iter))
    start_step = min(max(start_step, 0), num_iter - 1)

    orders = model.sample_orders(bsz)

    if start_step == 0:
        mask = torch.ones(bsz, seq_len, device=device)
    else:
        mask_ratio = np.cos(math.pi / 2.0 * start_step / num_iter)
        mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).to(device)
        mask = mask_by_order(mask_len[0], orders, bsz, seq_len).float()

    tokens = torch.zeros_like(lq_tokens)
    unmasked = mask == 0
    tokens[unmasked] = lq_tokens[unmasked]

    indices = range(start_step, num_iter)
    if progress:
        from tqdm import tqdm

        indices = tqdm(indices)

    for step in indices:
        cur_tokens = tokens.clone()

        if labels is not None:
            class_embedding = model.class_emb(labels)
        else:
            class_embedding = model.fake_latent.repeat(bsz, 1)

        step_tokens, step_mask = tokens, mask
        if not cfg_scale == 1.0:
            step_tokens = torch.cat([tokens, tokens], dim=0)
            class_embedding = torch.cat([class_embedding, model.fake_latent.repeat(bsz, 1)], dim=0)
            step_mask = torch.cat([mask, mask], dim=0)

        x = model.forward_encoder(step_tokens, step_mask, class_embedding)
        z = model.forward_decoder(x, step_mask)

        mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
        mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).to(device)
        mask_len = torch.maximum(
            torch.Tensor([1]).to(device),
            torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
        )

        mask_next = mask_by_order(mask_len[0], orders, bsz, seq_len)
        if step >= num_iter - 1:
            mask_to_pred = mask.bool()
        else:
            mask_to_pred = torch.logical_xor(mask.bool(), mask_next.bool())
        mask = mask_next
        if not cfg_scale == 1.0:
            mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

        pred_indices = mask_to_pred.nonzero(as_tuple=True)
        z_sel = z[pred_indices]
        if z_sel.shape[0] == 0:
            continue

        if cfg_schedule == "linear":
            cfg_iter = 1 + (cfg_scale - 1) * (seq_len - mask_len[0]) / seq_len
        elif cfg_schedule == "constant":
            cfg_iter = cfg_scale
        else:
            raise NotImplementedError(f"Unknown cfg_schedule '{cfg_schedule}'")

        sampled_token_latent = model.diffloss.sample(
            z_sel, temperature=temperature, cfg_scale=cfg_iter, time_shifting_factor=time_shifting_factor
        )
        if not cfg_scale == 1.0:
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
            mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
            pred_indices = mask_to_pred.nonzero(as_tuple=True)

        cur_tokens[pred_indices] = sampled_token_latent
        tokens = cur_tokens.clone()

    return model.unpatchify(tokens)


class DJEPARestorer(nn.Module):
    """Bundles the D-JEPA generator + KL16 VAE and exposes a `restore(lq)` API.

    All tensors are in this repo's usual [-1, 1] normalized image convention.
    """

    def __init__(
        self,
        variant: str = "base",
        img_size: int = 256,
        checkpoint_path: str | None = None,
        vae_ckpt_path: str | None = None,
        cache_dir: str | None = None,
        use_ema: bool = True,
        num_sampling_steps: str = "100",
        logger=None,
    ):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.img_size = img_size
        self.scaling_factor = KL16_SCALING_FACTOR
        self.shift = KL16_SHIFT

        djepa_ckpt = resolve_djepa_checkpoint(checkpoint_path, variant, self.logger)
        self.model = build_djepa(variant, img_size, num_sampling_steps)
        load_djepa_weights(self.model, djepa_ckpt, use_ema=use_ema, logger=self.logger)

        vae_ckpt = resolve_kl16_vae_checkpoint(vae_ckpt_path, cache_dir, self.logger)
        self.vae = AutoencoderKL(ckpt_path=vae_ckpt, **KL16_VAE_KWARGS)

        self.model.requires_grad_(False)
        self.model.eval()
        self.vae.requires_grad_(False)
        self.vae.eval()

    @torch.no_grad()
    def encode(self, img_pm1: torch.Tensor, mode: str = "mode") -> torch.Tensor:
        posterior = self.vae.encode(img_pm1)
        z = posterior.mode() if mode == "mode" else posterior.sample()
        return z.sub(self.shift).mul(self.scaling_factor)

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(tokens.div(self.scaling_factor).add(self.shift))

    @torch.no_grad()
    def restore(
        self,
        lq_pm1: torch.Tensor,
        strength: float = 0.6,
        num_iter: int = 64,
        cfg_scale: float = 1.0,
        cfg_schedule: str = "linear",
        temperature: float = 1.0,
        time_shifting_factor: float = 1.0,
        class_label: int | None = None,
        vae_encode_mode: str = "mode",
        progress: bool = False,
    ) -> torch.Tensor:
        bsz, _, h, w = lq_pm1.shape
        needs_resize = (h, w) != (self.img_size, self.img_size)
        lq_in = lq_pm1
        if needs_resize:
            lq_in = F.interpolate(lq_pm1, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        x0 = self.encode(lq_in, mode=vae_encode_mode)
        lq_tokens = self.model.patchify(x0)

        labels = None
        if class_label is not None:
            labels = torch.full((bsz,), int(class_label), dtype=torch.long, device=lq_pm1.device)

        tokens = sdedit_sample_tokens(
            self.model,
            lq_tokens,
            strength=strength,
            num_iter=num_iter,
            cfg_scale=cfg_scale,
            cfg_schedule=cfg_schedule,
            labels=labels,
            temperature=temperature,
            time_shifting_factor=time_shifting_factor,
            progress=progress,
        )
        out = self.decode(tokens)
        if needs_resize:
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out.clamp(-1, 1)
