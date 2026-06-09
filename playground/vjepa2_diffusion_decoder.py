"""
V-JEPA 2.1 tokens → Stable Diffusion 3 / Flux.1 decoder
=======================================================
Replaces the text conditioning in SD3 or Flux with V-JEPA 2.1 patch embeddings
by training a small projector MLP. The diffusion backbone stays completely frozen.

Architecture
------------
  Frozen V-JEPA 2.1 encoder (torch.hub ``facebookresearch/vjepa2``)
        ↓  [B, T*N, 1024]
  JEPAProjector (trained)
        ↓  [B, S, joint_attn_dim]   S = sequence slots
  Frozen SD3 MMDiT / Flux DiT
        ↓  (denoises VAE latents conditioned on JEPA tokens)
  Frozen VAE decoder
        ↓  pixel image

Key insight: SD3 and Flux process text tokens and image tokens jointly in the
same self-attention sequence. Replacing the text token stream with projected
JEPA tokens costs nothing extra — only the projector is trained.

Dim targets
-----------
  SD3   MMDiT: joint_attention_dim = 4096,  pooled_projection_dim = 2048
  Flux  DiT:   text stream dim     = 4096,  pooled dim             = 768
  V-JEPA 2.1 ViT-L @384: embed_dim = 1024,  sequence length varies with tubes × patches

Usage
-----

python vjepa2_diffusion_decoder.py train-flux --data_dir /depot/chan129/data/datasets_deconvolution/Flickr2K/train/ --resume ./vjepa2_diffusion_out/projector_flux.pth --epochs 100 --batch_size 8
  
  # Train projector (SD3)
  python jepa_diffusion_decoder.py train-sd3 \
      --data_dir /path/to/images --epochs 10 --batch_size 2

  # Train projector (Flux)
  python jepa_diffusion_decoder.py train-flux \
      --data_dir /path/to/images --epochs 10 --batch_size 2

  # Resume (``--epochs`` is the final epoch number; train until that epoch)
  python jepa_diffusion_decoder.py train-flux \
      --data_dir /path/to/images --epochs 20 --resume jepa_diffusion_out/projector_flux.pth

  # Each training epoch saves ``out_dir/epoch_XXX_decode.png`` (use ``--no_epoch_decode`` to disable)

  # Decode an image
  python jepa_diffusion_decoder.py decode \
      --image photo.jpg --checkpoint projector_sd3.pth --model sd3

Requirements
------------
  pip install torch torchvision diffusers accelerate pillow einops
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.LucidFlux.src.flux.lucidflux import (
    extract_vjepa_tokens,
    load_vjepa2_model,
    vjepa_from_unit_tensor,
)


# ─────────────────────────────────────────────────────────────────────────────
# Conditioning dimensions for each diffusion model
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIMS = {
    # (seq_dim, pooled_dim, max_seq_len)
    "sd3":  (4096, 2048, 77),
    "flux": (4096,  768, 77),
}

# V-JEPA 2.1 (Meta torch.hub; not the HF V-JEPA 2 feature extractor checkpoints)
JEPA_HUB_ENTRY_DEFAULT = "vjepa2_1_vit_large_384"
JEPA_IMAGE_SIZE_DEFAULT = 384
SD3_REPO   = "stabilityai/stable-diffusion-3-medium-diffusers"
FLUX_REPO  = "black-forest-labs/FLUX.1-dev"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Frozen V-JEPA 2.1 Encoder  (torch.hub, float32 — RoPE mixes fp32 grids)
# ─────────────────────────────────────────────────────────────────────────────

class VJEPAEncoder(nn.Module):
    def __init__(
        self,
        device="cpu",
        hub_entry: str = JEPA_HUB_ENTRY_DEFAULT,
        image_size: int = JEPA_IMAGE_SIZE_DEFAULT,
        pretrained: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.jepa_dtype = torch.float32
        print(f"Loading V-JEPA 2.1 encoder ({hub_entry}, {image_size}px) …")
        self.model = load_vjepa2_model(
            hub_entry,
            device,
            self.jepa_dtype,
            pretrained=pretrained,
            trust_repo=True,
        )
        self.embed_dim = int(self.model.embed_dim)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: [B, C, T, H, W] (ImageNet-norm, float32) → [B, T_tubes*N, D]"""
        return extract_vjepa_tokens(
            self.model,
            pixel_values,
            pixel_values.device,
            self.jepa_dtype,
        )

    def preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """Single PIL image → [1, C, 1, H, W] (single-frame 'video')"""
        x = transforms.ToTensor()(img.convert("RGB")).unsqueeze(0)
        x = vjepa_from_unit_tensor(
            x,
            size=self.image_size,
            device=None,
            out_dtype=torch.float32,
        )
        return x.unsqueeze(2)

    def preprocess_frames(self, frames) -> torch.Tensor:
        tensors = torch.stack(
            [transforms.ToTensor()(f.convert("RGB")) for f in frames],
            dim=0,
        ).unsqueeze(0)
        B, T, c, h, w = tensors.shape
        assert B == 1
        out = []
        for t in range(T):
            out.append(
                vjepa_from_unit_tensor(
                    tensors[:, t],
                    size=self.image_size,
                    device=None,
                    out_dtype=torch.float32,
                )
            )
        return torch.stack(out, dim=2)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  JEPA → Diffusion Projector  (the only trained component)
# ─────────────────────────────────────────────────────────────────────────────

class JEPAProjector(nn.Module):
    """
    Maps V-JEPA 2.1 patch embeddings [B, N, D] to the conditioning format
    expected by SD3 or Flux:
      - seq_emb:    [B, max_seq_len, seq_dim]   (replaces text token stream)
      - pooled_emb: [B, pooled_dim]              (replaces pooled text vec)

    Trainable parameters: ~20–40M depending on target dims.
    """

    def __init__(
        self,
        jepa_dim:    int = 1024,
        seq_dim:     int = 4096,
        pooled_dim:  int = 2048,
        max_seq_len: int = 77,
        hidden_dim:  int = 2048,
        num_heads:   int = 8,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        # ── Cross-attention pool: N JEPA tokens → max_seq_len slots ─────────
        # Learned query tokens act as the "text slots"
        self.query_tokens = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )
        self.jepa_proj  = nn.Linear(jepa_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)

        # ── Output projections ───────────────────────────────────────────────
        self.seq_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, seq_dim),
        )

        # Pooled embedding: global average of JEPA tokens → pooled_dim
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(jepa_dim),
            nn.Linear(jepa_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pooled_dim),
        )

    def forward(self, jepa_emb: torch.Tensor):
        """
        Args:
            jepa_emb: [B, N, D]  (T×N patches from V-JEPA 2.1 hub encoder)
        Returns:
            seq_emb:    [B, max_seq_len, seq_dim]
            pooled_emb: [B, pooled_dim]
        """
        B = jepa_emb.size(0)

        # ── Cross-attention: learned queries attend to JEPA patch keys/values
        k = self.norm_k(self.jepa_proj(jepa_emb))          # [B, N, H]
        q = self.norm_q(
            self.query_tokens.expand(B, -1, -1)
        )                                                   # [B, S, H]
        out, _ = self.cross_attn(q, k, k)                  # [B, S, H]
        seq_emb = self.seq_out(out)                         # [B, S, seq_dim]

        # ── Pooled: global mean of raw JEPA embeddings ─────────────────────
        pooled_emb = self.pool_proj(jepa_emb.mean(dim=1))  # [B, pooled_dim]

        return seq_emb, pooled_emb

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters()) / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Frozen Diffusion Pipeline wrappers
# ─────────────────────────────────────────────────────────────────────────────

def load_sd3_pipeline(repo=SD3_REPO, device="cuda"):
    from diffusers import StableDiffusion3Pipeline
    print(f"Loading SD3 …")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        repo,
        torch_dtype=torch.float16,
    ).to(device)
    # Freeze everything — we only train the projector
    for p in pipe.transformer.parameters():  p.requires_grad_(False)
    for p in pipe.vae.parameters():          p.requires_grad_(False)
    return pipe


def load_flux_pipeline(repo=FLUX_REPO, device="cuda"):
    from diffusers import FluxPipeline
    print(f"Loading Flux …")
    pipe = FluxPipeline.from_pretrained(
        repo,
        torch_dtype=torch.bfloat16,
    ).to(device)
    for p in pipe.transformer.parameters(): p.requires_grad_(False)
    for p in pipe.vae.parameters():         p.requires_grad_(False)
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Diffusion training step helpers
# ─────────────────────────────────────────────────────────────────────────────

def sd3_train_step(pipe, projector, jepa_emb, images, noise_scheduler):
    """
    One diffusion training step for SD3 using JEPA conditioning.
    Returns scalar loss.
    """
    vae        = pipe.vae
    transformer= pipe.transformer
    dtype      = torch.float16
    device     = images.device

    # ── VAE encode ──────────────────────────────────────────────────────────
    with torch.no_grad():
        latents = vae.encode(images.to(dtype)).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

    # ── Noise ───────────────────────────────────────────────────────────────
    noise     = torch.randn_like(latents)
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],), device=device
    ).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # ── JEPA conditioning (projector stays float32; transformer expects dtype)
    prompt_embeds, pooled_embeds = projector(jepa_emb.float())
    prompt_embeds = prompt_embeds.to(dtype)
    pooled_embeds = pooled_embeds.to(dtype)

    # ── Transformer forward ─────────────────────────────────────────────────
    noise_pred = transformer(
        hidden_states        = noisy_latents,
        timestep             = timesteps,
        encoder_hidden_states= prompt_embeds,
        pooled_projections   = pooled_embeds,
        return_dict          = False,
    )[0]

    return F.mse_loss(noise_pred, noise)


def flowmatch_noisy_latents(
    noise_scheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    discrete_flow_shift: float = 3.1582,
):
    """
    Build noisy latents for FlowMatch schedulers that do not expose ``add_noise``.
    """
    batch_size = latents.shape[0]
    num_timesteps = int(noise_scheduler.config.num_train_timesteps)
    sigmas = torch.sigmoid(
        torch.randn((batch_size,), device=latents.device, dtype=torch.float32)
        + discrete_flow_shift
    )
    timesteps = (sigmas * num_timesteps).long().clamp_(0, num_timesteps - 1)
    sigmas = sigmas.view(-1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
    return noisy_latents, timesteps


def flux_train_step(
    pipe, projector, jepa_emb, images, noise_scheduler, guidance_scale=3.5
):
    """
    One diffusion training step for Flux using JEPA conditioning.
    Matches ``FluxPipeline`` latent packing, 2-D position ids, and optional
    guidance embedding (FLUX.1-dev).
    """
    from diffusers import FluxPipeline

    vae         = pipe.vae
    transformer = pipe.transformer
    dtype       = torch.bfloat16
    device      = images.device

    with torch.no_grad():
        latents = vae.encode(images.to(dtype)).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

    noise = torch.randn_like(latents)
    noisy_latents, timesteps = flowmatch_noisy_latents(
        noise_scheduler, latents, noise
    )

    prompt_embeds, pooled_embeds = projector(jepa_emb.float())
    prompt_embeds = prompt_embeds.to(dtype)
    pooled_embeds = pooled_embeds.to(dtype)

    B, C, H, W = latents.shape
    packed_in = FluxPipeline._pack_latents(noisy_latents, B, C, H, W)
    target = FluxPipeline._pack_latents(noise - latents, B, C, H, W)

    # 2-D ids (batch dim is *not* part of img/txt ids), same as diffusers FluxPipeline
    img_ids = FluxPipeline._prepare_latent_image_ids(B, H // 2, W // 2, device, dtype)
    txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)

    t_in = timesteps.to(device=device, dtype=dtype).float() / 1000.0
    if getattr(transformer.config, "guidance_embeds", False):
        guidance = torch.full(
            (B,), float(guidance_scale), device=device, dtype=torch.float32
        )
    else:
        guidance = None

    noise_pred = transformer(
        hidden_states        = packed_in,
        timestep             = t_in,
        guidance             = guidance,
        encoder_hidden_states= prompt_embeds,
        pooled_projections   = pooled_embeds,
        img_ids              = img_ids,
        txt_ids              = txt_ids,
        return_dict          = False,
    )[0]

    return F.mse_loss(noise_pred.float(), target.float())


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ImageFolderDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, root: str, size: int = 512):
        self.paths = []
        for d, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in self.EXTS:
                    self.paths.append(os.path.join(d, f))
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.paths.sort()
        print(f"Found {len(self.paths)} images")

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # [-1, 1]  for VAE
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args, model_key: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_dim, pooled_dim, max_seq = MODEL_DIMS[model_key]

    # ── Load models ─────────────────────────────────────────────────────────
    jepa    = VJEPAEncoder(
        device=device,
        hub_entry=args.jepa_hub_entry,
        image_size=args.vjepa_image_size,
        pretrained=not args.jepa_no_pretrained,
    )
    projector = JEPAProjector(
        jepa_dim    = jepa.embed_dim,  # 1024
        seq_dim     = seq_dim,
        pooled_dim  = pooled_dim,
        max_seq_len = max_seq,
        hidden_dim  = args.hidden_dim,
    ).to(device)
    print(f"Projector: {projector.num_params:.1f}M trainable params")

    start_epoch = 1
    resume_ckpt = None
    if args.resume:
        resume_path = os.path.abspath(os.path.expanduser(args.resume))
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if resume_ckpt.get("model_key") != model_key:
            raise ValueError(
                f"Checkpoint model_key={resume_ckpt.get('model_key')!r} does not match "
                f"this command ({model_key!r})."
            )
        saved = resume_ckpt.get("args") or {}
        if saved.get("hidden_dim") != args.hidden_dim:
            raise ValueError(
                f"Checkpoint hidden_dim={saved.get('hidden_dim')} but you passed "
                f"--hidden_dim {args.hidden_dim}; they must match."
            )
        projector.load_state_dict(resume_ckpt["state_dict"])
        start_epoch = int(resume_ckpt["epoch"]) + 1
        print(f"Resume: loaded {resume_path} (completed epoch {start_epoch - 1}), "
              f"continuing from epoch {start_epoch} → target epoch {args.epochs}")

    if model_key == "sd3":
        pipe = load_sd3_pipeline(device=device)
        from diffusers import FlowMatchEulerDiscreteScheduler
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            SD3_REPO, subfolder="scheduler"
        )
    else:
        pipe = load_flux_pipeline(device=device)
        from diffusers import FlowMatchEulerDiscreteScheduler
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            FLUX_REPO, subfolder="scheduler"
        )

    # ── Prep for V-JEPA input (matches ``vjepa_from_unit_tensor``) ─────────
    to_01 = transforms.Normalize([-1], [2])   # [-1,1] → [0,1] for encoder

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset    = ImageFolderDataset(args.data_dir, size=args.image_size)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    steps_per_epoch = len(dataloader)
    total_cosine_steps = max(1, int(args.epochs) * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_cosine_steps, eta_min=1e-6
    )

    if resume_ckpt is not None:
        has_opt_ckpt = resume_ckpt.get("optimizer_state_dict") is not None
        if has_opt_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        else:
            print(
                "Note: checkpoint has no optimizer_state_dict — optimizer resets; "
                "matching cosine schedule by stepping the LR scheduler.",
                flush=True,
            )
        prev_cosine_steps = resume_ckpt.get("cosine_t_max")
        sched_sd = resume_ckpt.get("scheduler_state_dict")
        same_cosine = prev_cosine_steps is not None and int(prev_cosine_steps) == int(
            total_cosine_steps
        )
        # Only restore scheduler snapshot together with optimizer; otherwise rewind steps.
        if sched_sd is not None and same_cosine and has_opt_ckpt:
            scheduler.load_state_dict(sched_sd)
        elif start_epoch > 1:
            if sched_sd is not None and prev_cosine_steps is not None and not same_cosine:
                print(
                    f"Warning: --epochs*{steps_per_epoch} → T_max={total_cosine_steps} "
                    f"differs from saved cosine_t_max={prev_cosine_steps}; "
                    f"scheduler state ignored (fast-forward {(start_epoch - 1) * steps_per_epoch} steps).",
                    flush=True,
                )
            n_done = max(0, (start_epoch - 1) * steps_per_epoch)
            for _ in range(n_done):
                scheduler.step()

    os.makedirs(args.out_dir, exist_ok=True)

    if start_epoch > args.epochs:
        print(
            f"No training: checkpoint already finished epoch {start_epoch - 1} "
            f"and --epochs={args.epochs}. Increase --epochs to continue.",
            flush=True,
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        projector.train()
        total_loss = 0.0

        for step, images in enumerate(dataloader):
            images = images.to(device)            # [B, 3, H, W]  in [-1, 1]

            # ── Get JEPA embeddings ─────────────────────────────────────────
            imgs_01 = to_01(images).clamp(0, 1)
            imgs_jepa = vjepa_from_unit_tensor(
                imgs_01,
                size=jepa.image_size,
                device=device,
                out_dtype=torch.float32,
            )
            imgs_vid = imgs_jepa.unsqueeze(2)

            with torch.no_grad():
                jepa_emb = jepa(imgs_vid)

            # ── Diffusion step ──────────────────────────────────────────────
            if model_key == "flux":
                loss = flux_train_step(
                    pipe,
                    projector,
                    jepa_emb,
                    images,
                    noise_scheduler,
                    guidance_scale=args.flux_guidance_scale,
                )
            else:
                loss = sd3_train_step(
                    pipe, projector, jepa_emb, images, noise_scheduler
                )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % args.log_every == 0:
                avg = total_loss / (step + 1)
                print(f"  Epoch {epoch:3d} | step {step+1:5d} | "
                      f"loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        # ── Save ────────────────────────────────────────────────────────────
        ckpt_path = os.path.join(args.out_dir, f"projector_{model_key}.pth")
        torch.save({
            "epoch":      epoch,
            "state_dict": projector.state_dict(),
            "model_key":  model_key,
            "args":       vars(args),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "cosine_t_max":        total_cosine_steps,
        }, ckpt_path)
        print(f"Epoch {epoch} done — {ckpt_path}\n")

        if not args.no_epoch_decode:
            sample_img = args.epoch_decode_image or dataset.paths[0]
            preview_path = os.path.join(
                args.out_dir, f"epoch_{epoch:03d}_decode.png"
            )
            g = args.epoch_decode_guidance
            if g is None:
                g = args.flux_guidance_scale if model_key == "flux" else 7.0
            decode_epoch_preview(
                device=device,
                pipe=pipe,
                projector=projector,
                jepa=jepa,
                model_key=model_key,
                image_path=sample_img,
                out_path=preview_path,
                image_size=args.image_size,
                steps=args.epoch_decode_steps,
                guidance_scale=g,
                num_samples=args.epoch_decode_samples,
            )


@torch.no_grad()
def decode_epoch_preview(
    device,
    pipe,
    projector: nn.Module,
    jepa,
    model_key: str,
    image_path: str,
    out_path: str,
    image_size: int,
    *,
    steps: int,
    guidance_scale: float,
    num_samples: int = 1,
):
    """
    Run frozen ``pipe`` with current projector weights; save original + gens as one grid PNG.
    """
    was_training = projector.training
    projector.eval()
    try:
        img = Image.open(image_path).convert("RGB")
        img_vid = jepa.preprocess_image(img).to(device)
        jepa_emb = jepa(img_vid)

        dtype = torch.float16 if model_key == "sd3" else torch.bfloat16
        prompt_embeds, pooled_embeds = projector(jepa_emb.float())
        prompt_embeds = prompt_embeds.to(dtype)
        pooled_embeds = pooled_embeds.to(dtype)

        imgs_out = []
        for _ in range(num_samples):
            if model_key == "sd3":
                gen = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=torch.zeros_like(prompt_embeds),
                    pooled_prompt_embeds=pooled_embeds,
                    negative_pooled_prompt_embeds=torch.zeros_like(pooled_embeds),
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    height=image_size,
                    width=image_size,
                ).images[0]
            else:
                gen = pipe(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_embeds,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    height=image_size,
                    width=image_size,
                ).images[0]
            imgs_out.append(gen)

        to_t = transforms.ToTensor()
        orig = img.resize((image_size, image_size))
        grid_tensors = [to_t(orig)] + [to_t(x) for x in imgs_out]
        grid = torch.stack(grid_tensors)
        save_image(grid, out_path, nrow=len(grid_tensors))
        print(f"  Epoch preview saved → {out_path}", flush=True)
    finally:
        if was_training:
            projector.train()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Inference / decode
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt      = torch.load(args.checkpoint, map_location=device)
    model_key = ckpt["model_key"]
    saved     = ckpt["args"]
    seq_dim, pooled_dim, max_seq = MODEL_DIMS[model_key]

    hub_entry = args.jepa_hub_entry or saved.get("jepa_hub_entry") or JEPA_HUB_ENTRY_DEFAULT
    vj_sz = (
        args.vjepa_image_size
        if args.vjepa_image_size is not None
        else int(saved.get("vjepa_image_size", JEPA_IMAGE_SIZE_DEFAULT))
    )

    # ── Load encoder + projector ─────────────────────────────────────────────
    jepa = VJEPAEncoder(
        device=device,
        hub_entry=hub_entry,
        image_size=vj_sz,
        pretrained=not args.jepa_no_pretrained,
    )
    projector = JEPAProjector(
        jepa_dim    = jepa.embed_dim,
        seq_dim     = seq_dim,
        pooled_dim  = pooled_dim,
        max_seq_len = max_seq,
        hidden_dim  = saved["hidden_dim"],
    ).to(device)
    projector.load_state_dict(ckpt["state_dict"])
    projector.eval()
    print(f"Loaded {model_key} projector (epoch {ckpt['epoch']})")

    # ── Load diffusion pipeline ──────────────────────────────────────────────
    if model_key == "sd3":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            SD3_REPO, torch_dtype=torch.float16
        ).to(device)
    else:
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            FLUX_REPO, torch_dtype=torch.bfloat16
        ).to(device)

    # ── Encode input image with JEPA ─────────────────────────────────────────
    img      = Image.open(args.image).convert("RGB")
    img_vid  = jepa.preprocess_image(img).to(device)
    jepa_emb = jepa(img_vid)

    dtype     = torch.float16 if model_key == "sd3" else torch.bfloat16
    prompt_embeds, pooled_embeds = projector(jepa_emb.float())
    prompt_embeds = prompt_embeds.to(dtype)
    pooled_embeds = pooled_embeds.to(dtype)

    # ── Run diffusion ────────────────────────────────────────────────────────
    print(f"Generating {args.num_samples} samples …")
    images_out = []
    for i in range(args.num_samples):
        if model_key == "sd3":
            result = pipe(
                prompt_embeds        = prompt_embeds,
                negative_prompt_embeds = torch.zeros_like(prompt_embeds),
                pooled_prompt_embeds = pooled_embeds,
                negative_pooled_prompt_embeds = torch.zeros_like(pooled_embeds),
                num_inference_steps  = args.steps,
                guidance_scale       = args.guidance_scale,
                height               = args.image_size,
                width                = args.image_size,
            ).images[0]
        else:
            result = pipe(
                prompt_embeds        = prompt_embeds,
                pooled_prompt_embeds = pooled_embeds,
                num_inference_steps  = args.steps,
                guidance_scale       = args.guidance_scale,
                height               = args.image_size,
                width                = args.image_size,
            ).images[0]
        images_out.append(result)

    # ── Save grid ────────────────────────────────────────────────────────────
    to_t = transforms.ToTensor()
    orig = img.resize((args.image_size, args.image_size))
    grid_tensors = [to_t(orig)] + [to_t(x) for x in images_out]
    grid = torch.stack(grid_tensors)
    save_image(grid, args.out, nrow=len(grid_tensors))
    print(f"Saved → {args.out}")
    print(f"  Col 0: original input image")
    print(f"  Cols 1–{args.num_samples}: JEPA-conditioned diffusion samples")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Library API  (use from your own code)
# ─────────────────────────────────────────────────────────────────────────────

class JEPADiffusionDecoder:
    """
    Minimal end-to-end decoder: image / JEPA embedding → generated image(s).

    Example
    -------
    from jepa_diffusion_decoder import JEPADiffusionDecoder

    dec = JEPADiffusionDecoder("projector_sd3.pth")
    images = dec.decode_image(PIL_image, num_samples=4)
    images[0].save("decoded.png")
    """

    def __init__(self, checkpoint: str, device: str = "cuda"):
        ckpt      = torch.load(checkpoint, map_location=device)
        self.mkey = ckpt["model_key"]
        saved     = ckpt["args"]
        self.dev  = torch.device(device)
        seq_dim, pooled_dim, max_seq = MODEL_DIMS[self.mkey]

        hub_entry = saved.get("jepa_hub_entry", JEPA_HUB_ENTRY_DEFAULT)
        vj_sz = int(saved.get("vjepa_image_size", JEPA_IMAGE_SIZE_DEFAULT))
        pretrained = not saved.get("jepa_no_pretrained", False)
        self.encoder = VJEPAEncoder(
            device=device,
            hub_entry=hub_entry,
            image_size=vj_sz,
            pretrained=pretrained,
        )
        self.projector = JEPAProjector(
            jepa_dim    = self.encoder.embed_dim,
            seq_dim     = seq_dim,
            pooled_dim  = pooled_dim,
            max_seq_len = max_seq,
            hidden_dim  = saved["hidden_dim"],
        ).to(device)
        self.projector.load_state_dict(ckpt["state_dict"])
        self.projector.eval()

        if self.mkey == "sd3":
            from diffusers import StableDiffusion3Pipeline
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                SD3_REPO, torch_dtype=torch.float16
            ).to(device)
            self.dtype = torch.float16
        else:
            from diffusers import FluxPipeline
            self.pipe = FluxPipeline.from_pretrained(
                FLUX_REPO, torch_dtype=torch.bfloat16
            ).to(device)
            self.dtype = torch.bfloat16

    @torch.no_grad()
    def encode(self, img: Image.Image) -> torch.Tensor:
        pv = self.encoder.preprocess_image(img).to(self.dev)
        return self.encoder(pv)

    @torch.no_grad()
    def decode_embedding(
        self,
        jepa_emb:     torch.Tensor,
        num_samples:  int   = 4,
        steps:        int   = 28,
        guidance:     float = 7.0,
        size:         int   = 512,
    ):
        pe, pp = self.projector(jepa_emb.float())
        pe = pe.to(self.dtype)
        pp = pp.to(self.dtype)
        out = []
        for _ in range(num_samples):
            kw = dict(
                prompt_embeds       = pe,
                pooled_prompt_embeds= pp,
                num_inference_steps = steps,
                guidance_scale      = guidance,
                height=size, width=size,
            )
            if self.mkey == "sd3":
                kw["negative_prompt_embeds"] = torch.zeros_like(pe)
                kw["negative_pooled_prompt_embeds"] = torch.zeros_like(pp)
            out.append(self.pipe(**kw).images[0])
        return out

    def decode_image(self, img: Image.Image, **kw):
        return self.decode_embedding(self.encode(img), **kw)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def make_parser():
    p   = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def shared(s):
        s.add_argument("--data_dir",   required=True)
        s.add_argument("--out_dir",    default="vjepa2_diffusion_out")
        s.add_argument("--epochs",     type=int,   default=10)
        s.add_argument("--batch_size", type=int,   default=2)
        s.add_argument("--image_size", type=int,   default=512)
        s.add_argument("--lr",         type=float, default=1e-4)
        s.add_argument("--hidden_dim", type=int,   default=2048)
        s.add_argument("--log_every",  type=int,   default=20)
        s.add_argument(
            "--jepa-hub-entry",
            default=JEPA_HUB_ENTRY_DEFAULT,
            help="torch.hub entry in facebookresearch/vjepa2 (default: V-JEPA 2.1 ViT-L 384).",
        )
        s.add_argument(
            "--vjepa-image-size",
            type=int,
            default=JEPA_IMAGE_SIZE_DEFAULT,
            help="Encoder spatial size after resize (typically 384 for vjepa2_1_vit_large_384).",
        )
        s.add_argument(
            "--jepa-no-pretrained",
            action="store_true",
            help="Load encoder architecture without hub pretrained weights.",
        )
        s.add_argument(
            "--flux-guidance-scale",
            type=float,
            default=3.5,
            help="train-flux only: guidance tensor when the transformer uses guidance_embeds (matches FluxPipeline).",
        )
        s.add_argument(
            "--resume",
            default=None,
            metavar="PATH",
            help="Load projector.pth from a previous run and continue training. "
                 "--epochs is the last epoch index to train (e.g. 20 means stop after epoch 20). "
                 "Requires matching --hidden_dim and jepa size entry vs saved args.",
        )
        s.add_argument(
            "--no_epoch_decode",
            action="store_true",
            help="Disable diffusion preview after each epoch (default: run one decode and save a grid).",
        )
        s.add_argument(
            "--epoch_decode_image",
            default=None,
            metavar="PATH",
            help="Image for epoch previews; default: first lexicographic file under --data_dir.",
        )
        s.add_argument(
            "--epoch_decode_steps",
            type=int,
            default=28,
            help="Denoising steps for epoch-preview decode.",
        )
        s.add_argument(
            "--epoch_decode_guidance",
            type=float,
            default=None,
            help="Guidance for preview; Flux default: --flux-guidance-scale, SD3: 7.0.",
        )
        s.add_argument(
            "--epoch_decode_samples",
            type=int,
            default=1,
            help="Number of random diffusion samples beside the resized input column in epoch PNGs.",
        )

    t3 = sub.add_parser("train-sd3",  help="Train projector for SD3")
    shared(t3)

    tf = sub.add_parser("train-flux", help="Train projector for Flux")
    shared(tf)

    d  = sub.add_parser("decode",     help="Decode an image")
    d.add_argument("--image",         required=True)
    d.add_argument("--checkpoint",    required=True)
    d.add_argument("--out",           default="decoded.png")
    d.add_argument("--num_samples",   type=int,   default=4)
    d.add_argument("--steps",         type=int,   default=28)
    d.add_argument("--guidance_scale",type=float, default=7.0)
    d.add_argument("--image_size",    type=int,   default=512)
    d.add_argument(
        "--jepa-hub-entry",
        default=None,
        help="Override checkpoint; default restores from checkpoint or ViT-L 2.1 384.",
    )
    d.add_argument(
        "--vjepa-image-size",
        type=int,
        default=None,
        help="Override checkpoint encoder input size.",
    )
    d.add_argument(
        "--jepa-no-pretrained",
        action="store_true",
        help="Load random encoder weights (mostly for debugging).",
    )

    return p


if __name__ == "__main__":
    args = make_parser().parse_args()
    if args.cmd == "train-sd3":
        train(args, "sd3")
    elif args.cmd == "train-flux":
        train(args, "flux")
    elif args.cmd == "decode":
        decode(args)