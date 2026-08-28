"""LumaFlux model -- adapts a frozen FLUX.1-dev MM-DiT for SDR->HDR ITM.

Wiring (paper Fig. 3 / Algorithm 1):
  * Each FLUX block is wrapped so that, before the frozen block runs, the image
    latent tokens are modulated by PCM (FiLM from SigLIP) and the HDR Residual
    Coupler (physical + perceptual additive residual).
  * Each block's attention uses a :class:`PGAAttnProcessor`, injecting the gated
    low-rank physical residual into the value projection.
  * After the frozen VAE decodes the final latent, the RQS tone-field decoder
    expands luma into PQ/BT.2020 HDR.

The backbone (transformer + VAE) and SigLIP are frozen; only the adapters,
coupler, RQS head, modulation conditioner and feature/connector layers train.

Flow-matching convention follows FLUX/SD3: ``z = (1-sigma) * data + sigma * noise``
with the transformer predicting velocity ``v = noise - data`` (data estimate
``= z - sigma * v``).  SDR enters purely through conditioning, so the pretrained
prior stays meaningful (``init_from_sdr`` offers an optional SDEdit-style start at
inference, in the spirit of Algorithm 1 line 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .context import LumaContext
from .features import PhysicalFeatureExtractor, PerceptualEncoder, map_to_tokens, tokens_to_grid
from .modulation import TimestepLayerModulation
from .pga import PGAModule, PGAAttnProcessor
from .pcm import PCMModule
from .coupler import HDRResidualCoupler
from .rqs import RQSToneDecoder
from .color import pq_oetf  # noqa: F401  (re-export convenience)


@dataclass
class LumaFluxConfig:
    phys_dim: int = 64
    global_dim: int = 64
    num_bands: int = 16
    pga_rank: int = 8
    adapter_hidden: int = 64       # bottleneck width for PCM/Coupler (keeps params minimal)
    rqs_knots: int = 8
    siglip_name: str = "google/siglip2-so400m-patch16-512"
    siglip_dim: int = 1152
    mock_siglip: bool = False
    guidance_scale: float = 1.0           # FLUX.1-dev is guidance-distilled
    vae_scale_factor: int = 8


class LumaBlockWrapper(nn.Module):
    """Wraps a frozen FLUX block; applies PCM + Coupler, then sets PGA step state."""

    def __init__(self, block: nn.Module, ell: int, context: LumaContext):
        super().__init__()
        self.block = block
        self.ell = ell
        self.context = context  # plain dataclass -> not registered as a submodule

    def forward(self, hidden_states, encoder_hidden_states, temb,
                image_rotary_emb=None, joint_attention_kwargs=None):
        ctx = self.context
        if ctx.enabled:
            mod = ctx.modulation(ctx.t, self.ell)
            hidden_states = ctx.pcm[self.ell](hidden_states, ctx.tperc_tokens, mod)
            hidden_states = ctx.coupler[self.ell](
                hidden_states, ctx.tphys_tokens, ctx.tperc_tokens, mod)
            proc = self.block.attn.processor
            if isinstance(proc, PGAAttnProcessor):
                proc.set_step(mod)
        return self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )


class LumaFlux(nn.Module):
    def __init__(self, transformer: nn.Module, vae: nn.Module,
                 config: LumaFluxConfig | None = None):
        super().__init__()
        self.config = config or LumaFluxConfig()
        self.transformer = transformer
        self.vae = vae

        heads = transformer.config.num_attention_heads
        head_dim = transformer.config.attention_head_dim
        dim = heads * head_dim
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim

        num_dual = len(transformer.transformer_blocks)
        num_single = len(transformer.single_transformer_blocks)
        self.num_blocks = num_dual + num_single

        # --- trainable adapters ---
        cfg = self.config
        self.modulation = TimestepLayerModulation(self.num_blocks)
        self.physical = PhysicalFeatureExtractor(cfg.phys_dim, cfg.global_dim, cfg.num_bands)
        self.perceptual = PerceptualEncoder(dim, cfg.siglip_dim, cfg.mock_siglip, cfg.siglip_name)
        self.rqs = RQSToneDecoder(dim, cfg.rqs_knots)

        self.pga = nn.ModuleList([
            PGAModule(dim, heads, head_dim, cfg.phys_dim, cfg.global_dim, cfg.num_bands, cfg.pga_rank)
            for _ in range(self.num_blocks)
        ])
        self.pcm = nn.ModuleList([PCMModule(dim, cfg.adapter_hidden)
                                  for _ in range(self.num_blocks)])
        self.coupler = nn.ModuleList([HDRResidualCoupler(dim, cfg.phys_dim, cfg.adapter_hidden)
                                      for _ in range(self.num_blocks)])

        # --- shared conditioning context (plain object; holds refs for wrappers) ---
        self.context = LumaContext()
        self.context.modulation = self.modulation
        self.context.pcm = self.pcm
        self.context.coupler = self.coupler

        self._install_adapters(num_dual, num_single)
        self._freeze_backbone()

        # Prompt-free null text embeddings (overridable with real "" embeddings).
        self.register_buffer("null_t5", torch.zeros(1, 1, transformer.config.joint_attention_dim))
        self.register_buffer("null_clip", torch.zeros(1, transformer.config.pooled_projection_dim))

    # ------------------------------------------------------------------ #
    def _install_adapters(self, num_dual: int, num_single: int) -> None:
        idx = 0
        for blocks in (self.transformer.transformer_blocks,
                       self.transformer.single_transformer_blocks):
            for j in range(len(blocks)):
                block = blocks[j]
                block.attn.processor = PGAAttnProcessor(self.pga[idx], self.context)
                blocks[j] = LumaBlockWrapper(block, idx, self.context)
                idx += 1

    def _freeze_backbone(self) -> None:
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.transformer.eval()
        self.vae.eval()

    def enable_gradient_checkpointing(self) -> None:
        """Checkpoint the frozen backbone blocks to save activation memory.

        Note: the conditioning context must stay populated through ``backward()``
        (the blocks are recomputed there), so :meth:`training_step` deliberately
        does not reset it.
        """
        self.transformer.enable_gradient_checkpointing()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def set_null_text(self, t5: torch.Tensor, clip: torch.Tensor) -> None:
        self.null_t5 = t5.to(self.null_t5.device, self.null_t5.dtype)
        self.null_clip = clip.to(self.null_clip.device, self.null_clip.dtype)

    # ------------------------------------------------------------------ #
    # VAE latent helpers (FLUX packing).
    # ------------------------------------------------------------------ #
    def _vae_dtype(self):
        return next(self.vae.parameters()).dtype

    @staticmethod
    def _pack(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        x = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(b, (h // 2) * (w // 2), c * 4)

    @staticmethod
    def _unpack(latents: torch.Tensor, lat_h: int, lat_w: int) -> torch.Tensor:
        b, n, c4 = latents.shape
        c = c4 // 4
        x = latents.view(b, lat_h // 2, lat_w // 2, c, 2, 2).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(b, c, lat_h, lat_w)

    def _image_ids(self, lat_h: int, lat_w: int, device, dtype) -> torch.Tensor:
        ids = torch.zeros(lat_h // 2, lat_w // 2, 3, device=device, dtype=dtype)
        ids[..., 1] += torch.arange(lat_h // 2, device=device, dtype=dtype)[:, None]
        ids[..., 2] += torch.arange(lat_w // 2, device=device, dtype=dtype)[None, :]
        return ids.reshape(-1, 3)

    @torch.no_grad()
    def encode(self, img01: torch.Tensor) -> torch.Tensor:
        """RGB in [0,1] -> packed FLUX latent [B, N, 64]."""
        x = (img01 * 2.0 - 1.0).to(self._vae_dtype())
        z = self.vae.encode(x).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return self._pack(z)

    def decode(self, packed: torch.Tensor, lat_h: int, lat_w: int) -> torch.Tensor:
        """Packed latent -> RGB in [0,1] (differentiable through frozen VAE)."""
        z = self._unpack(packed, lat_h, lat_w)
        z = z / self.vae.config.scaling_factor + self.vae.config.shift_factor
        img = self.vae.decode(z.to(self._vae_dtype())).sample
        return (img / 2.0 + 0.5).clamp(0.0, 1.0).float()

    # ------------------------------------------------------------------ #
    # Conditioning.
    # ------------------------------------------------------------------ #
    def set_conditioning(self, sdr: torch.Tensor, t: torch.Tensor,
                         lat_h: int, lat_w: int) -> None:
        tok_h, tok_w = lat_h // 2, lat_w // 2
        phys = self.physical(sdr)
        tperc = self.perceptual(sdr)  # [B, Np, dim]

        ctx = self.context
        ctx.enabled = True
        ctx.t = t
        ctx.tphys_tokens = map_to_tokens(phys.tphys_map, tok_h, tok_w).to(self.dim_dtype())
        ctx.tperc_tokens = tokens_to_grid(tperc, tok_h, tok_w).to(self.dim_dtype())
        ctx.tphys_pooled = phys.tphys_map.mean(dim=[2, 3])
        ctx.g = phys.g
        ctx.r = phys.r
        ctx.num_img_tokens = tok_h * tok_w

    def dim_dtype(self):
        return next(self.modulation.parameters()).dtype

    def predict_velocity(self, z: torch.Tensor, sigma: torch.Tensor,
                         lat_h: int, lat_w: int) -> torch.Tensor:
        b = z.shape[0]
        device = z.device
        tdtype = next(self.transformer.parameters()).dtype
        enc = self.null_t5.expand(b, -1, -1).to(device, tdtype)
        pooled = self.null_clip.expand(b, -1).to(device, tdtype)
        img_ids = self._image_ids(lat_h, lat_w, device, tdtype)
        txt_ids = torch.zeros(enc.shape[1], 3, device=device, dtype=tdtype)

        guidance = None
        if getattr(self.transformer.config, "guidance_embeds", False):
            guidance = torch.full((b,), self.config.guidance_scale, device=device, dtype=tdtype)

        out = self.transformer(
            hidden_states=z.to(tdtype),
            encoder_hidden_states=enc,
            pooled_projections=pooled,
            timestep=sigma.to(tdtype),
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]
        return out.float()

    # ------------------------------------------------------------------ #
    # Training.
    # ------------------------------------------------------------------ #
    def training_step(self, sdr: torch.Tensor, hdr: torch.Tensor):
        """sdr, hdr: ``[B,3,H,W]`` in [0,1] (sdr=BT.709 display, hdr=PQ/BT.2020).

        Returns ``(x_hat_pq, v_pred, v_target)`` for the loss module.
        """
        b, _, h, w = hdr.shape
        lat_h, lat_w = h // self.config.vae_scale_factor, w // self.config.vae_scale_factor

        x1 = self.encode(hdr).float()                       # HDR latent (data)
        noise = torch.randn_like(x1)
        sigma = _sample_sigma(b, x1.device)                 # [B]
        s = sigma.view(b, 1, 1)
        z = (1.0 - s) * x1 + s * noise
        v_target = noise - x1

        self.set_conditioning(sdr, sigma, lat_h, lat_w)
        v_pred = self.predict_velocity(z, sigma, lat_h, lat_w)

        x1_est = z - s * v_pred                             # data estimate
        x_out = self.decode(x1_est, lat_h, lat_w)           # RGB [0,1]
        x_hat = self.rqs(x_out, self._latent_pool(x1_est))
        # NB: do not reset the context here -- with gradient checkpointing the
        # block forward is recomputed during backward() and must still see the
        # conditioning.  It is overwritten by the next set_conditioning() call.
        return x_hat, v_pred, v_target

    def _latent_pool(self, z: torch.Tensor) -> torch.Tensor:
        """Pool packed latent [B, N, 64] to the RQS conditioning vector [B, dim]."""
        # Project 64-d packed channels into `dim` via the transformer x_embedder
        # (frozen) so the RQS head sees a backbone-consistent summary.
        with torch.no_grad():
            emb = self.transformer.x_embedder(z.to(next(self.transformer.parameters()).dtype))
        return emb.float().mean(dim=1)

    # ------------------------------------------------------------------ #
    # Inference (Algorithm 1).
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(self, sdr: torch.Tensor, num_steps: int = 40,
               init_from_sdr: bool = False, sdr_strength: float = 0.6) -> torch.Tensor:
        b, _, h, w = sdr.shape
        lat_h, lat_w = h // self.config.vae_scale_factor, w // self.config.vae_scale_factor
        device = sdr.device

        noise = torch.randn(b, (lat_h // 2) * (lat_w // 2), 64, device=device)
        if init_from_sdr:
            x_sdr = self.encode(sdr).float()
            z = (1.0 - sdr_strength) * x_sdr + sdr_strength * noise
            sigmas = torch.linspace(sdr_strength, 0.0, num_steps + 1, device=device)
        else:
            z = noise
            sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            sigma = sigmas[i].expand(b)
            self.set_conditioning(sdr, sigma, lat_h, lat_w)
            v = self.predict_velocity(z, sigma, lat_h, lat_w)
            z = z + (sigmas[i + 1] - sigmas[i]) * v

        x_out = self.decode(z, lat_h, lat_w)
        x_hat = self.rqs(x_out, self._latent_pool(z))
        self.context.reset()
        return x_hat  # PQ/BT.2020 in [0,1]

    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained_flux(cls, flux_path: str, config: LumaFluxConfig | None = None,
                             torch_dtype=torch.bfloat16) -> "LumaFlux":
        from diffusers import FluxTransformer2DModel, AutoencoderKL
        transformer = FluxTransformer2DModel.from_pretrained(
            flux_path, subfolder="transformer", torch_dtype=torch_dtype)
        vae = AutoencoderKL.from_pretrained(flux_path, subfolder="vae", torch_dtype=torch_dtype)
        return cls(transformer, vae, config)


def _sample_sigma(b: int, device, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Logit-normal sigma sampling in (0,1), as used by SD3/FLUX training."""
    n = torch.randn(b, device=device) * std + mean
    return torch.sigmoid(n).clamp(1e-3, 1.0 - 1e-3)
