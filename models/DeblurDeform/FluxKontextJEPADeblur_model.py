"""
FluxJEPADeblur_model — Flux deblurring with I-JEPA structural conditioning.

Novel contribution: replaces/augments VLM text captions with I-JEPA patch-token
features from the blurry input image, injected as additional encoder_hidden_states
tokens in the Flux transformer. JEPA captures structural/spatial relationships that
language cannot describe, making it better suited as a conditioning signal for
restoration than text captions.

Architecture overview:
    LQ image ──► I-JEPA ViT encoder (frozen) ──► patch tokens [B, N, D_jepa]
                                                  │
                                      JEPAProjection MLP (trainable)
                                                  │
                                             [B, N, 4096]
                                                  │
                        (optional) concat with T5 text embeddings
                                                  │
                                    encoder_hidden_states in FluxTransformer2DModel

No prior work uses JEPA features to condition a Flux-based restoration model.

Config keys (in addition to FluxDeblur_model keys):
    jepa_model_path:        HuggingFace model ID or local path to I-JEPA ViT weights
                            (default: "facebook/ijepa_vith14_1k")
    jepa_image_size:        Input size for JEPA encoder (default: 224)
    jepa_projection_layers: Depth of projection MLP (default: 2)
    finetune_jepa:          Whether to fine-tune the JEPA encoder (default: false)
    use_text_conditioning:  Concat T5 captions alongside JEPA tokens (default: true)
    jepa_use_proxy:         Use restore proxy image for JEPA features (default: true)
"""

import gc
import copy
import einops
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
)
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from ram import inference_ram as inference
from transformers import AutoModel

from models.DeblurDeform.FluxKontextDeblur_model import (
    FluxKontextDeblur_model,
    encode_images_flux,
)
from utils import log_image, log_metrics
from utils.dataset_utils import merge_patches
from models.archs import find_network_class


# ---------------------------------------------------------------------------
# Scheduler helpers
# ---------------------------------------------------------------------------

def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Resolution-dependent time-shift (mu) for FLUX.1-dev dynamic shifting.

    Replicates diffusers.pipelines.flux.pipeline_flux.calculate_shift so that
    our manual denoising loop matches what FluxControlPipeline does internally.
    Linear interpolation: mu = m * image_seq_len + b, clamped to [base_shift, max_shift].
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = float(image_seq_len) * m + b
    return mu


# ---------------------------------------------------------------------------
# Restore proxy
# ---------------------------------------------------------------------------

class RestoreProxy(nn.Module):
    """Lightweight restoration proxy applied to the blurry LQ image before
    caption extraction, producing a partially-restored image with cleaner
    semantic content for the VLM to caption.

    Two modes (set via config proxy_type):
      "unsharp" (default) — differentiable unsharp masking; zero extra parameters.
                            Fast, always available, no checkpoint needed.
      "model"             — any pretrained arch from models/archs/ loaded from a
                            checkpoint. Frozen during training.

    Config keys:
        proxy_type:           "unsharp" | "model"          (default: "unsharp")
        proxy_unsharp_amount: strength of sharpening       (default: 1.5)
        proxy_unsharp_sigma:  Gaussian blur sigma           (default: 1.5)
        proxy_unsharp_kernel: Gaussian kernel size (odd)   (default: 9)
        proxy_arch_type:      arch class name               (model mode only)
        proxy_arch_opt:       dict of arch constructor args (model mode only)
        proxy_model_path:     path to .pth checkpoint       (model mode only)
    """

    def __init__(
        self,
        proxy_type: str = "unsharp",
        unsharp_amount: float = 1.5,
        unsharp_sigma: float = 1.5,
        unsharp_kernel: int = 9,
        arch_type: str | None = None,
        arch_opt: dict | None = None,
        model_path: str | None = None,
    ):
        super().__init__()
        self.proxy_type = proxy_type

        if proxy_type == "unsharp":
            self.amount = unsharp_amount
            self.sigma  = unsharp_sigma
            self.kernel = unsharp_kernel if unsharp_kernel % 2 == 1 else unsharp_kernel + 1
            self.net = None

        elif proxy_type == "model":
            assert arch_type is not None, "proxy_arch_type must be set when proxy_type='model'"
            net_cls, cfg_cls = find_network_class(arch_type)
            arch_opt = arch_opt or {}
            if cfg_cls is not None:
                self.net = net_cls(cfg_cls(**arch_opt))
            else:
                self.net = net_cls(**arch_opt)
            if model_path is not None:
                ckpt = torch.load(model_path, map_location="cpu")
                state = ckpt.get("params", ckpt.get("state_dict", ckpt))
                missing, unexpected = self.net.load_state_dict(state, strict=False)
                if missing:
                    print(f"[RestoreProxy] Missing keys: {missing}")
            # self.net.requires_grad_(False)
            # self.net.eval()
        else:
            raise ValueError(f"Unknown proxy_type '{proxy_type}'. Use 'unsharp' or 'model'.")

    def _unsharp_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Apply unsharp masking in [-1, 1] space."""
        from torchvision.transforms.functional import gaussian_blur
        # gaussian_blur expects uint8 or float in [0,1]; work in float directly
        x01 = x * 0.5 + 0.5                                      # → [0, 1]
        blurred = gaussian_blur(x01, kernel_size=self.kernel, sigma=self.sigma)
        sharpened = x01 + self.amount * (x01 - blurred)
        return sharpened.clamp(0, 1) * 2 - 1                     # → [-1, 1]

    @torch.no_grad()
    def forward(self, lq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lq: [B, 3, H, W] blurry image in [-1, 1]
        Returns:
            proxy: [B, 3, H, W] partially restored image in [-1, 1]
        """
        if self.proxy_type == "unsharp":
            return self._unsharp_mask(lq.float()).to(lq.dtype)
        else:
            out = self.net(lq.to(next(self.net.parameters()).dtype))
            return out.clamp(-1, 1).to(lq.dtype)


# ---------------------------------------------------------------------------
# JEPA projection MLP
# ---------------------------------------------------------------------------

class JEPAProjection(nn.Module):
    """Projects I-JEPA patch tokens into Flux's T5 text-embedding space (4096-dim).

    Uses a simple MLP: in_dim → hidden_dim → ... → out_dim with GELU activations.
    Only the final linear has no activation (standard projection head design).
    """

    def __init__(self, in_dim: int, out_dim: int = 4096, num_layers: int = 2):
        super().__init__()
        hidden_dim = in_dim * 2
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FluxKontextJEPADeblur_model(FluxKontextDeblur_model):
    """Flux deblurring model with I-JEPA structural conditioning."""

    def __init__(self, opt, logger):
        # Parent loads: transformer, vae, LoRA, noise_scheduler, text_encoding_pipeline, RAM VLM
        super().__init__(opt, logger)

        # ---- JEPA config -----------------------------------------------
        jepa_model_path = getattr(opt, "jepa_model_path", "facebook/ijepa_vith14_1k")
        self.jepa_image_size = getattr(opt, "jepa_image_size", 224)
        jepa_proj_layers = getattr(opt, "jepa_projection_layers", 2)
        self.finetune_jepa = getattr(opt, "finetune_jepa", False)
        self.use_text_conditioning = getattr(opt, "use_text_conditioning", True)
        self.jepa_use_proxy = getattr(opt, "jepa_use_proxy", True)

        # ---- Load I-JEPA encoder (AutoModel respects the ijepa model type) ----
        self.logger.info(f"Loading I-JEPA encoder from {jepa_model_path}")
        self.jepa_encoder = AutoModel.from_pretrained(jepa_model_path)
        jepa_hidden_dim = self.jepa_encoder.config.hidden_size  # 1280 for ViT-H/14

        # I-JEPA has no CLS token (position_embeddings shape [1, N_patches, D]).
        # Standard ViT has CLS at index 0 (position_embeddings shape [1, 1+N_patches, D]).
        # Detect which case we have so _extract_jepa_patch_tokens slices correctly.
        model_type = getattr(self.jepa_encoder.config, "model_type", "vit")
        self.jepa_has_cls = model_type not in ("ijepa",)
        self.logger.info(
            f"JEPA model_type={model_type}, has_cls={self.jepa_has_cls}"
        )

        self.jepa_encoder.requires_grad_(self.finetune_jepa)
        self.jepa_encoder = self.jepa_encoder.to(
            self.accelerator.device, dtype=self.weight_dtype
        )
        self.logger.info(
            f"I-JEPA encoder: hidden_dim={jepa_hidden_dim}, "
            f"{'trainable' if self.finetune_jepa else 'frozen'}"
        )

        # ---- Projection MLP: JEPA_dim → 4096 (Flux T5 dim) ------------
        flux_text_dim = 4096
        self.jepa_projection = JEPAProjection(
            in_dim=jepa_hidden_dim,
            out_dim=flux_text_dim,
            num_layers=jepa_proj_layers,
        ).to(self.accelerator.device, dtype=self.weight_dtype)

        n_proj = sum(p.numel() for p in self.jepa_projection.parameters())
        self.logger.info(
            f"JEPAProjection: {jepa_hidden_dim}→{flux_text_dim}, "
            f"{n_proj:,} trainable params"
        )

        # ---- Image preprocessing for JEPA (ImageNet stats) -------------
        self.jepa_transforms = transforms.Compose([
            transforms.Resize(
                (self.jepa_image_size, self.jepa_image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # ---- Restore proxy for caption extraction ----------------------
        # Applies a lightweight restoration to the blurry LQ image so the
        # VLM receives a cleaner image and produces more accurate captions.
        self.proxy = self._build_proxy(opt)
        self.proxy = self.proxy.to(self.accelerator.device, dtype=self.weight_dtype)
        self.logger.info(f"RestoreProxy mode='{self.proxy.proxy_type}' ready")
        if self.proxy.proxy_type == "model":
            self.proxy.net.requires_grad_(True)
            self.models.append(self.proxy.net)

        # Register projection (and optionally encoder) for checkpoint saving.
        # Frozen modules (jepa_encoder, proxy.net) are excluded to avoid
        # redundant checkpoint storage of pretrained weights.
        self.models.append(self.jepa_projection)
        if self.finetune_jepa:
            self.models.append(self.jepa_encoder)

    # ------------------------------------------------------------------
    # Proxy helpers
    # ------------------------------------------------------------------

    def _build_proxy(self, opt) -> RestoreProxy:
        proxy_type   = getattr(opt, "proxy_type",           "unsharp")
        amount       = getattr(opt, "proxy_unsharp_amount", 1.5)
        sigma        = getattr(opt, "proxy_unsharp_sigma",  1.5)
        kernel       = getattr(opt, "proxy_unsharp_kernel", 9)
        arch_type    = getattr(opt, "proxy_arch_type",      None)
        arch_opt     = getattr(opt, "proxy_arch_opt",       None)
        model_path   = getattr(opt, "proxy_model_path",     None)
        if arch_opt is not None and hasattr(arch_opt, "__dict__"):
            arch_opt = arch_opt.__dict__
        return RestoreProxy(
            proxy_type=proxy_type,
            unsharp_amount=amount,
            unsharp_sigma=sigma,
            unsharp_kernel=kernel,
            arch_type=arch_type,
            arch_opt=arch_opt,
            model_path=model_path,
        )

    @torch.no_grad()
    def _get_proxy(self, lq: torch.Tensor) -> torch.Tensor:
        """Return a partially-restored proxy of the blurry input for caption extraction."""
        return self.proxy(lq)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def setup_optimizers(self):
        """Extend parent optimizer to include JEPAProjection parameters."""
        super().setup_optimizers()
        if self.proxy.proxy_type == "model":
            proxy_params = [p for p in self.proxy.net.parameters() if p.requires_grad]
            if proxy_params:
                self.optimizers[0].add_param_group({"params": proxy_params})
                n = sum(p.numel() for p in proxy_params)
                self.logger.info(f"Added {n:,} proxy params to optimizer")
                
        proj_params = [p for p in self.jepa_projection.parameters() if p.requires_grad]
        if proj_params:
            self.optimizers[0].add_param_group({"params": proj_params})
            n = sum(p.numel() for p in proj_params)
            self.logger.info(f"Added {n:,} JEPA projection params to optimizer")

        if self.finetune_jepa:
            enc_params = [p for p in self.jepa_encoder.parameters() if p.requires_grad]
            if enc_params:
                self.optimizers[0].add_param_group({"params": enc_params})
                n = sum(p.numel() for p in enc_params)
                self.logger.info(f"Added {n:,} JEPA encoder params to optimizer")

    # ------------------------------------------------------------------
    # Core JEPA feature extraction + conditioning
    # ------------------------------------------------------------------

    def _extract_jepa_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Run I-JEPA encoder; return patch tokens (CLS excluded).

        Args:
            images: [B, 3, H, W] float in [-1, 1]
        Returns:
            [B, N_patches, D_jepa]
        """
        images_01 = images.float() * 0.5 + 0.5                   # [-1,1] → [0,1]
        images_jepa = self.jepa_transforms(images_01)             # [B, 3, 224, 224]
        images_jepa = images_jepa.to(
            self.jepa_encoder.device,
            dtype=next(self.jepa_encoder.parameters()).dtype,
        )
        outputs = self.jepa_encoder(pixel_values=images_jepa)
        # IjepaModel: last_hidden_state is [B, N_patches, D]  (no CLS token)
        # ViTModel:   last_hidden_state is [B, 1+N_patches, D] (index 0 = CLS)
        patch_tokens = (
            outputs.last_hidden_state[:, 1:, :]   # skip CLS
            if self.jepa_has_cls
            else outputs.last_hidden_state         # all tokens are patches
        )
        return patch_tokens

    def get_jepa_conditioning(
        self,
        lq_images: torch.Tensor,
        text_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build encoder_hidden_states by projecting JEPA features and
        optionally prepending them to T5 text embeddings.

        Args:
            lq_images:   [B, 3, H, W] blurry input in [-1, 1]
            text_embeds: [B, T, 4096] T5 embeddings, or None
        Returns:
            combined: [B, N_jepa (+ T), 4096]
        """
        # Gradient flows through projection MLP but not through frozen encoder
        if self.finetune_jepa:
            patch_tokens = self._extract_jepa_patch_tokens(lq_images)
        else:
            with torch.no_grad():
                patch_tokens = self._extract_jepa_patch_tokens(lq_images)

        jepa_embeds = self.jepa_projection(
            patch_tokens.to(self.jepa_projection.net[0].weight.dtype)
        ).to(self.weight_dtype)                                    # [B, N, 4096]

        if text_embeds is not None and self.use_text_conditioning:
            combined = torch.cat([jepa_embeds, text_embeds], dim=1)
        else:
            combined = jepa_embeds
        return combined

    @torch.no_grad()
    def _get_jepa_image(self, lq_images: torch.Tensor) -> torch.Tensor:
        """Choose the image fed into JEPA (raw LQ vs restore-proxy LQ)."""
        if self.jepa_use_proxy:
            return self._get_proxy(lq_images)
        return lq_images

    def _make_txt_ids(self, seq_len: int, device, dtype) -> torch.Tensor:
        """Zero position IDs for encoder tokens (consistent with Flux text token convention).
        Shape: [seq_len, 3] — no batch dimension, as required by FluxTransformer2DModel.
        """
        return torch.zeros(seq_len, 3, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def optimize_parameters(self):
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        image_1 = self.sample[lq_key]   # blurry conditioning
        gt_1    = self.sample[gt_key]   # sharp target
        bsz     = image_1.shape[0]

        if image_1.shape[1] == 1:
            image_1 = einops.repeat(image_1, 'b 1 h w -> b 3 h w')
        if gt_1.shape[1] == 1:
            gt_1 = einops.repeat(gt_1, 'b 1 h w -> b 3 h w')

        image_1 = image_1.clip(-1, 1)
        gt_1    = gt_1.clip(-1, 1)

        # Encode to Flux latents
        vae_dev = self._to_device_safe(self.vae, self.accelerator.device)
        pixel_latents   = encode_images_flux(gt_1,    vae_dev, self.weight_dtype)
        control_latents = encode_images_flux(image_1, vae_dev, self.weight_dtype)

        # Timestep sampling (flow matching)
        noise = torch.randn_like(pixel_latents)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=bsz,
            logit_mean=getattr(self.opt, "logit_mean", 0.0),
            logit_std=getattr(self.opt, "logit_std", 1.0),
            mode_scale=getattr(self.opt, "mode_scale", 1.29),
        )
        indices   = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(pixel_latents.device)

        sigmas            = self.get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
        noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

        # Concatenate noisy target + control (blur) latents. When packing, pass the
        # *base* latent channel count (pixel_latents.shape[1]) so that the internal
        # Flux packing logic reshapes tokens with the expected feature dimension.
        packed_noisy_model_input = FluxKontextPipeline._pack_latents(
            noisy_model_input,
            bsz,
            self.transformer.config.in_channels // 4,
            noisy_model_input.shape[2],
            noisy_model_input.shape[3],
        )
        packed_control_latents = FluxKontextPipeline._pack_latents(
            control_latents,
            bsz,
            control_latents.shape[1],
            control_latents.shape[2],
            control_latents.shape[3],
        )
        model_input = torch.cat([packed_noisy_model_input, packed_control_latents], dim=1)

        latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
            bsz,
            noisy_model_input.shape[2] // 2,
            noisy_model_input.shape[3] // 2,
            self.accelerator.device,
            self.weight_dtype,
        )
        image_ids = FluxKontextPipeline._prepare_latent_image_ids(
            bsz, 
            noisy_model_input.shape[2] // 2, 
            noisy_model_input.shape[3] // 2, 
            self.accelerator.device, 
            self.weight_dtype,
        )
        latent_image_ids = torch.cat([latent_image_ids, image_ids], dim=0) # dim 0 is sequence dimension

        # Guidance vector (FLUX.1-dev uses guidance distillation)
        flux_transformer = self._unwrap_model(self.transformer)
        guidance_vec = (
            torch.full((bsz,), self.guidance_scale,
                       device=noisy_model_input.device, dtype=self.weight_dtype)
            if getattr(flux_transformer.config, "guidance_embeds", False)
            else None
        )

        # ---- Text conditioning (optional) -----------------------------
        with torch.no_grad():
            self.text_encoding_pipeline = self.text_encoding_pipeline.to(
                self.accelerator.device
            )
            if self.use_text_conditioning:
                if self.caption_from_gt:
                    # GT available during training: caption from sharp image (best quality)
                    caption_img = gt_1
                else:
                    # No GT at inference: apply restore proxy to blurry image so the
                    # VLM receives a sharper image and produces more accurate captions.
                    caption_img = self._get_proxy(image_1)
                gt_ram  = self.model_vlm_transforms(caption_img * 0.5 + 0.5)
                caption = inference(gt_ram.to(dtype=torch.float16), self.model_vlm)
                prompt_embeds, pooled_prompt_embeds, _ = (
                    self.text_encoding_pipeline.encode_prompt(list(caption), prompt_2=None)
                )
            else:
                # Caption-free: empty strings give pooled embeds for guidance
                prompt_embeds, pooled_prompt_embeds, _ = (
                    self.text_encoding_pipeline.encode_prompt([""] * bsz, prompt_2=None)
                )

        # ---- JEPA conditioning ----------------------------------------
        # Gradient flows through projection MLP; JEPA encoder is frozen.
        jepa_img = self._get_jepa_image(image_1)
        combined_embeds = self.get_jepa_conditioning(
            jepa_img,
            text_embeds=prompt_embeds if self.use_text_conditioning else None,
        )  # [B, N_jepa (+ T), 4096]

        # txt_ids: zeros over combined sequence (Flux convention for text tokens)
        # Shape must be [seq_len, 3] — no batch dim (3D is deprecated).
        combined_txt_ids = self._make_txt_ids(
            combined_embeds.shape[1],
            self.accelerator.device, self.weight_dtype,
        )

        # ---- Transformer forward pass ---------------------------------
        model_pred = self.transformer(
            hidden_states=model_input,
            timestep=timesteps / 1000.0,
            guidance=guidance_vec,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=combined_embeds,
            txt_ids=combined_txt_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        model_pred = model_pred[:, : packed_noisy_model_input.size(1)]
        
        model_pred = FluxKontextPipeline._unpack_latents(
            model_pred,
            height=noisy_model_input.shape[2] * self.vae_scale_factor,
            width=noisy_model_input.shape[3] * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )

        # Flow-matching loss: predict velocity field (noise − x0)
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme, sigmas=sigmas
        )
        target = noise - pixel_latents
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2
             ).reshape(target.shape[0], -1),
            1,
        ).mean()

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.transformer.parameters(),
                getattr(self.opt.train.optim, "max_grad_norm", 1.0),
            )
        torch.cuda.empty_cache()
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()
        torch.cuda.empty_cache()

        return {"all": loss}

    # ------------------------------------------------------------------
    # Validation — manual denoising loop (bypasses FluxControlPipeline
    # so we can inject JEPA conditioning at every denoising step)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validation(self):
        gc.collect()
        torch.cuda.empty_cache()

        for model in self.models:
            model.eval()

        # Text encoders needed for encoding prompts
        self.text_encoding_pipeline = self.text_encoding_pipeline.to(
            self.accelerator.device
        )
        # VAE decode in float32 for numerical stability
        self.vae.to(torch.float32)

        idx = 0
        for batch in tqdm(self.test_dataloader):
            idx = self.validate_step(
                batch, idx,
                self.test_dataloader.dataset.lq_key,
                self.test_dataloader.dataset.gt_key,
            )
            if idx >= self.max_val_steps:
                break

        gc.collect()
        torch.cuda.empty_cache()

        # Restore training dtype and mode
        self.vae.to(self.weight_dtype)
        for model in self.models:
            self._to_device_safe(model, self.accelerator.device)
            model.train()

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode Flux latents back to pixel space [-1, 1]."""
        shift = getattr(self.vae.config, "shift_factor", 0.0)
        scale = getattr(self.vae.config, "scaling_factor", 1.0)
        latents_in = latents.to(torch.float32) / scale + shift
        image = self.vae.decode(latents_in.to(self.vae.dtype)).sample
        return image.float().clamp(-1, 1)

    @torch.no_grad()
    def _run_jepa_inference(
        self,
        lq: torch.Tensor,
        height: int,
        width: int,
        num_steps: int,
    ) -> torch.Tensor:
        """Single-image denoising with JEPA conditioning.

        Args:
            lq:     [1, 3, H, W] blurry image in [-1, 1]
            height: pixel height
            width:  pixel width
            num_steps: number of denoising steps
        Returns:
            [1, 3, H, W] deblurred image in [-1, 1]
        """
        bsz = lq.shape[0]
        device = lq.device

        # ---- Text encoding (or empty caption) -------------------------
        # Apply restore proxy before captioning: the VLM sees a sharpened
        # version of the blurry input, producing more accurate content tags.
        if self.use_text_conditioning:
            proxy_img = self._get_proxy(lq)          # [B, 3, H, W] in [-1, 1]
            lq_ram    = self.model_vlm_transforms(proxy_img * 0.5 + 0.5)
            caption   = inference(lq_ram.to(dtype=torch.float16), self.model_vlm)
            captions  = list(caption)
        else:
            captions = [""] * bsz

        prompt_embeds, pooled_prompt_embeds, _ = (
            self.text_encoding_pipeline.encode_prompt(captions, prompt_2=None)
        )

        # ---- JEPA conditioning ----------------------------------------
        jepa_img = self._get_jepa_image(lq)
        combined_embeds = self.get_jepa_conditioning(
            jepa_img,
            text_embeds=prompt_embeds if self.use_text_conditioning else None,
        )  # [B, N_jepa(+T), 4096]
        combined_txt_ids = self._make_txt_ids(
            combined_embeds.shape[1], device, self.weight_dtype
        )

        # ---- Control latents (blurry image) ---------------------------
        vae_dev = self._to_device_safe(self.vae, device)
        control_latents = encode_images_flux(lq, vae_dev, self.weight_dtype)

        # Derive latent spatial size directly from encoded latents to stay
        # consistent with the VAE config (scale_factor, padding, etc.).
        lat_c = control_latents.shape[1]        # base Flux latent channels (e.g. 16)
        lat_h = control_latents.shape[2]
        lat_w = control_latents.shape[3]

        latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
            bsz, lat_h // 2, lat_w // 2, device, self.weight_dtype
        )
        image_ids = FluxKontextPipeline._prepare_latent_image_ids(
            bsz, lat_h // 2, lat_w // 2, device, self.weight_dtype
        )
        latent_image_ids = torch.cat([latent_image_ids, image_ids], dim=0) # dim 0 is sequence dimension

        # ---- Guidance vec ----------------------------------------------
        flux_transformer = self._unwrap_model(self.transformer)
        guidance_vec = (
            torch.full((bsz,), self.guidance_scale, device=device, dtype=self.weight_dtype)
            if getattr(flux_transformer.config, "guidance_embeds", False)
            else None
        )

        # ---- Scheduler ------------------------------------------------
        # FLUX.1-dev uses dynamic time-shifting: set_timesteps requires a mu
        # value computed from the packed token sequence length (resolution-dependent).
        # This mirrors FluxKontextPipeline._get_t5_prompt_embeds / prepare_latents logic.
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.opt.pretrained_model_name_or_path, subfolder="scheduler"
        )
        if getattr(scheduler.config, "use_dynamic_shifting", False):
            # image_seq_len = number of packed spatial tokens
            image_seq_len = (lat_h // 2) * (lat_w // 2)
            mu = _calculate_shift(
                image_seq_len,
                base_seq_len=getattr(scheduler.config, "base_image_seq_len", 256),
                max_seq_len=getattr(scheduler.config, "max_image_seq_len", 4096),
                base_shift=getattr(scheduler.config, "base_shift", 0.5),
                max_shift=getattr(scheduler.config, "max_shift", 1.15),
            )
            scheduler.set_timesteps(num_steps, device=device, mu=mu)
        else:
            scheduler.set_timesteps(num_steps, device=device)

        # ---- Initial noise latents ------------------------------------
        latents = torch.randn(bsz, lat_c, lat_h, lat_w,
                              device=device, dtype=self.weight_dtype)
        
        # ---- Denoising loop -------------------------------------------
        for t in scheduler.timesteps:
            # Use only the base latents when packing, matching the official
            # FluxKontext `prepare_latents` behavior (conditioning comes via
            # JEPA/text embeds, not via channel concatenation here).
            packed_latents = FluxKontextPipeline._pack_latents(
                latents,
                bsz,
                lat_c,
                lat_h,
                lat_w,
            )
            packed_control_latents = FluxKontextPipeline._pack_latents(
                control_latents,
                bsz,
                lat_c,
                lat_h,
                lat_w,
            )
            model_input = torch.cat([packed_latents, packed_control_latents], dim=1)

            noise_pred = flux_transformer(
                hidden_states=model_input,
                timestep=t.unsqueeze(0).to(self.weight_dtype) / 1000.0,
                guidance=guidance_vec,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=combined_embeds,
                txt_ids=combined_txt_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : packed_latents.size(1)]
            
            noise_pred = FluxKontextPipeline._unpack_latents(
                noise_pred,
                height=lat_h * self.vae_scale_factor,
                width=lat_w  * self.vae_scale_factor,
                vae_scale_factor=self.vae_scale_factor,
            )
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return self._decode_latents(latents)   # [B, 3, H, W] in [-1, 1]

    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        num_steps = getattr(self.opt.val, "num_inference_steps", 50)

        if self.opt.val.patched:
            b, c, h, w = self.original_size[lq_key]
            pred_patches = []
            for _ in self.setup_patches():
                lq_patch = self.sample[lq_key]
                if lq_patch.shape[1] == 1:
                    lq_patch = einops.repeat(lq_patch, "b 1 h w -> b 3 h w")
                out_list = []
                for i in range(lq_patch.shape[0]):
                    out = self._run_jepa_inference(
                        lq_patch[i:i+1],
                        lq_patch.shape[-2], lq_patch.shape[-1],
                        num_steps,
                    )
                    out_list.append(out)
                pred_patches.append(torch.cat(out_list, dim=0))
            pred = torch.cat(pred_patches, dim=0)
            pred = einops.rearrange(pred, "(b n) c h w -> b n c h w", b=b)
            out_list = []
            for i in range(len(pred)):
                merged = merge_patches(pred[i], self.sample[lq_key + "_patched_pos"])
                out_list.append(merged[..., :h, :w])
            lq1 = self.sample[lq_key + "_original"]
            gt  = self.sample[gt_key + "_original"]
            out = torch.stack(out_list)
        else:
            lq1 = self.sample[lq_key]
            gt  = self.sample[gt_key]
            if lq1.shape[1] == 1:
                lq1 = einops.repeat(lq1, "b 1 h w -> b 3 h w")
            out_list = []
            for i in range(lq1.shape[0]):
                o = self._run_jepa_inference(
                    lq1[i:i+1],
                    lq1.shape[-2], lq1.shape[-1],
                    num_steps,
                )
                out_list.append(o)
            out = torch.cat(out_list, dim=0)

        # Convert to numpy [0, 1] for logging
        lq1_np = (lq1.cpu().float().numpy() * 0.5 + 0.5)
        gt_np  = (gt.cpu().float().numpy()  * 0.5 + 0.5)
        out_np = (out.cpu().float().numpy() * 0.5 + 0.5)

        for i in range(len(gt_np)):
            idx += 1
            imgs = [lq1_np[i], gt_np[i], out_np[i]]
            for j in range(len(imgs)):
                if imgs[j].shape[0] == 1:
                    imgs[j] = einops.repeat(imgs[j], "1 h w -> 3 h w")
            image_grid = np.clip(np.stack(imgs), 0, 1)
            log_image(self.opt, self.accelerator, image_grid,
                      f"{idx:04d}", self.global_step)
            log_image(self.opt, self.accelerator,
                      np.clip(np.stack([out_np[i]]), 0, 1),
                      f"out_{idx:04d}", self.global_step)
            log_metrics(gt_np[i], out_np[i], self.opt.val.metrics,
                        self.accelerator, self.global_step)
        return idx
