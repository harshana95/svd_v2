"""End-to-end Flux2 deblurring with LQ blur-map + JEPA Coupler/PGA/PCM adapters."""

from __future__ import annotations

import copy
import gc
import os

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel

try:
    from diffusers import AutoencoderKLFlux2
except ImportError:
    from diffusers import AutoencoderKL as AutoencoderKLFlux2
from tqdm import tqdm

from models.base_model import BaseModel
from models.LucidFlux.helpers import (
    PROMPT,
    calculate_shift,
    decode_latents_flux,
    encode_vae_image,
    get_flux2_pipeline_class,
    get_noisy_model_input_and_timesteps,
    prepare_image_latents,
)
from models.LucidFlux.jepa_helpers import (
    extract_vjepa_tokens,
    get_vjepa_feature_dim,
    load_vjepa2_model,
    load_vjepa2_weights,
    vjepa_from_unit_tensor,
)
from models.LucidFlux.src.flux.lucidflux import load_precomputed_embeddings, load_swinir
from utils import log_image, log_metrics, save_eval_png

from .blur_adapters import (
    BlurMapExtractor,
    JEPAPerceptualEncoder,
    install_blur_adapters,
    map_to_tokens,
    tokens_to_grid,
)


class ContextFluxBlurMap_model(BaseModel):
    """Frozen Flux2 + trainable blur-map / JEPA adapters (Coupler + PGA + PCM)."""

    def __init__(self, opt, logger):
        super().__init__(opt, logger)

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
        self.vjepa_dtype = getattr(opt, "vjepa_dtype", torch.float32)

        pretrained = opt.pretrained_model_name_or_path
        self.flux2_pipe_cls = get_flux2_pipeline_class(pretrained)
        default_guidance_scale = 1.0 if "klein" in pretrained.lower() else 4.0

        self.blur_dim = int(getattr(opt, "blur_dim", 64))
        self.global_dim = int(getattr(opt, "global_dim", 64))
        self.num_bands = int(getattr(opt, "num_bands", 16))
        self.pga_rank = int(getattr(opt, "pga_rank", 8))
        self.adapter_hidden = int(getattr(opt, "adapter_hidden", 64))
        self.use_condition_image_for_transformer = bool(
            getattr(opt, "use_condition_image_for_transformer", False)
        )
        self.use_vjepa_condition = bool(getattr(opt, "use_vjepa_condition", True))

        self.vjepa_hub_entry = getattr(opt, "vjepa_hub_entry", "vjepa2_1_vit_large_384")
        self.vjepa_image_size = int(getattr(opt, "vjepa_image_size", 512))
        self.vjepa_pretrained = getattr(opt, "vjepa_pretrained", True)
        self.vjepa_checkpoint_path = getattr(opt, "vjepa_checkpoint_path", None)
        self.vjepa_feature_dim = get_vjepa_feature_dim(
            self.vjepa_hub_entry,
            fallback_dim=getattr(opt, "vjepa_feature_dim", 1024),
        )
        self.vjepa_source_image = getattr(opt, "vjepa_source_image", "lq")

        # --- frozen V-JEPA ---
        self.vjepa = None
        if self.use_vjepa_condition:
            logger.info(
                "Loading V-JEPA 2.1 encoder from torch.hub entry %s", self.vjepa_hub_entry
            )
            self.vjepa = load_vjepa2_model(
                self.vjepa_hub_entry,
                "cpu",
                self.vjepa_dtype,
                offload=False,
                pretrained=self.vjepa_pretrained,
            )
            if self.vjepa_checkpoint_path and os.path.exists(self.vjepa_checkpoint_path):
                logger.info(
                    "Loading V-JEPA checkpoint from %s", self.vjepa_checkpoint_path
                )
                load_vjepa2_weights(self.vjepa, self.vjepa_checkpoint_path)
            self.vjepa.requires_grad_(False)
            self.vjepa.eval()

        swinir_path = getattr(opt, "swinir_model_path", None)
        self.swinir = None
        if swinir_path and os.path.exists(swinir_path):
            logger.info("Loading SwinIR from %s", swinir_path)
            self.swinir = load_swinir("cpu", swinir_path, offload=False)

        embeddings_path = getattr(opt, "precomputed_embeddings_path", None)
        if embeddings_path and os.path.exists(embeddings_path):
            embeds = load_precomputed_embeddings(embeddings_path, "cpu")
            self.register_buffer_txt = embeds["txt"]
            self.register_buffer_txt_ids = embeds.get("txt_ids")
        else:
            logger.warning(
                "No precomputed embeddings path found. "
                "Will generate null embeddings at first forward pass."
            )
            self.register_buffer_txt = None
            self.register_buffer_txt_ids = None

        # --- frozen Flux2 ---
        self.guidance_scale = getattr(opt, "guidance_scale", default_guidance_scale)
        self.discrete_flow_shift = getattr(opt, "discrete_flow_shift", 3.1582)
        revision = getattr(opt, "revision", None)
        variant = getattr(opt, "variant", None)

        self.transformer = Flux2Transformer2DModel.from_pretrained(
            pretrained, subfolder="transformer", revision=revision, variant=variant
        )
        self.transformer.requires_grad_(False)
        self.transformer.eval()
        if getattr(opt, "transformer_gradient_checkpointing", False):
            if hasattr(self.transformer, "enable_gradient_checkpointing"):
                self.transformer.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled for frozen transformer")

        self.vae = AutoencoderKLFlux2.from_pretrained(
            pretrained, subfolder="vae", revision=revision, variant=variant
        )
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained, subfolder="scheduler"
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.use_transformer_guidance = getattr(
            self.transformer.config, "guidance_embeds", True
        )

        heads = self.transformer.config.num_attention_heads
        head_dim = self.transformer.config.attention_head_dim
        self.inner_dim = heads * head_dim

        # --- trainable adapters ---
        (
            self.blur_context,
            self.modulation,
            self.pga,
            self.pcm,
            self.coupler,
        ) = install_blur_adapters(
            self.transformer,
            blur_dim=self.blur_dim,
            global_dim=self.global_dim,
            num_bands=self.num_bands,
            pga_rank=self.pga_rank,
            adapter_hidden=self.adapter_hidden,
        )
        self.blur_extractor = BlurMapExtractor(
            self.blur_dim, self.global_dim, self.num_bands
        )
        self.jepa_connector = None
        if self.vjepa is not None:
            self.jepa_connector = JEPAPerceptualEncoder(
                self.vjepa_feature_dim, self.inner_dim
            )

        self.transformer = self.transformer.to(
            self.accelerator.device, dtype=weight_dtype
        )
        self.vae = self.vae.to(self.accelerator.device, dtype=weight_dtype)
        if self.vjepa is not None:
            self.vjepa = self.vjepa.to(self.accelerator.device, dtype=self.vjepa_dtype)
        if self.swinir is not None:
            self.swinir = self.swinir.to(self.accelerator.device)

        # Keep adapters in fp32 (LumaFlux-style); they cast back to block dtype.
        adapter_dtype = torch.float32
        self.blur_extractor = self.blur_extractor.to(
            self.accelerator.device, dtype=adapter_dtype
        )
        self.modulation = self.modulation.to(
            self.accelerator.device, dtype=adapter_dtype
        )
        self.pga = self.pga.to(self.accelerator.device, dtype=adapter_dtype)
        self.pcm = self.pcm.to(self.accelerator.device, dtype=adapter_dtype)
        self.coupler = self.coupler.to(self.accelerator.device, dtype=adapter_dtype)
        if self.jepa_connector is not None:
            self.jepa_connector = self.jepa_connector.to(
                self.accelerator.device, dtype=adapter_dtype
            )

        # Register trainable modules for accelerate / checkpointing.
        # The transformer ModuleList already owns the per-block PGA/PCM/Coupler
        # via wrappers, but we also keep top-level refs for the optimizer.
        self.models.append(self.blur_extractor)
        self.models.append(self.modulation)
        self.models.append(self.pga)
        self.models.append(self.pcm)
        self.models.append(self.coupler)
        if self.jepa_connector is not None:
            self.models.append(self.jepa_connector)

        for m in self.models:
            all_param = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            logger.info(
                "%s: trainable=%s / total=%s (%.1f%%)",
                m.__class__.__name__,
                f"{trainable:,}",
                f"{all_param:,}",
                100 * trainable / max(all_param, 1),
            )

    def _trainable_parameters(self):
        params = []
        for m in (
            self.blur_extractor,
            self.modulation,
            self.pga,
            self.pcm,
            self.coupler,
            self.jepa_connector,
        ):
            if m is None:
                continue
            params.extend(p for p in m.parameters() if p.requires_grad)
        return params

    def setup_optimizers(self):
        opt = self.opt.train.optim
        optimizer_type = getattr(opt, "type", "adamw").lower()
        trainable_params = self._trainable_parameters()

        if optimizer_type == "adafactor":
            from transformers import Adafactor

            optimizer = Adafactor(
                trainable_params,
                lr=opt.learning_rate,
                relative_step=getattr(opt, "relative_step", False),
                scale_parameter=getattr(opt, "scale_parameter", False),
                warmup_init=getattr(opt, "warmup_init", False),
            )
        else:
            use_8bit = getattr(opt, "use_8bit_adam", False)
            if use_8bit:
                try:
                    import bitsandbytes as bnb

                    optimizer_class = bnb.optim.AdamW8bit
                except ImportError:
                    optimizer_class = torch.optim.AdamW
            else:
                optimizer_class = torch.optim.AdamW
            optimizer = optimizer_class(
                trainable_params,
                lr=opt.learning_rate,
                betas=(opt.adam_beta1, opt.adam_beta2),
                weight_decay=opt.adam_weight_decay,
                eps=opt.adam_epsilon,
            )
        self.optimizers = [optimizer]

    def _run_swinir(self, lq_01: torch.Tensor) -> torch.Tensor:
        if self.swinir is None:
            return lq_01
        _, _, h, w = lq_01.shape
        window_size = int(getattr(self.swinir, "window_size", 1))
        unshuffle_scale = int(getattr(self.swinir, "unshuffle_scale", 1) or 1)
        required_multiple = max(1, window_size * unshuffle_scale)
        pad_h = (required_multiple - (h % required_multiple)) % required_multiple
        pad_w = (required_multiple - (w % required_multiple)) % required_multiple
        if pad_h or pad_w:
            lq_01 = F.pad(lq_01, (0, pad_w, 0, pad_h), mode="reflect")
        pre_01 = self.swinir(lq_01).detach().clamp(0.0, 1.0)
        return pre_01[:, :, :h, :w]

    def _ensure_null_embeddings(self):
        if (
            self.register_buffer_txt is not None
            and self.register_buffer_txt_ids is not None
        ):
            return
        self.logger.info("Generating null-text embeddings via Flux 2 text encoders...")
        pipe = self.flux2_pipe_cls.from_pretrained(
            self.opt.pretrained_model_name_or_path,
            transformer=None,
            vae=None,
            torch_dtype=self.weight_dtype,
        ).to(self.accelerator.device)
        with torch.no_grad():
            prompt_embeds, text_ids = pipe.encode_prompt(
                [PROMPT], num_images_per_prompt=1
            )
        self.register_buffer_txt = prompt_embeds.cpu()
        self.register_buffer_txt_ids = text_ids.cpu()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    def _prepare_text_inputs(self, batch_size: int, device: torch.device):
        self._ensure_null_embeddings()
        txt = (
            self.register_buffer_txt.to(device)
            .expand(batch_size, -1, -1)
            .to(self.weight_dtype)
        )
        txt_ids = self.register_buffer_txt_ids.to(device).to(self.weight_dtype)
        if txt_ids.ndim == 2:
            txt_ids = txt_ids.unsqueeze(0).expand(batch_size, -1, -1)
        elif txt_ids.shape[0] == 1 and batch_size > 1:
            txt_ids = txt_ids.expand(batch_size, -1, -1)
        return txt, txt_ids

    def _transformer_guidance(self, batch_size: int, device: torch.device):
        if not self.use_transformer_guidance:
            return None
        return torch.full(
            (batch_size,),
            self.guidance_scale,
            device=device,
            dtype=self.weight_dtype,
        )

    def _resolve_lq_01(
        self, gt_image: torch.Tensor, lq_image: torch.Tensor
    ) -> torch.Tensor:
        if self.vjepa_source_image == "gt":
            return torch.clamp((gt_image.float() + 1.0) / 2.0, 0.0, 1.0)
        lq_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
        if self.swinir is not None and self.vjepa_source_image == "pre":
            return self._run_swinir(lq_01)
        if self.swinir is not None and self.vjepa_source_image == "lq":
            return self._run_swinir(lq_01)
        return lq_01

    def _extract_jepa_tokens(self, lq_01: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            pixels = vjepa_from_unit_tensor(
                lq_01,
                size=self.vjepa_image_size,
                device=self.accelerator.device,
                out_dtype=self.vjepa_dtype,
            )
            return extract_vjepa_tokens(
                self.vjepa,
                pixels,
                device=self.accelerator.device,
                dtype=self.vjepa_dtype,
            )

    def set_conditioning(
        self,
        lq_01: torch.Tensor,
        t: torch.Tensor,
        lat_h: int,
        lat_w: int,
        jepa_tokens: torch.Tensor | None = None,
        num_cond_tokens: int = 0,
    ) -> None:
        """Populate BlurContext for the upcoming transformer forward."""
        # Flux2 encode_vae_image already patchifies 2x2 into channels, so the
        # packed sequence length is lat_h * lat_w (not (lat_h/2)*(lat_w/2)).
        tok_h, tok_w = int(lat_h), int(lat_w)
        phys = self.blur_extractor(lq_01)

        ctx = self.blur_context
        ctx.enabled = True
        ctx.t = t.float()
        ctx.blur_tokens = map_to_tokens(phys.blur_map, tok_h, tok_w).float()
        ctx.blur_pooled = phys.blur_map.mean(dim=[2, 3]).float()
        ctx.g = phys.g.float()
        ctx.r = phys.r.float()
        ctx.num_img_tokens = tok_h * tok_w
        ctx.num_cond_tokens = int(num_cond_tokens)

        if self.jepa_connector is not None:
            if jepa_tokens is None:
                jepa_tokens = self._extract_jepa_tokens(lq_01)
            # Connector is trainable; JEPA tokens themselves are detached.
            perc = self.jepa_connector(jepa_tokens.detach())
            ctx.jepa_tokens = tokens_to_grid(perc, tok_h, tok_w).float()
        else:
            # Zero perceptual track if JEPA disabled.
            b = lq_01.shape[0]
            ctx.jepa_tokens = torch.zeros(
                b,
                tok_h * tok_w,
                self.inner_dim,
                device=lq_01.device,
                dtype=torch.float32,
            )

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = (
            self.dataloader.dataset.gt_key
            if is_train
            else self.test_dataloader.dataset.gt_key
        )
        lq_key = (
            self.dataloader.dataset.lq_key
            if is_train
            else self.test_dataloader.dataset.lq_key
        )
        if is_train and self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.train)
        elif not is_train and self.opt.val.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.val)

    def optimize_parameters(self):
        self._ensure_null_embeddings()

        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key
        gt_image = self.sample[gt_key].clip(-1, 1)
        lq_image = self.sample[lq_key].clip(-1, 1)
        bsz = gt_image.shape[0]

        if gt_image.shape[1] == 1:
            gt_image = einops.repeat(gt_image, "b 1 h w -> b 3 h w")
        if lq_image.shape[1] == 1:
            lq_image = einops.repeat(lq_image, "b 1 h w -> b 3 h w")

        with torch.inference_mode():
            pixel_latents = encode_vae_image(
                self.vae,
                gt_image.to(self.weight_dtype),
                None,
                flux2_pipe_cls=self.flux2_pipe_cls,
            )
            if self.use_condition_image_for_transformer:
                image_latents, image_latent_ids = prepare_image_latents(
                    vae=self.vae,
                    ref_images=lq_image.to(self.weight_dtype),
                    batch_size=bsz,
                    generator=None,
                    device=self.accelerator.device,
                    dtype=self.weight_dtype,
                    flux2_pipe_cls=self.flux2_pipe_cls,
                )
            else:
                image_latents = None
                image_latent_ids = None

            lq_01 = self._resolve_lq_01(gt_image, lq_image)
            jepa_tokens = None
            if self.vjepa is not None:
                jepa_tokens = self._extract_jepa_tokens(lq_01)

        noise = torch.randn_like(pixel_latents)
        noisy_latents, timesteps, _ = get_noisy_model_input_and_timesteps(
            self.noise_scheduler_copy,
            pixel_latents,
            noise,
            pixel_latents.device,
            discrete_flow_shift=self.discrete_flow_shift,
        )

        packed = self.flux2_pipe_cls._pack_latents(noisy_latents)
        img_ids = self.flux2_pipe_cls._prepare_latent_ids(noisy_latents).to(
            device=self.accelerator.device, dtype=self.weight_dtype
        )
        latent_token_count = packed.shape[1]
        lat_h, lat_w = noisy_latents.shape[-2], noisy_latents.shape[-1]

        txt, txt_ids = self._prepare_text_inputs(bsz, self.accelerator.device)
        normalized_t = (timesteps.float() / 1000).to(self.weight_dtype)
        num_cond = int(image_latents.shape[1]) if image_latents is not None else 0
        self.set_conditioning(
            lq_01,
            normalized_t,
            lat_h,
            lat_w,
            jepa_tokens=jepa_tokens,
            num_cond_tokens=num_cond,
        )

        latent_model_input = packed
        latent_image_ids = img_ids
        if image_latents is not None:
            latent_model_input = torch.cat([packed, image_latents], dim=1)
            latent_image_ids = torch.cat([img_ids, image_latent_ids], dim=1)

        guidance = self._transformer_guidance(bsz, self.accelerator.device)
        model_pred = self.transformer(
            hidden_states=latent_model_input.to(self.weight_dtype),
            encoder_hidden_states=txt.to(self.weight_dtype),
            timestep=normalized_t,
            guidance=guidance,
            txt_ids=txt_ids.to(self.weight_dtype),
            img_ids=latent_image_ids.to(self.weight_dtype),
            return_dict=False,
        )[0]
        model_pred = model_pred[:, :latent_token_count]
        model_pred = self.flux2_pipe_cls._unpack_latents_with_ids(model_pred, img_ids)

        target = noise - pixel_latents
        loss = F.mse_loss(model_pred.float(), target.float())

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self._trainable_parameters(),
                getattr(self.opt.train.optim, "max_grad_norm", 1.0),
            )
            self.optimizers[0].step()
            self.optimizers[0].zero_grad()

        return {"all": loss}

    @torch.no_grad()
    def validation(self):
        gc.collect()
        torch.cuda.empty_cache()
        for model in self.models:
            model.eval()

        idx = 0
        for batch in tqdm(self.test_dataloader, desc="Validation"):
            idx = self.validate_step(
                batch,
                idx,
                self.test_dataloader.dataset.lq_key,
                self.test_dataloader.dataset.gt_key,
            )
            if idx >= self.max_val_steps:
                break

        gc.collect()
        torch.cuda.empty_cache()
        for model in self.models:
            model.train()

    @torch.no_grad()
    def forwardpass(
        self, lq_image: torch.Tensor, height: int, width: int, num_steps: int = 50
    ):
        self._ensure_null_embeddings()
        device = self.accelerator.device
        dtype = self.weight_dtype
        bsz = lq_image.shape[0]

        txt, txt_ids = self._prepare_text_inputs(bsz, device)

        latent_spatial_divisor = self.vae_scale_factor * 2
        lat_c = self.transformer.config.in_channels
        lat_h = int(height) // latent_spatial_divisor
        lat_w = int(width) // latent_spatial_divisor
        latents = torch.randn(bsz, lat_c, lat_h, lat_w, device=device, dtype=dtype)
        guidance = self._transformer_guidance(bsz, device)

        if self.use_condition_image_for_transformer:
            image_latents, image_latent_ids = prepare_image_latents(
                vae=self.vae,
                ref_images=lq_image.to(dtype),
                batch_size=bsz,
                generator=None,
                device=device,
                dtype=dtype,
                flux2_pipe_cls=self.flux2_pipe_cls,
            )
        else:
            image_latents = None
            image_latent_ids = None

        lq_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
        if self.swinir is not None:
            lq_01 = self._run_swinir(lq_01)
        jepa_tokens = None
        if self.vjepa is not None:
            jepa_tokens = self._extract_jepa_tokens(lq_01)

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.opt.pretrained_model_name_or_path, subfolder="scheduler"
        )
        if getattr(scheduler.config, "use_dynamic_shifting", False):
            image_seq_len = (lat_h // 2) * (lat_w // 2)
            mu = calculate_shift(
                image_seq_len,
                base_seq_len=getattr(scheduler.config, "base_image_seq_len", 256),
                max_seq_len=getattr(scheduler.config, "max_image_seq_len", 4096),
                base_shift=getattr(scheduler.config, "base_shift", 0.5),
                max_shift=getattr(scheduler.config, "max_shift", 1.15),
            )
            scheduler.set_timesteps(num_steps, device=device, mu=mu)
        else:
            scheduler.set_timesteps(num_steps, device=device)

        for t in tqdm(scheduler.timesteps):
            packed = self.flux2_pipe_cls._pack_latents(latents)
            img_ids = self.flux2_pipe_cls._prepare_latent_ids(latents).to(
                device=device, dtype=dtype
            )
            latent_token_count = packed.shape[1]
            timestep = t.unsqueeze(0).expand(bsz).to(device=device, dtype=dtype) / 1000.0

            num_cond = int(image_latents.shape[1]) if image_latents is not None else 0
            self.set_conditioning(
                lq_01,
                timestep,
                lat_h,
                lat_w,
                jepa_tokens=jepa_tokens,
                num_cond_tokens=num_cond,
            )

            latent_model_input = packed
            latent_image_ids = img_ids
            if image_latents is not None:
                latent_model_input = torch.cat([packed, image_latents], dim=1)
                latent_image_ids = torch.cat([img_ids, image_latent_ids], dim=1)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=txt.to(dtype),
                timestep=timestep,
                guidance=guidance,
                txt_ids=txt_ids.to(dtype),
                img_ids=latent_image_ids.to(dtype),
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, :latent_token_count]
            noise_pred = self.flux2_pipe_cls._unpack_latents_with_ids(
                noise_pred, img_ids
            )
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        decoded = decode_latents_flux(
            latents, self.vae, flux2_pipe_cls=self.flux2_pipe_cls
        )
        return decoded.clamp(-1, 1)

    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        num_inference_steps = getattr(self.opt.val, "num_inference_steps", 50)

        lq = self.sample[lq_key]
        gt = self.sample[gt_key]
        bsz = lq.shape[0]

        if lq.shape[1] == 1:
            lq = einops.repeat(lq, "b 1 h w -> b 3 h w")

        out = self.forwardpass(lq, lq.shape[-2], lq.shape[-1], num_inference_steps)

        lq_np = (lq.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        gt_np = (gt.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        out_np = (out.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)

        names = batch.get("name") if isinstance(batch, dict) else None

        for i in range(bsz):
            idx += 1
            images = [lq_np[i], gt_np[i], out_np[i]]
            for j in range(len(images)):
                if images[j].shape[0] == 1:
                    images[j] = einops.repeat(images[j], "1 h w -> 3 h w")
            images = np.stack(images)
            log_image(
                self.opt, self.accelerator, images, f"{idx:04d}", self.global_step
            )
            log_image(
                self.opt,
                self.accelerator,
                out_np[i : i + 1],
                f"out_{idx:04d}",
                self.global_step,
            )
            log_metrics(
                gt_np[i],
                out_np[i],
                self.opt.val.metrics,
                self.accelerator,
                self.global_step,
            )

            # Under -infer, also dump a lossless PNG named after the real sample so a
            # metrics script can match predictions back to GT by filename instead of by
            # position in a (possibly truncated-by-max_val_steps) directory listing.
            if not self.opt.is_train:
                stem = (
                    os.path.splitext(os.path.basename(names[i]))[0]
                    if names is not None
                    else f"{idx:05d}"
                )
                save_eval_png(self.opt.path.experiments_root, "eval_preds", stem, out_np[i])
        return idx
