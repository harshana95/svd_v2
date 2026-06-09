import copy
import gc
import os

import einops
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from tqdm import tqdm

from models.base_model import BaseModel
from utils import log_image, log_metrics

from .helpers import (
    calculate_shift,
    decode_latents_flux,
    encode_vae_image,
    get_flux2_pipeline_class,
    get_noisy_model_input_and_timesteps,
    initialize_transformer_lora,
    prepare_image_latents,
)
from .jepa_helpers import (
    extract_vjepa_tokens,
    get_vjepa_feature_dim,
    load_vjepa2_model,
    load_vjepa2_weights,
    vjepa_from_unit_tensor,
)
from .src.flux.lucidflux import (
    load_precomputed_embeddings,
    load_swinir,
)

try:
    from diffusers import AutoencoderKLFlux2
except ImportError:
    from diffusers import AutoencoderKL as AutoencoderKLFlux2

class LucidFlux2LoRAJEPA_model(BaseModel):
    def __init__(self, opt, logger):
        super(LucidFlux2LoRAJEPA_model, self).__init__(opt, logger)

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
        self.vjepa_dtype = getattr(opt, "vjepa_dtype", torch.float32)

        pretrained = opt.pretrained_model_name_or_path
        self.flux2_pipe_cls = get_flux2_pipeline_class(pretrained)
        transformer_config = Flux2Transformer2DModel.load_config(pretrained, subfolder="transformer")
        text_in_dim = opt.get("text_in_dim")
        if text_in_dim is None:
            text_in_dim = transformer_config.get("joint_attention_dim")
        if text_in_dim is None:
            text_in_dim = 15360

        self.vjepa_hub_entry = getattr(opt, "vjepa_hub_entry", "vjepa2_1_vit_large_384")
        self.vjepa_image_size = getattr(opt, "vjepa_image_size", 384)
        self.vjepa_pretrained = getattr(opt, "vjepa_pretrained", True)
        self.vjepa_checkpoint_path = getattr(opt, "vjepa_checkpoint_path", None)

        self.vjepa_feature_dim = get_vjepa_feature_dim(
            self.vjepa_hub_entry,
            fallback_dim=getattr(opt, "vjepa_feature_dim", 1024),
        )

        logger.info(f"Loading V-JEPA 2.1 encoder from torch.hub entry {self.vjepa_hub_entry}")
        self.vjepa = load_vjepa2_model(
            self.vjepa_hub_entry,
            "cpu",
            self.vjepa_dtype,
            offload=False,
            pretrained=self.vjepa_pretrained,
        )
        if self.vjepa_checkpoint_path and os.path.exists(self.vjepa_checkpoint_path):
            logger.info(f"Loading V-JEPA checkpoint from {self.vjepa_checkpoint_path}")
            load_vjepa2_weights(self.vjepa, self.vjepa_checkpoint_path)
        self.vjepa.requires_grad_(False)
        self.vjepa.eval()

        swinir_path = getattr(opt, "swinir_model_path", None)
        self.swinir = None
        if swinir_path and os.path.exists(swinir_path):
            logger.info(f"Loading SwinIR from {swinir_path}")
            self.swinir = load_swinir("cpu", swinir_path, offload=False)

        embeddings_path = getattr(opt, "precomputed_embeddings_path", None)
        if embeddings_path and os.path.exists(embeddings_path):
            embeds = load_precomputed_embeddings(embeddings_path, "cpu")
            self.register_buffer_txt = embeds["txt"]
            self.register_buffer_vec = embeds.get("vec")
            self.register_buffer_txt_ids = embeds.get("txt_ids")
        else:
            logger.warning(
                "No precomputed embeddings path found. Will generate null embeddings at first forward pass."
            )
            self.register_buffer_txt = None
            self.register_buffer_vec = None
            self.register_buffer_txt_ids = None

        self.guidance_scale = getattr(opt, "guidance_scale", 4.0)
        self.discrete_flow_shift = getattr(opt, "discrete_flow_shift", 3.1582)
        revision = getattr(opt, "revision", None)
        variant = getattr(opt, "variant", None)

        base_transformer = Flux2Transformer2DModel.from_pretrained(
            pretrained, subfolder="transformer", revision=revision, variant=variant
        )
        base_transformer.requires_grad_(False)
        if getattr(opt, "transformer_gradient_checkpointing", False):
            if hasattr(base_transformer, "enable_gradient_checkpointing"):
                base_transformer.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled for LoRA transformer")

        lora_rank = int(getattr(opt.train, "lora_rank", 4))
        lora_alpha = int(getattr(opt.train, "lora_alpha", max(1, lora_rank * 2)))
        lora_dropout = float(getattr(opt.train, "lora_dropout", 0.0))
        lora_target_modules = getattr(opt.train, "lora_target_modules", None)

        self.transformer = initialize_transformer_lora(
            base_transformer,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_patterns=lora_target_modules,
        )
        for name, param in self.transformer.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        logger.info(
            f"Enabled Flux2 transformer LoRA: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}"
        )

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
        self.transformer_context_dim = text_in_dim
        self.use_transformer_guidance = getattr(
            self.transformer.config, "guidance_embeds", True
        )

        self.transformer = self.transformer.to(self.accelerator.device, dtype=weight_dtype)
        self.vae = self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.vjepa = self.vjepa.to(self.accelerator.device, dtype=self.vjepa_dtype)
        if self.swinir is not None:
            self.swinir = self.swinir.to(self.accelerator.device)

        self.models.append(self.transformer)

        for m in self.models:
            all_param = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            logger.info(
                f"{m.__class__.__name__}: trainable={trainable:,} / total={all_param:,} "
                f"({100 * trainable / all_param:.3f}%)"
            )

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
        if self.register_buffer_txt is not None and self.register_buffer_txt_ids is not None:
            return
        self.logger.info("Generating null-text embeddings via Flux 2 text encoders...")
        pipe = self.flux2_pipe_cls.from_pretrained(
            self.opt.pretrained_model_name_or_path,
            transformer=None,
            vae=None,
            torch_dtype=self.weight_dtype,
        ).to(self.accelerator.device)
        with torch.no_grad():
            prompt_embeds, text_ids = pipe.encode_prompt([""], num_images_per_prompt=1)

        self.register_buffer_txt = prompt_embeds.cpu()
        self.register_buffer_txt_ids = text_ids.cpu()
        self.register_buffer_vec = None
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    def setup_optimizers(self):
        opt = self.opt.train.optim
        optimizer_type = getattr(opt, "type", "adamw").lower()
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]

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
                weight_decay=getattr(opt, "adam_weight_decay", 0.01),
                eps=getattr(opt, "adam_epsilon", 1e-8),
            )
        self.optimizers.append(optimizer)

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        if is_train and self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.train)
        elif not is_train and self.opt.val.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.val)

    def _extract_vision_features(self, pre_01: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            vjepa_pixels = vjepa_from_unit_tensor(
                pre_01,
                size=self.vjepa_image_size,
                device=self.accelerator.device,
                out_dtype=self.vjepa_dtype,
            )
            return extract_vjepa_tokens(
                self.vjepa,
                vjepa_pixels,
                device=self.accelerator.device,
                dtype=self.vjepa_dtype,
            )

    def _prepare_text_inputs(self, batch_size: int, device: torch.device):
        self._ensure_null_embeddings()
        txt = self.register_buffer_txt.to(device).expand(batch_size, -1, -1).to(self.weight_dtype)
        txt_ids = self.register_buffer_txt_ids.to(device).to(self.weight_dtype)
        if txt_ids.ndim == 2:
            txt_ids = txt_ids.unsqueeze(0).expand(batch_size, -1, -1)
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

    def _project_vision_tokens(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        vision_tokens = vision_tokens.to(self.weight_dtype)
        feature_dim = vision_tokens.shape[-1]
        if feature_dim == self.transformer_context_dim:
            return vision_tokens
        if feature_dim > self.transformer_context_dim:
            return vision_tokens[:, :, : self.transformer_context_dim]

        pad_dim = self.transformer_context_dim - feature_dim
        zeros = torch.zeros(
            vision_tokens.shape[0],
            vision_tokens.shape[1],
            pad_dim,
            device=vision_tokens.device,
            dtype=vision_tokens.dtype,
        )
        return torch.cat([vision_tokens, zeros], dim=-1)

    def _build_encoder_states(
        self,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        vision_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vision_tokens = self._project_vision_tokens(vision_tokens)
        encoder_hidden_states = torch.cat([txt, vision_tokens], dim=1)

        vision_txt_ids = torch.zeros(
            txt_ids.shape[0],
            vision_tokens.shape[1],
            txt_ids.shape[-1],
            device=txt_ids.device,
            dtype=txt_ids.dtype,
        )
        encoder_txt_ids = torch.cat([txt_ids, vision_txt_ids], dim=1)
        return encoder_hidden_states, encoder_txt_ids

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
                self.vae, gt_image.to(self.weight_dtype), None, flux2_pipe_cls=self.flux2_pipe_cls
            )
            image_latents, image_latent_ids = prepare_image_latents(
                vae=self.vae,
                ref_images=lq_image.to(self.weight_dtype),
                batch_size=bsz,
                generator=None,
                device=self.accelerator.device,
                dtype=self.weight_dtype,
                flux2_pipe_cls=self.flux2_pipe_cls,
            )
            if self.swinir is not None:
                lq_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
                pre_01 = self._run_swinir(lq_01)
            else:
                pre_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
            vision_tokens = self._extract_vision_features(pre_01)

        noise = torch.randn_like(pixel_latents)
        noisy_latents, timesteps, _ = get_noisy_model_input_and_timesteps(
            self.noise_scheduler_copy,
            pixel_latents,
            noise,
            pixel_latents.device,
            discrete_flow_shift=self.discrete_flow_shift,
        )

        packed_transformer_input = self.flux2_pipe_cls._pack_latents(noisy_latents)
        transformer_image_ids = self.flux2_pipe_cls._prepare_latent_ids(noisy_latents).to(
            device=self.accelerator.device, dtype=self.weight_dtype
        )
        latent_token_count = packed_transformer_input.shape[1]

        txt, txt_ids = self._prepare_text_inputs(bsz, self.accelerator.device)
        encoder_hidden_states, encoder_txt_ids = self._build_encoder_states(
            txt, txt_ids, vision_tokens
        )
        transformer_guidance = self._transformer_guidance(bsz, self.accelerator.device)
        normalized_timesteps = (timesteps.float() / 1000).to(self.weight_dtype)

        latent_model_input = packed_transformer_input
        latent_image_ids = transformer_image_ids
        if image_latents is not None:
            latent_model_input = torch.cat([packed_transformer_input, image_latents], dim=1)
            latent_image_ids = torch.cat([transformer_image_ids, image_latent_ids], dim=1)

        model_pred = self.transformer(
            hidden_states=latent_model_input.to(self.weight_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weight_dtype),
            timestep=normalized_timesteps,
            guidance=transformer_guidance,
            txt_ids=encoder_txt_ids.to(self.weight_dtype),
            img_ids=latent_image_ids.to(self.weight_dtype),
            return_dict=False,
        )[0]
        model_pred = model_pred[:, :latent_token_count]
        model_pred = self.flux2_pipe_cls._unpack_latents_with_ids(
            model_pred,
            transformer_image_ids,
        )

        target = noise - pixel_latents
        loss = F.mse_loss(model_pred.float(), target.float())

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                [p for p in self.transformer.parameters() if p.requires_grad],
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
    def forwardpass(self, lq_image: torch.Tensor, height: int, width: int, num_steps: int = 50):
        self._ensure_null_embeddings()
        device = self.accelerator.device
        dtype = self.weight_dtype
        bsz = lq_image.shape[0]

        if self.swinir is not None:
            lq_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
            pre_01 = self._run_swinir(lq_01)
        else:
            pre_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)

        txt, txt_ids = self._prepare_text_inputs(bsz, device)

        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        latent_spatial_divisor = vae_scale_factor * 2
        lat_c = self.transformer.config.in_channels
        lat_h = int(height) // latent_spatial_divisor
        lat_w = int(width) // latent_spatial_divisor

        transformer_guidance = self._transformer_guidance(bsz, device)
        with torch.inference_mode():
            image_latents, image_latent_ids = prepare_image_latents(
                vae=self.vae,
                ref_images=lq_image.to(dtype),
                batch_size=bsz,
                generator=None,
                device=device,
                dtype=dtype,
                flux2_pipe_cls=self.flux2_pipe_cls,
            )
            vision_tokens = self._extract_vision_features(pre_01)
            encoder_hidden_states, encoder_txt_ids = self._build_encoder_states(
                txt, txt_ids, vision_tokens
            )

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

        latents = torch.randn(bsz, lat_c, lat_h, lat_w, device=device, dtype=dtype)

        for t in tqdm(scheduler.timesteps):
            packed_transformer_input = self.flux2_pipe_cls._pack_latents(latents)
            transformer_image_ids = self.flux2_pipe_cls._prepare_latent_ids(latents).to(
                device=device, dtype=dtype
            )
            latent_token_count = packed_transformer_input.shape[1]

            timestep = t.unsqueeze(0).expand(bsz).to(device=device, dtype=dtype) / 1000.0
            latent_model_input = packed_transformer_input
            latent_image_ids = transformer_image_ids
            if image_latents is not None:
                latent_model_input = torch.cat([packed_transformer_input, image_latents], dim=1)
                latent_image_ids = torch.cat([transformer_image_ids, image_latent_ids], dim=1)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                timestep=timestep,
                guidance=transformer_guidance,
                txt_ids=encoder_txt_ids.to(dtype),
                img_ids=latent_image_ids.to(dtype),
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, :latent_token_count]
            noise_pred = self.flux2_pipe_cls._unpack_latents_with_ids(
                noise_pred,
                transformer_image_ids,
            )
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        decoded = decode_latents_flux(latents, self.vae, flux2_pipe_cls=self.flux2_pipe_cls)
        return decoded.clamp(-1, 1)

    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        num_inference_steps = getattr(self.opt.val, "num_inference_steps", 50)

        lq = self.sample[lq_key]
        gt = self.sample[gt_key]
        bsz = lq.shape[0]
        swinir_np = None

        if lq.shape[1] == 1:
            lq = einops.repeat(lq, "b 1 h w -> b 3 h w")

        out = self.forwardpass(lq, lq.shape[-2], lq.shape[-1], num_inference_steps)

        if self.swinir is not None:
            with torch.inference_mode():
                lq_01 = torch.clamp((lq.float() + 1.0) / 2.0, 0.0, 1.0)
                pre_01 = self._run_swinir(lq_01)
            swinir_np = pre_01.cpu().float().numpy().clip(0, 1)

        lq_np = (lq.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        gt_np = (gt.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        out_np = (out.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)

        for i in range(bsz):
            idx += 1
            images = [lq_np[i], gt_np[i], out_np[i]]
            for j in range(len(images)):
                if images[j].shape[0] == 1:
                    images[j] = einops.repeat(images[j], "1 h w -> 3 h w")
            images = np.stack(images)
            images = np.clip(images, 0, 1)
            log_image(self.opt, self.accelerator, images, f"{idx:04d}", self.global_step)
            log_image(self.opt, self.accelerator, out_np[i : i + 1], f"out_{idx:04d}", self.global_step)
            if swinir_np is not None:
                log_image(
                    self.opt,
                    self.accelerator,
                    swinir_np[i : i + 1],
                    f"swinir_{idx:04d}",
                    self.global_step,
                )
            log_metrics(gt_np[i], out_np[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
