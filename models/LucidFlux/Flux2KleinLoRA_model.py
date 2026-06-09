import copy
import gc

import einops
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from torch.utils.data import DataLoader, Subset
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
from .src.flux.lucidflux import load_precomputed_embeddings

try:
    from diffusers import AutoencoderKLFlux2
except ImportError:
    from diffusers import AutoencoderKL as AutoencoderKLFlux2


class Flux2KleinLoRA_model(BaseModel):
    """Flux2 Klein LoRA trainer without JEPA conditioning."""

    def __init__(self, opt, logger):
        super(Flux2KleinLoRA_model, self).__init__(opt, logger)

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        pretrained = str(opt.pretrained_model_name_or_path)
        if "klein" not in pretrained.lower():
            logger.warning(
                "Flux2KleinLoRA_model is intended for Flux2 Klein backbones, "
                f"got: {pretrained}"
            )

        self.flux2_pipe_cls = get_flux2_pipeline_class(pretrained)
        self.guidance_scale = float(getattr(opt, "guidance_scale", 1.0))
        self.discrete_flow_shift = float(getattr(opt, "discrete_flow_shift", 3.1582))
        revision = getattr(opt, "revision", None)
        variant = getattr(opt, "variant", None)

        base_transformer = Flux2Transformer2DModel.from_pretrained(
            pretrained,
            subfolder="transformer",
            revision=revision,
            variant=variant,
        )
        base_transformer.requires_grad_(False)
        if getattr(opt, "transformer_gradient_checkpointing", False):
            if hasattr(base_transformer, "enable_gradient_checkpointing"):
                base_transformer.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled for LoRA transformer")

        lora_rank = int(getattr(opt.train, "lora_rank", 8))
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
            param.requires_grad = "lora" in name
        logger.info(
            f"Enabled Flux2 LoRA: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}"
        )

        self.vae = AutoencoderKLFlux2.from_pretrained(
            pretrained,
            subfolder="vae",
            revision=revision,
            variant=variant,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained,
            subfolder="scheduler",
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        self.use_transformer_guidance = getattr(self.transformer.config, "guidance_embeds", True)

        embeddings_path = getattr(opt, "precomputed_embeddings_path", None)
        if embeddings_path:
            embeds = load_precomputed_embeddings(embeddings_path, "cpu")
            self.register_buffer_txt = embeds["txt"]
            self.register_buffer_txt_ids = embeds["txt_ids"]
        else:
            self.register_buffer_txt = None
            self.register_buffer_txt_ids = None

        self.transformer = self.transformer.to(self.accelerator.device, dtype=weight_dtype)
        self.vae = self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.models.append(self.transformer)

        for m in self.models:
            all_param = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            logger.info(
                f"{m.__class__.__name__}: trainable={trainable:,} / total={all_param:,} "
                f"({100 * trainable / all_param:.3f}%)"
            )

    def _ensure_null_embeddings(self):
        if self.register_buffer_txt is not None and self.register_buffer_txt_ids is not None:
            return
        self.logger.info("Generating null-text embeddings from Flux2 text encoders")
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
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

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

    def optimize_parameters(self):
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
            image_latents, image_latent_ids = prepare_image_latents(
                vae=self.vae,
                ref_images=lq_image.to(self.weight_dtype),
                batch_size=bsz,
                generator=None,
                device=self.accelerator.device,
                dtype=self.weight_dtype,
                flux2_pipe_cls=self.flux2_pipe_cls,
            )

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
            device=self.accelerator.device,
            dtype=self.weight_dtype,
        )
        latent_token_count = packed_transformer_input.shape[1]

        txt, txt_ids = self._prepare_text_inputs(bsz, self.accelerator.device)
        transformer_guidance = self._transformer_guidance(bsz, self.accelerator.device)
        normalized_timesteps = (timesteps.float() / 1000).to(self.weight_dtype)

        latent_model_input = torch.cat([packed_transformer_input, image_latents], dim=1)
        latent_image_ids = torch.cat([transformer_image_ids, image_latent_ids], dim=1)

        model_pred = self.transformer(
            hidden_states=latent_model_input.to(self.weight_dtype),
            encoder_hidden_states=txt.to(self.weight_dtype),
            timestep=normalized_timesteps,
            guidance=transformer_guidance,
            txt_ids=txt_ids.to(self.weight_dtype),
            img_ids=latent_image_ids.to(self.weight_dtype),
            return_dict=False,
        )[0]
        model_pred = model_pred[:, :latent_token_count]
        model_pred = self.flux2_pipe_cls._unpack_latents_with_ids(model_pred, transformer_image_ids)

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
        idx=0
        for model in self.models:
            model.eval()

        dataloader = DataLoader(Subset(self.dataloader.dataset, np.arange(3)), 
                                shuffle=False, 
                                batch_size=1)
        print(f"Tesing using {len(dataloader)} training data...")
        dataloader = self.accelerator.prepare(dataloader)
        for batch in dataloader:
            idx = self.validate_step(batch, idx, self.dataloader.dataset.lq_key, self.dataloader.dataset.gt_key)
        self.accelerator._dataloaders.remove(dataloader)

        print(f"Tesing using {len(self.test_dataloader)} testing data...")
        for batch in tqdm(self.test_dataloader):
            idx = self.validate_step(batch, idx, self.test_dataloader.dataset.lq_key, self.test_dataloader.dataset.gt_key)
            if idx > self.max_val_steps:
                break

        gc.collect()
        torch.cuda.empty_cache()
        for model in self.models:
            model.train()

    @torch.no_grad()
    def forwardpass(self, lq_image: torch.Tensor, height: int, width: int, num_steps: int = 50):
        device = self.accelerator.device
        dtype = self.weight_dtype
        bsz = lq_image.shape[0]

        txt, txt_ids = self._prepare_text_inputs(bsz, device)
        transformer_guidance = self._transformer_guidance(bsz, device)

        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        latent_spatial_divisor = vae_scale_factor * 2
        lat_c = self.transformer.config.in_channels
        lat_h = int(height) // latent_spatial_divisor
        lat_w = int(width) // latent_spatial_divisor

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

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.opt.pretrained_model_name_or_path,
            subfolder="scheduler",
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

        for t in scheduler.timesteps:
            packed_transformer_input = self.flux2_pipe_cls._pack_latents(latents)
            transformer_image_ids = self.flux2_pipe_cls._prepare_latent_ids(latents).to(
                device=device, dtype=dtype
            )
            latent_token_count = packed_transformer_input.shape[1]
            timestep = t.unsqueeze(0).expand(bsz).to(device=device, dtype=dtype) / 1000.0

            latent_model_input = torch.cat([packed_transformer_input, image_latents], dim=1)
            latent_image_ids = torch.cat([transformer_image_ids, image_latent_ids], dim=1)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=txt.to(dtype),
                timestep=timestep,
                guidance=transformer_guidance,
                txt_ids=txt_ids.to(dtype),
                img_ids=latent_image_ids.to(dtype),
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, :latent_token_count]
            noise_pred = self.flux2_pipe_cls._unpack_latents_with_ids(noise_pred, transformer_image_ids)
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        decoded = decode_latents_flux(latents, self.vae, flux2_pipe_cls=self.flux2_pipe_cls)
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
        # breakpoint()
        for i in range(bsz):
            idx += 1
            images = [lq_np[i], gt_np[i], out_np[i]]
            for j in range(len(images)):
                if images[j].shape[0] == 1:
                    images[j] = einops.repeat(images[j], "1 h w -> 3 h w")
            images = np.hstack(images)[None]
            images = np.clip(images, 0, 1)
            log_image(self.opt, self.accelerator, images, f"{idx:04d}", self.global_step)
            log_image(self.opt, self.accelerator, out_np[i : i + 1], f"out_{idx:04d}", self.global_step)
            log_metrics(gt_np[i], out_np[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
