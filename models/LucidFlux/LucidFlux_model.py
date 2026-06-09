# LucidFlux model integration for the training framework.
# Caption-free universal image restoration via dual-branch conditioner on Flux.1-dev.
# Reference: W2GenAI-Lab/LucidFlux (Apache 2.0)

import copy
import gc
import math
import os

import einops
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from tqdm import tqdm

from models.base_model import BaseModel
from utils import log_image, log_metrics

from .src.flux.condition import (
    FluxParams,
    ConditionBranchWithRedux,
    build_condition_branch_fresh,
    export_lucidflux_state,
    load_condition_branch_with_redux,
)
from .src.flux.lucidflux import (
    load_lucidflux_weights,
    load_precomputed_embeddings,
    load_siglip_model,
    load_swinir,
    move_modules_to_device,
    prepare_with_embeddings,
    siglip_from_unit_tensor,
)
from .src.flux.sampling import denoise_lucidflux, get_noise, get_schedule, unpack


# ── Flux latent encoding ────────────────────────────────────────────────────

def encode_images_flux(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    shift = getattr(vae.config, "shift_factor", 0.0)
    scale = getattr(vae.config, "scaling_factor", vae.config.scaling_factor)
    pixel_latents = (pixel_latents - shift) * scale
    return pixel_latents.to(weight_dtype)


def decode_latents_flux(latents: torch.Tensor, vae: torch.nn.Module):
    shift = getattr(vae.config, "shift_factor", 0.0)
    scale = getattr(vae.config, "scaling_factor", vae.config.scaling_factor)
    latents = latents / scale + shift
    image = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
    return image


# ── Timestep sampling (LucidFlux sigmoid schedule) ──────────────────────────

def get_noisy_model_input_and_timesteps(
    noise_scheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
    discrete_flow_shift: float = 3.1582,
):
    batch_size = latents.shape[0]
    num_timesteps = noise_scheduler.config.num_train_timesteps

    sigmas = torch.sigmoid(
        torch.randn((batch_size,), device=device) + discrete_flow_shift
    )
    timesteps = (sigmas * num_timesteps).long()
    sigmas = sigmas.view(-1, 1, 1)
    noisy_model_input = (1 - sigmas) * latents + sigmas * noise
    return noisy_model_input, timesteps, sigmas.squeeze()


# ── Packing / unpacking utilities ────────────────────────────────────────────

def pack_latents(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def unpack_latents(
    x: torch.Tensor, packed_latent_height: int, packed_latent_width: int
) -> torch.Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=packed_latent_height,
        w=packed_latent_width,
        ph=2,
        pw=2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LucidFlux_model
# ═══════════════════════════════════════════════════════════════════════════════

class LucidFlux_model(BaseModel):

    def __init__(self, opt, logger):
        super(LucidFlux_model, self).__init__(opt, logger)

        # ── dtype ────────────────────────────────────────────────────────
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        # ── Config ───────────────────────────────────────────────────────
        self.guidance_scale = getattr(opt, "guidance_scale", 4.0)
        self.discrete_flow_shift = getattr(opt, "discrete_flow_shift", 3.1582)
        pretrained = opt.pretrained_model_name_or_path
        revision = getattr(opt, "revision", None)
        variant = getattr(opt, "variant", None)

        # ── Frozen backbone: Flux transformer + VAE ──────────────────────
        self.transformer = FluxTransformer2DModel.from_pretrained(
            pretrained, subfolder="transformer", revision=revision, variant=variant
        )
        self.transformer.requires_grad_(False)
        self.transformer.eval()
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing enabled for frozen transformer")

        self.vae = AutoencoderKL.from_pretrained(
            pretrained, subfolder="vae", revision=revision, variant=variant
        )
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained, subfolder="scheduler"
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

        # ── Trainable condition branch ───────────────────────────────────
        flux_params = FluxParams()
        lucidflux_weights_path = getattr(opt, "lucidflux_weights_path", None)
        if lucidflux_weights_path and os.path.exists(lucidflux_weights_path):
            logger.info(f"Loading pretrained LucidFlux weights from {lucidflux_weights_path}")
            lucidflux_weights = load_lucidflux_weights(lucidflux_weights_path)
            self.condition_branch = load_condition_branch_with_redux(
                params=flux_params,
                lucidflux_weights=lucidflux_weights,
                device="cpu",
                offload=False,
                branch_dtype=torch.float32,
                connector_dtype=torch.float32,
            )
        else:
            logger.info("Building fresh LucidFlux condition branch (no pretrained weights)")
            self.condition_branch = build_condition_branch_fresh(
                params=flux_params,
                device="cpu",
                branch_dtype=torch.float32,
                connector_dtype=torch.float32,
            )
        self.condition_branch.train()
        self.condition_branch.requires_grad_(True)

        # Enable gradient checkpointing on condition branch sub-modules
        for module in [
            self.condition_branch.condition_branch.lq,
            self.condition_branch.condition_branch.pre,
        ]:
            if hasattr(module, "_set_gradient_checkpointing"):
                module._set_gradient_checkpointing(module, True)

        # ── Frozen auxiliary: SwinIR + SigLIP ────────────────────────────
        swinir_path = getattr(opt, "swinir_model_path", None)
        self.swinir = None
        if swinir_path and os.path.exists(swinir_path):
            logger.info(f"Loading SwinIR from {swinir_path}")
            self.swinir = load_swinir("cpu", swinir_path, offload=False)

        siglip_path = getattr(opt, "siglip_model_path", None)
        self.siglip = None
        siglip_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.siglip_dtype = siglip_dtype
        if siglip_path:
            logger.info(f"Loading SigLIP from {siglip_path}")
            self.siglip = load_siglip_model(siglip_path, "cpu", siglip_dtype, offload=False)

        # ── Precomputed null-text embeddings ─────────────────────────────
        embeddings_path = getattr(opt, "precomputed_embeddings_path", None)
        if embeddings_path and os.path.exists(embeddings_path):
            embeds = load_precomputed_embeddings(embeddings_path, "cpu")
            self.register_buffer_txt = embeds["txt"]
            self.register_buffer_vec = embeds["vec"]
        else:
            # Generate null embeddings on the fly using T5/CLIP from the Flux pipeline
            logger.warning(
                "No precomputed embeddings path found. Will generate null embeddings at first forward pass."
            )
            self.register_buffer_txt = None
            self.register_buffer_vec = None

        # ── Move to device ───────────────────────────────────────────────
        self.transformer = self.transformer.to(self.accelerator.device, dtype=weight_dtype)
        self.vae = self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.condition_branch = self.condition_branch.to(self.accelerator.device)

        if self.swinir is not None:
            self.swinir = self.swinir.to(self.accelerator.device)
        if self.siglip is not None:
            self.siglip = self.siglip.to(self.accelerator.device)

        # Only condition_branch goes into self.models (for accelerator.prepare)
        self.models.append(self.condition_branch)

        # Print parameter counts
        for m in self.models:
            all_param = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            logger.info(
                f"{m.__class__.__name__}: trainable={trainable:,} / total={all_param:,} "
                f"({100 * trainable / all_param:.1f}%)"
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

    # ── Null embedding generation (lazy, first call only) ────────────────

    def _ensure_null_embeddings(self):
        if self.register_buffer_txt is not None:
            return
        self.logger.info("Generating null-text embeddings via Flux text encoders...")
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            self.opt.pretrained_model_name_or_path,
            transformer=None, vae=None,
            torch_dtype=self.weight_dtype,
        ).to(self.accelerator.device)
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                [""], prompt_2=None
            )
        self.register_buffer_txt = prompt_embeds.cpu()
        self.register_buffer_vec = pooled_prompt_embeds.cpu()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        # Save for next time
        save_path = os.path.join(
            self.opt.path.get("root", "."), "null_prompt_embeddings.pt"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {"txt": self.register_buffer_txt, "vec": self.register_buffer_vec, "prompt": ""},
            save_path,
        )
        self.logger.info(f"Saved null embeddings to {save_path}")

    # ── Optimizer ────────────────────────────────────────────────────────

    def setup_optimizers(self):
        opt = self.opt.train.optim
        optimizer_type = getattr(opt, "type", "adamw").lower()

        trainable_params = [
            p for p in self.condition_branch.parameters() if p.requires_grad
        ]

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

    # ── Data ─────────────────────────────────────────────────────────────

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        if is_train and self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.train)
        elif not is_train and self.opt.val.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.val)

    # ── Training step ────────────────────────────────────────────────────

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

        # ── Encode GT to latents ─────────────────────────────────────────
        with torch.no_grad():
            pixel_latents = encode_images_flux(gt_image, self.vae, self.weight_dtype)

        # ── Flow matching noise schedule ─────────────────────────────────
        noise = torch.randn_like(pixel_latents)
        noisy_latents, timesteps, sigmas = get_noisy_model_input_and_timesteps(
            self.noise_scheduler_copy,
            pixel_latents,
            noise,
            pixel_latents.device,
            discrete_flow_shift=self.discrete_flow_shift,
        )

        _, _, lat_h, lat_w = pixel_latents.shape
        packed_latent_h = lat_h // 2
        packed_latent_w = lat_w // 2

        # ── Prepare packed inputs ────────────────────────────────────────
        with torch.no_grad():
            model_inputs = prepare_with_embeddings(
                img=noisy_latents,
                precomputed_txt=self.register_buffer_txt.to(self.accelerator.device),
                precomputed_vec=self.register_buffer_vec.to(self.accelerator.device),
            )

        # ── Prepare conditioning ─────────────────────────────────────────
        condition_cond_lq = lq_image.to(self.weight_dtype)
        with torch.no_grad():
            if self.swinir is not None:
                lq_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
                pre_01 = self._run_swinir(lq_01)
                condition_cond_pre = (pre_01 * 2.0 - 1.0).to(self.weight_dtype)
            else:
                pre_01 = None
                condition_cond_pre = condition_cond_lq

        # ── SigLIP features ──────────────────────────────────────────────
        siglip_image_pre_fts = None
        if self.siglip is not None and self.swinir is not None and pre_01 is not None:
            with torch.no_grad():
                siglip_size = getattr(
                    getattr(self.siglip, "config", None), "image_size", 512
                )
                siglip_pixels = siglip_from_unit_tensor(
                    pre_01, size=(siglip_size, siglip_size)
                )
                siglip_image_pre_fts = self.siglip(
                    pixel_values=siglip_pixels.to(
                        device=self.accelerator.device, dtype=self.siglip_dtype
                    )
                ).last_hidden_state

        # ── Guidance ─────────────────────────────────────────────────────
        guidance_vec = torch.full(
            (bsz,), self.guidance_scale,
            device=self.accelerator.device, dtype=self.weight_dtype,
        )
        normalized_timesteps = (timesteps.float() / 1000).to(self.weight_dtype)

        # ── Condition branch forward (trainable) ─────────────────────────
        block_res_samples, siglip_txt, siglip_txt_ids = self.condition_branch(
            img=model_inputs["img"].to(self.weight_dtype),
            img_ids=model_inputs["img_ids"].to(self.weight_dtype),
            condition_cond_lq=condition_cond_lq,
            condition_cond_pre=condition_cond_pre,
            txt=model_inputs["txt"].to(self.weight_dtype),
            txt_ids=model_inputs["txt_ids"].to(self.weight_dtype),
            y=model_inputs["vec"].to(self.weight_dtype),
            timesteps=normalized_timesteps,
            guidance=guidance_vec,
            siglip_image_pre_fts=siglip_image_pre_fts,
        )

        # ── Frozen transformer forward (gradients flow through block_res_samples) ──
        model_pred = self.transformer(
            hidden_states=model_inputs["img"].to(self.weight_dtype),
            encoder_hidden_states=siglip_txt.to(self.weight_dtype),
            pooled_projections=model_inputs["vec"].to(self.weight_dtype),
            timestep=normalized_timesteps,
            guidance=guidance_vec,
            txt_ids=siglip_txt_ids.to(self.weight_dtype),
            img_ids=model_inputs["img_ids"][0].to(self.weight_dtype), # no need for batch dim
            controlnet_block_samples=[
                s.to(dtype=self.weight_dtype) for s in block_res_samples
            ],
            return_dict=False,
        )[0]

        # ── Loss ─────────────────────────────────────────────────────────
        model_pred = unpack_latents(model_pred, packed_latent_h, packed_latent_w)
        target = noise - pixel_latents
        loss = F.mse_loss(model_pred.float(), target.float())

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.condition_branch.parameters(),
                getattr(self.opt.train.optim, "max_grad_norm", 1.0),
            )
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()

        return {"all": loss}

    # ── Validation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def validation(self):
        gc.collect()
        torch.cuda.empty_cache()

        for model in self.models:
            model.eval()

        idx = 0
        for batch in tqdm(self.test_dataloader, desc="Validation"):
            idx = self.validate_step(
                batch, idx,
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
        """Run full LucidFlux denoising loop for inference."""
        self._ensure_null_embeddings()
        device = self.accelerator.device
        dtype = self.weight_dtype
        bsz = lq_image.shape[0]

        # Prepare conditioning
        condition_cond_lq = lq_image.to(dtype)
        if self.swinir is not None:
            lq_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
            pre_01 = self._run_swinir(lq_01)
            condition_cond_pre = (pre_01 * 2.0 - 1.0).to(dtype)
        else:
            pre_01 = None
            condition_cond_pre = condition_cond_lq

        # SigLIP features
        siglip_image_pre_fts = None
        if self.siglip is not None and pre_01 is not None:
            siglip_size = getattr(
                getattr(self.siglip, "config", None), "image_size", 512
            )
            siglip_pixels = siglip_from_unit_tensor(
                pre_01, size=(siglip_size, siglip_size)
            )
            siglip_image_pre_fts = self.siglip(
                pixel_values=siglip_pixels.to(device=device, dtype=self.siglip_dtype)
            ).last_hidden_state

        # Prepare embeddings
        txt = self.register_buffer_txt.to(device).expand(bsz, -1, -1).to(dtype)
        vec = self.register_buffer_vec.to(device).expand(bsz, -1).to(dtype)
        # FluxTransformer2DModel expects txt_ids as 2D: [seq_len, 3] (batch dim is deprecated)
        txt_ids = torch.zeros(txt.shape[1], 3, device=device, dtype=dtype)

        # Build siglip_txt (T5 + Redux projected SigLIP)
        siglip_txt = txt
        siglip_txt_ids = txt_ids
        if siglip_image_pre_fts is not None:
            cb = self.accelerator.unwrap_model(self.condition_branch)
            enc_dtype = cb.redux_image_encoder.redux_up.weight.dtype
            image_embeds = cb.redux_image_encoder(
                siglip_image_pre_fts.to(device=device, dtype=enc_dtype)
            )["image_embeds"].to(dtype=dtype)
            siglip_txt = torch.cat([txt, image_embeds], dim=1)
            extra_ids = torch.zeros(1024, 3, device=device, dtype=dtype)
            siglip_txt_ids = torch.cat([txt_ids, extra_ids], dim=0)

        # Generate noise
        noise = get_noise(bsz, height, width, device, dtype, seed=42)

        # Prepare packed noise + img_ids
        lat_h, lat_w = noise.shape[2], noise.shape[3]
        img = rearrange(noise, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        img_ids = torch.zeros(lat_h // 2, lat_w // 2, 3, device=device, dtype=dtype)
        img_ids[..., 1] += torch.arange(lat_h // 2, device=device)[:, None]
        img_ids[..., 2] += torch.arange(lat_w // 2, device=device)[None, :]
        # img_ids = rearrange(img_ids, "h w c -> (h w) c")
        img_ids = img_ids.unsqueeze(0).expand(bsz, -1, -1, -1)
        img_ids = rearrange(img_ids, "b h w c -> b (h w) c")

        # Get schedule
        image_seq_len = img.shape[1]
        timesteps_schedule = get_schedule(num_steps, image_seq_len)

        # Unwrap for inference
        cb = self.accelerator.unwrap_model(self.condition_branch)
        
        # Denoise
        img = denoise_lucidflux(
            model=self.transformer,
            dual_condition_model=cb.condition_branch,
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            siglip_txt=siglip_txt,
            siglip_txt_ids=siglip_txt_ids,
            vec=vec,
            timesteps=timesteps_schedule,
            guidance=self.guidance_scale,
            condition_cond_lq=condition_cond_lq,
            condition_cond_pre=condition_cond_pre,
        )

        # Unpack and decode
        img = unpack(img, height, width)
        decoded = decode_latents_flux(img, self.vae)
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

        out = self.forwardpass(
            lq, lq.shape[-2], lq.shape[-1], num_inference_steps
        )

        if self.swinir is not None:
            with torch.inference_mode():
                lq_01 = torch.clamp((lq.float() + 1.0) / 2.0, 0.0, 1.0)
                pre_01 = self._run_swinir(lq_01)
            swinir_np = pre_01.cpu().float().numpy().clip(0, 1)

        # Convert to [0, 1] for logging
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
            if swinir_np is not None:
                log_image(self.opt, self.accelerator, swinir_np[i : i + 1], f"swinir_{idx:04d}", self.global_step)
           
            log_metrics(gt_np[i], out_np[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
