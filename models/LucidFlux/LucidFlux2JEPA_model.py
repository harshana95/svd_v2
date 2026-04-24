import copy
import gc
import os

import einops
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Pipeline, Flux2Transformer2DModel
try:
    from diffusers import AutoencoderKLFlux2
except ImportError:
    from diffusers import AutoencoderKL as AutoencoderKLFlux2
from tqdm import tqdm

from models.base_model import BaseModel
from utils import log_image, log_metrics

from .src.flux.condition import (
    FluxParams,
    build_condition_branch_fresh,
    load_condition_branch_with_redux,
)
from .src.flux.lucidflux import (
    extract_vjepa_tokens,
    load_lucidflux_weights,
    load_precomputed_embeddings,
    load_swinir,
    load_vjepa2_model,
    load_vjepa2_weights,
    vjepa_from_unit_tensor,
)


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def encode_images_flux2(
    pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype, generator=None
):
    if hasattr(vae, "bn") and hasattr(vae, "_patchify_latents"):
        image_latents = retrieve_latents(
            vae.encode(pixels.to(vae.dtype)), generator=generator, sample_mode="argmax"
        )
        image_latents = vae._patchify_latents(image_latents)
        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
            image_latents.device, image_latents.dtype
        )
        latents_bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        )
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std
    else:
        pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
        shift = getattr(vae.config, "shift_factor", 0.0)
        scale = getattr(vae.config, "scaling_factor", 1.0)
        image_latents = (pixel_latents - shift) * scale

    return image_latents.to(weight_dtype)


def decode_latents_flux(latents: torch.Tensor, vae: torch.nn.Module):
    shift = getattr(vae.config, "shift_factor", 0.0)
    scale = getattr(vae.config, "scaling_factor", 1.0)
    latents = latents / scale + shift
    image = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
    return image


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
    sigmas = sigmas.view(-1, 1, 1, 1)
    noisy_model_input = (1 - sigmas) * latents + sigmas * noise
    return noisy_model_input, timesteps, sigmas


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return float(image_seq_len) * m + b


class LucidFlux2JEPA_model(BaseModel):
    def __init__(self, opt, logger):
        super(LucidFlux2JEPA_model, self).__init__(opt, logger)

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        # creating condition branch with redux
        flux_params = FluxParams(
            in_channels=32,
            cond_in_channels=16,
            cond_patch_size=1,
            context_in_dim=4096,
            text_in_dim=15360,
            vec_in_dim=512*4,
            depth=0, # we use redux as a single block
        )
        lucidflux_weights_path = getattr(opt, "lucidflux_weights_path", None)
        self.vjepa_hub_entry = getattr(opt, "vjepa_hub_entry", "vjepa2_1_vit_large_384")
        self.vjepa_image_size = getattr(opt, "vjepa_image_size", 384)
        self.vjepa_pretrained = getattr(opt, "vjepa_pretrained", False)
        self.vjepa_checkpoint_path = getattr(opt, "vjepa_checkpoint_path", None)
        self.vjepa_dtype = torch.float32

        if "vit_base" in self.vjepa_hub_entry:
            self.vjepa_feature_dim = 768
        elif "vit_large" in self.vjepa_hub_entry:
            self.vjepa_feature_dim = 1024
        elif "vit_giant" in self.vjepa_hub_entry and "gigantic" not in self.vjepa_hub_entry:
            self.vjepa_feature_dim = 1280
        elif "vit_gigantic" in self.vjepa_hub_entry:
            self.vjepa_feature_dim = 1664
        else:
            self.vjepa_feature_dim = getattr(opt, "vjepa_feature_dim", 1024)

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
                vision_feature_dim=self.vjepa_feature_dim,
            )
        else:
            logger.info("Building fresh LucidFlux 2 JEPA condition branch")
            self.condition_branch = build_condition_branch_fresh(
                params=flux_params,
                device="cpu",
                branch_dtype=torch.float32,
                connector_dtype=torch.float32,
                vision_feature_dim=self.vjepa_feature_dim,
            )
            
        self.condition_branch.train()
        self.condition_branch.requires_grad_(True)
        
        for module in [
            self.condition_branch.condition_branch.lq,
            self.condition_branch.condition_branch.pre,
        ]:
            if hasattr(module, "_set_gradient_checkpointing"):
                module._set_gradient_checkpointing(module, True)

        swinir_path = getattr(opt, "swinir_model_path", None)
        self.swinir = None
        if swinir_path and os.path.exists(swinir_path):
            logger.info(f"Loading SwinIR from {swinir_path}")
            self.swinir = load_swinir("cpu", swinir_path, offload=False)

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

        # loading precomputed embeddings
        embeddings_path = getattr(opt, "precomputed_embeddings_path", None)
        if embeddings_path and os.path.exists(embeddings_path):
            embeds = load_precomputed_embeddings(embeddings_path, "cpu")
            self.register_buffer_txt = embeds["txt"]
            self.register_buffer_vec = embeds["vec"]
        else:
            logger.warning(
                "No precomputed embeddings path found. Will generate null embeddings at first forward pass."
            )
            self.register_buffer_txt = None
            self.register_buffer_vec = None
        
        # loading Flux 2 transformer and vae
        self.guidance_scale = getattr(opt, "guidance_scale", 4.0)
        self.discrete_flow_shift = getattr(opt, "discrete_flow_shift", 3.1582)
        pretrained = opt.pretrained_model_name_or_path
        revision = getattr(opt, "revision", None)
        variant = getattr(opt, "variant", None)
        self.transformer = Flux2Transformer2DModel.from_pretrained(
            pretrained, subfolder="transformer", revision=revision, variant=variant
        )
        self.transformer.requires_grad_(False)
        self.transformer.eval()
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
        self.transformer_context_dim = getattr(self.transformer.config, "joint_attention_dim", 15360)
        self.transformer_rope_dims = len(
            getattr(self.transformer.config, "axes_dims_rope", (32, 32, 32, 32))
        )

        self.transformer = self.transformer.to(self.accelerator.device, dtype=weight_dtype)
        self.vae = self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.condition_branch = self.condition_branch.to(self.accelerator.device)

        self.encoder_state_adapter = torch.nn.Linear(
            flux_params.hidden_size, flux_params.context_in_dim
        )
        self.encoder_state_adapter.requires_grad_(True)
        self.encoder_state_adapter = self.encoder_state_adapter.to(
            self.accelerator.device, dtype=weight_dtype
        )

        self.encoder_state_projection = torch.nn.Linear(
            flux_params.context_in_dim, self.transformer_context_dim
        )
        self.encoder_state_projection.requires_grad_(True)
        self.encoder_state_projection = self.encoder_state_projection.to(self.accelerator.device, dtype=weight_dtype)
        
        self.vjepa = self.vjepa.to(self.accelerator.device, dtype=self.vjepa_dtype)
        if self.swinir is not None:
            self.swinir = self.swinir.to(self.accelerator.device)

        self.models.append(self.condition_branch)
        self.models.append(self.encoder_state_adapter)
        self.models.append(self.encoder_state_projection)

        for m in self.models:
            all_param = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            logger.info(
                f"{m.__class__.__name__}: trainable={trainable:,} / total={all_param:,} "
                f"({100 * trainable / all_param:.1f}%)"
            )

    def _ensure_null_embeddings(self):
        if self.register_buffer_txt is not None:
            return
        self.logger.info("Generating null-text embeddings via Flux 2 text encoders...")
        pipe = Flux2Pipeline.from_pretrained(
            self.opt.pretrained_model_name_or_path,
            transformer=None,
            vae=None,
            torch_dtype=self.weight_dtype,
        ).to(self.accelerator.device)
        with torch.no_grad():
            try:
                result = pipe.encode_prompt(
                    [""], num_images_per_prompt=1, do_classifier_free_guidance=False
                )
                if isinstance(result, tuple) and len(result) >= 2:
                    prompt_embeds, pooled_prompt_embeds = result[0], result[1]
                elif isinstance(result, dict):
                    prompt_embeds = result.get("prompt_embeds")
                    pooled_prompt_embeds = result.get("pooled_prompt_embeds")
                else:
                    prompt_embeds, pooled_prompt_embeds = result, None
            except Exception:
                prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
                    [""], prompt_2=None
                )
                
        self.register_buffer_txt = prompt_embeds.cpu()
        self.register_buffer_vec = (
            pooled_prompt_embeds.cpu()
            if pooled_prompt_embeds is not None
            else torch.zeros((1, 512, 4), dtype=prompt_embeds.dtype)
        )
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    def setup_optimizers(self):
        opt = self.opt.train.optim
        optimizer_type = getattr(opt, "type", "adamw").lower()
        trainable_params = [
            p
            for module in [self.condition_branch, self.encoder_state_adapter, self.encoder_state_projection]
            for p in module.parameters()
            if p.requires_grad
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

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        if is_train and self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.train)
        elif not is_train and self.opt.val.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.val)

    def _extract_vision_features(self, pre_01: torch.Tensor) -> torch.Tensor:
        vjepa_pixels = vjepa_from_unit_tensor(
            pre_01,
            size=self.vjepa_image_size,
            device=self.accelerator.device,
            out_dtype=self.vjepa_dtype,
        )
        with torch.no_grad():
            return extract_vjepa_tokens(
                self.vjepa,
                vjepa_pixels,
                device=self.accelerator.device,
                dtype=self.vjepa_dtype,
            )

    def _prepare_text_inputs(self, batch_size: int, device: torch.device):
        self._ensure_null_embeddings()
        txt = self.register_buffer_txt.to(device).expand(batch_size, -1, -1).to(self.weight_dtype)
        vec = self.register_buffer_vec.to(device).expand(batch_size, -1, -1).to(self.weight_dtype)
        vec = vec.view(batch_size, -1)
        txt_ids = torch.zeros(
            txt.shape[1], self.transformer_rope_dims, device=device, dtype=self.weight_dtype
        )
        return txt, vec, txt_ids

    def _block_res_to_encoder_states(
        self,
        block_res_samples: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
        imgtxt: torch.Tensor,
        imgtxt_ids: torch.Tensor,
        img_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # if isinstance(block_res_samples, torch.Tensor):
        #     last_block = block_res_samples
        # else:
        #     if len(block_res_samples) == 0:
        #         raise ValueError("Expected non-empty block_res_samples")
        #     last_block = block_res_samples[-1]
        
        # redux_like_embeds = self.encoder_state_adapter(last_block.to(self.weight_dtype))
        # encoder_hidden_states = torch.cat(
        #     [imgtxt.to(self.weight_dtype), redux_like_embeds], dim=1
        # ) # (B, 1088 + 4096 = 5184, 4096)
        # encoder_hidden_states = self.encoder_state_projection(
        #     encoder_hidden_states.to(self.weight_dtype)
        # )
        
        # if imgtxt_ids.ndim == 2:
        #     extra_ids = img_ids[0].to(self.weight_dtype)
        #     encoder_txt_ids = torch.cat([imgtxt_ids.to(self.weight_dtype), extra_ids], dim=0)
        # elif imgtxt_ids.ndim == 3:
        #     extra_ids = img_ids.to(self.weight_dtype)
        #     encoder_txt_ids = torch.cat([imgtxt_ids.to(self.weight_dtype), extra_ids], dim=1)
        # else:
        #     raise ValueError(f"Expected imgtxt_ids to be 2D or 3D, got {imgtxt_ids.ndim}D")

        encoder_hidden_states = self.encoder_state_projection(imgtxt.to(self.weight_dtype))
        encoder_txt_ids = imgtxt_ids.to(self.weight_dtype)

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

        with torch.no_grad():
            pixel_latents = encode_images_flux2(gt_image, self.vae, self.weight_dtype)
            # control_latents = encode_images_flux2(lq_image, self.vae, self.weight_dtype)

        noise = torch.randn_like(pixel_latents)
        noisy_latents, timesteps, _ = get_noisy_model_input_and_timesteps(
            self.noise_scheduler_copy,
            pixel_latents,
            noise,
            pixel_latents.device,
            discrete_flow_shift=self.discrete_flow_shift,
        )

        lat_h, lat_w = noisy_latents.shape[2], noisy_latents.shape[3]
        transformer_latents = Flux2Pipeline._patchify_latents(noisy_latents)
        
        packed_transformer_input = Flux2Pipeline._pack_latents(transformer_latents)
        packed_noisy_model_input = Flux2Pipeline._pack_latents(noisy_latents)
        
        latent_image_ids = Flux2Pipeline._prepare_latent_ids(
            noisy_latents
        ).to(device=self.accelerator.device, dtype=self.weight_dtype)
        
        transformer_image_ids = Flux2Pipeline._prepare_latent_ids(
            transformer_latents
        ).to(device=self.accelerator.device, dtype=self.weight_dtype)

        txt, vec, txt_ids = self._prepare_text_inputs(bsz, self.accelerator.device)

        condition_cond_lq = lq_image.to(self.weight_dtype)
        with torch.no_grad():
            if self.swinir is not None:
                lq_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
                pre_01 = self.swinir(lq_01).detach().clamp(0.0, 1.0)
                condition_cond_pre = (pre_01 * 2.0 - 1.0).to(self.weight_dtype)
            else:
                pre_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
                condition_cond_pre = condition_cond_lq

        vision_image_pre_fts = self._extract_vision_features(pre_01)

        guidance_vec = torch.full(
            (bsz,),
            self.guidance_scale,
            device=self.accelerator.device,
            dtype=self.weight_dtype,
        )
        normalized_timesteps = (timesteps.float() / 1000).to(self.weight_dtype)

        block_res_samples, vision_txt, vision_txt_ids = self.condition_branch(
            img=packed_noisy_model_input.to(self.weight_dtype),
            img_ids=latent_image_ids.to(self.weight_dtype),
            condition_cond_lq=condition_cond_lq,
            condition_cond_pre=condition_cond_pre,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=normalized_timesteps,
            guidance=guidance_vec,
            vision_image_pre_fts=vision_image_pre_fts,
            return_last_only=True,
        )

        # block_res_samples 19 x (1, 4096, 3072)
        # vision_txt (1, 1088, 4096)
        # vision_txt_ids (1088, 4)
        # latent_image_ids (1, 4096, 4)
        
        encoder_hidden_states, encoder_txt_ids = self._block_res_to_encoder_states(
            block_res_samples, vision_txt, vision_txt_ids, latent_image_ids
        )
        # encoder_hidden_states: (B, 5184, 15360)
        # encoder_txt_ids: (5184, 4)
        
        model_pred = self.transformer(
            hidden_states=packed_transformer_input.to(self.weight_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weight_dtype),
            timestep=normalized_timesteps,
            guidance=guidance_vec,
            txt_ids=encoder_txt_ids.to(self.weight_dtype),
            img_ids=transformer_image_ids.to(self.weight_dtype),
            return_dict=False,
        )[0]
        model_pred = Flux2Pipeline._unpack_latents_with_ids(
            model_pred,
            transformer_image_ids,
        )
        model_pred = Flux2Pipeline._unpatchify_latents(model_pred)

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

        condition_cond_lq = lq_image.to(dtype)
        if self.swinir is not None:
            lq_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
            pre_01 = self.swinir(lq_01).detach().clamp(0.0, 1.0)
            condition_cond_pre = (pre_01 * 2.0 - 1.0).to(dtype)
        else:
            pre_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
            condition_cond_pre = condition_cond_lq

        vision_image_pre_fts = self._extract_vision_features(pre_01)
        txt, vec, txt_ids = self._prepare_text_inputs(bsz, device)

        cb = self.accelerator.unwrap_model(self.condition_branch)

        control_latents = encode_images_flux2(lq_image, self.vae, dtype)
        lat_c, lat_h, lat_w = (
            control_latents.shape[1],
            control_latents.shape[2],
            control_latents.shape[3],
        )

        guidance_vec = torch.full((bsz,), self.guidance_scale, device=device, dtype=dtype)

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

        for t in scheduler.timesteps:
            transformer_latents = Flux2Pipeline._patchify_latents(latents)
            packed_transformer_input = Flux2Pipeline._pack_latents(transformer_latents)
            packed_latents = Flux2Pipeline._pack_latents(latents)

            latent_image_ids = Flux2Pipeline._prepare_latent_ids(
                latents
            ).to(device=device, dtype=dtype)
            transformer_image_ids = Flux2Pipeline._prepare_latent_ids(
                transformer_latents
            ).to(device=device, dtype=dtype)

            timestep = t.unsqueeze(0).expand(bsz).to(device=device, dtype=dtype) / 1000.0
            block_res_samples, vision_txt, vision_txt_ids = cb(
                img=packed_latents,
                img_ids=latent_image_ids,
                condition_cond_lq=condition_cond_lq,
                condition_cond_pre=condition_cond_pre,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=timestep,
                guidance=guidance_vec,
                vision_image_pre_fts=vision_image_pre_fts,
                return_last_only=True,
            )
            encoder_hidden_states, encoder_txt_ids = self._block_res_to_encoder_states(
                block_res_samples, vision_txt, vision_txt_ids, latent_image_ids
            )
            noise_pred = self.transformer(
                hidden_states=packed_transformer_input,
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                timestep=timestep,
                guidance=guidance_vec,
                txt_ids=encoder_txt_ids.to(dtype),
                img_ids=transformer_image_ids.to(dtype),
                return_dict=False,
            )[0]
            noise_pred = Flux2Pipeline._unpack_latents_with_ids(
                noise_pred,
                transformer_image_ids,
            )
            noise_pred = Flux2Pipeline._unpatchify_latents(noise_pred)
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        decoded = decode_latents_flux(latents, self.vae)
        return decoded.clamp(-1, 1)

    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        num_inference_steps = getattr(self.opt.val, "num_inference_steps", 50)

        lq = self.sample[lq_key]
        gt = self.sample[gt_key]
        bsz = lq.shape[0]

        if lq.shape[1] == 1:
            lq = einops.repeat(lq, "b 1 h w -> b 3 h w")

        out = self.forwardpass(
            lq, lq.shape[-2], lq.shape[-1], num_inference_steps
        )

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
            log_metrics(gt_np[i], out_np[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
