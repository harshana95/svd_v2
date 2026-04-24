import copy
import gc
import os

from diffusers import FluxPriorReduxPipeline
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
from .src.flux.condition2 import (
    build_condition_branch_fresh,
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

def encode_vae_image(vae, image: torch.Tensor, generator: torch.Generator):
    if image.ndim != 4:
        raise ValueError(f"Expected image dims 4, got {image.ndim}.")

    image_latents = retrieve_latents(vae.encode(image), generator=generator, sample_mode="argmax")
    image_latents = Flux2Pipeline._patchify_latents(image_latents)

    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    image_latents = (image_latents - latents_bn_mean) / latents_bn_std
    return image_latents

def prepare_image_latents(vae,images, batch_size, generator, device, dtype):
    image_latents = []
    for image in images:
        image = image.to(device=device, dtype=dtype)
        imagge_latent = encode_vae_image(vae, image=image, generator=generator)
        image_latents.append(imagge_latent)  # (1, 128, 32, 32)

    image_latent_ids = Flux2Pipeline._prepare_image_ids(image_latents)

    # Pack each latent and concatenate
    packed_latents = []
    for latent in image_latents:
        # latent: (1, 128, 32, 32)
        packed = Flux2Pipeline._pack_latents(latent)  # (1, 1024, 128)
        packed = packed.squeeze(0)  # (1024, 128) - remove batch dim
        packed_latents.append(packed)

    # Concatenate all reference tokens along sequence dimension
    image_latents = torch.cat(packed_latents, dim=0)  # (N*1024, 128)
    image_latents = image_latents.unsqueeze(0)  # (1, N*1024, 128)

    image_latents = image_latents.repeat(batch_size, 1, 1)
    image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
    image_latents = image_latents.to(device=device, dtype=dtype)
    image_latent_ids = image_latent_ids.to(device)

    return image_latents, image_latent_ids

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



def decode_latents_flux(latents: torch.Tensor, vae: torch.nn.Module):
    if hasattr(vae, "bn"):
        # Flux2 latents are patchified + BN-normalized before denoising.
        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = Flux2Pipeline._unpatchify_latents(latents)
    else:
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


class LucidFlux2BaselineJEPA_model(BaseModel):
    def __init__(self, opt, logger):
        super(LucidFlux2BaselineJEPA_model, self).__init__(opt, logger)

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
        
        self.pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev", 
            torch_dtype=self.weight_dtype).to(self.accelerator.device
        )
        #>>> image_embeds torch.Size([1, 1241, 4096]), torch.Size([1, 768])
        
        # self.pipe_prior_redux.requires_grad_(False)
        # self.pipe_prior_redux.eval()

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


        self.condition_branch = build_condition_branch_fresh(
            params=flux_params,
            device="cpu",
            connector_dtype=torch.float32,
            vision_feature_dim=self.vjepa_feature_dim,
        )
            
        self.condition_branch.train()
        self.condition_branch.requires_grad_(True)
        
        swinir_path = getattr(opt, "swinir_model_path", None)
        self.swinir = None
        if swinir_path and os.path.exists(swinir_path):
            logger.info(f"Loading SwinIR from {swinir_path}")
            self.swinir = load_swinir("cpu", swinir_path, offload=False)


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

        self.vjepa = self.vjepa.to(self.accelerator.device, dtype=self.vjepa_dtype)
        if self.swinir is not None:
            self.swinir = self.swinir.to(self.accelerator.device)

        self.models.append(self.condition_branch)
        self.models.append(self.encoder_state_adapter)

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
            for module in [self.condition_branch, self.encoder_state_adapter]
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

    def _block_res_to_encoder_states(self, imgtxt: torch.Tensor, imgtxt_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_hidden_states = imgtxt.to(self.weight_dtype) #self.encoder_state_projection()
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

        # prepare noisy latent tokens
        with torch.no_grad():
            pixel_latents = encode_vae_image(self.vae, gt_image.to(self.weight_dtype), None)
            
        noise = torch.randn_like(pixel_latents)
        noisy_latents, timesteps, _ = get_noisy_model_input_and_timesteps(
            self.noise_scheduler_copy,
            pixel_latents,
            noise,
            pixel_latents.device,
            discrete_flow_shift=self.discrete_flow_shift,
        )

        transformer_latents = noisy_latents
        packed_transformer_input = Flux2Pipeline._pack_latents(transformer_latents)
        transformer_image_ids = Flux2Pipeline._prepare_latent_ids(
            transformer_latents
        ).to(device=self.accelerator.device, dtype=self.weight_dtype)
        
        # prepare conditional latent tokens
        latent_token_count = packed_transformer_input.shape[1]
        ref_images = [lq_image[i : i + 1].to(self.weight_dtype) for i in range(bsz)]
        image_latents, image_latent_ids = prepare_image_latents(
            vae=self.vae,
            images=ref_images,
            batch_size=bsz,
            generator=None,
            device=self.accelerator.device,
            dtype=self.weight_dtype,
        )

        with torch.no_grad():
            if self.swinir is not None:
                lq_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
                pre_01 = self.swinir(lq_01).detach().clamp(0.0, 1.0)
            else:
                pre_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
        vision_image_pre_fts = self._extract_vision_features(pre_01)

        # prepare text inputs
        txt, vec, txt_ids = self._prepare_text_inputs(bsz, self.accelerator.device)

        
        guidance_vec = torch.full(
            (bsz,),
            self.guidance_scale,
            device=self.accelerator.device,
            dtype=self.weight_dtype,
        )
        normalized_timesteps = (timesteps.float() / 1000).to(self.weight_dtype)

        # join text and vision features
        vision_txt, vision_txt_ids = self.condition_branch(
            txt=txt,
            txt_ids=txt_ids,
            guidance=guidance_vec,
            vision_image_pre_fts=vision_image_pre_fts,
        )

        encoder_hidden_states, encoder_txt_ids = self._block_res_to_encoder_states(
            vision_txt, vision_txt_ids
        )
        

        latent_model_input = packed_transformer_input
        latent_image_ids = transformer_image_ids
        if image_latents is not None:
            latent_model_input = torch.cat([packed_transformer_input, image_latents], dim=1)
            latent_image_ids = torch.cat([transformer_image_ids, image_latent_ids], dim=1)

        model_pred = self.transformer(
            hidden_states=latent_model_input.to(self.weight_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weight_dtype),
            timestep=normalized_timesteps,
            guidance=guidance_vec,
            txt_ids=encoder_txt_ids.to(self.weight_dtype),
            img_ids=latent_image_ids.to(self.weight_dtype),
            return_dict=False,
        )[0]
        model_pred = model_pred[:, :latent_token_count]
        model_pred = Flux2Pipeline._unpack_latents_with_ids(
            model_pred,
            transformer_image_ids,
        )

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
        else:
            pre_01 = torch.clamp((lq_image.float() + 1.0) / 2.0, 0.0, 1.0)
        

        vision_image_pre_fts = self._extract_vision_features(pre_01)
        txt, vec, txt_ids = self._prepare_text_inputs(bsz, device)

        cb = self.accelerator.unwrap_model(self.condition_branch)

        # Match Flux2 prepare_latents() sizing without an extra VAE encode.
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        latent_spatial_divisor = vae_scale_factor * 2
        lat_c = self.transformer.config.in_channels
        lat_h = int(height) // latent_spatial_divisor
        lat_w = int(width) // latent_spatial_divisor
        # control_latents = encode_images_flux2(lq_image, self.vae, dtype)
        # lat_c, lat_h, lat_w = (
        #     control_latents.shape[1],
        #     control_latents.shape[2],
        #     control_latents.shape[3],
        # )

        guidance_vec = torch.full((bsz,), self.guidance_scale, device=device, dtype=dtype)
        ref_images = [lq_image[i : i + 1].to(dtype) for i in range(bsz)]
        image_latents, image_latent_ids = prepare_image_latents(
            vae=self.vae,
            images=ref_images,
            batch_size=bsz,
            generator=None,
            device=device,
            dtype=dtype,
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

        for t in scheduler.timesteps:
            transformer_latents = latents
            packed_transformer_input = Flux2Pipeline._pack_latents(transformer_latents)
            transformer_image_ids = Flux2Pipeline._prepare_latent_ids(
                transformer_latents
            ).to(device=device, dtype=dtype)
            latent_token_count = packed_transformer_input.shape[1]

            timestep = t.unsqueeze(0).expand(bsz).to(device=device, dtype=dtype) / 1000.0
            vision_txt, vision_txt_ids = cb(
                txt=txt,
                txt_ids=txt_ids,
                guidance=guidance_vec,
                vision_image_pre_fts=vision_image_pre_fts,
            )
            encoder_hidden_states, encoder_txt_ids = self._block_res_to_encoder_states(
                vision_txt, vision_txt_ids
            )
            latent_model_input = packed_transformer_input
            latent_image_ids = transformer_image_ids
            if image_latents is not None:
                latent_model_input = torch.cat([packed_transformer_input, image_latents], dim=1)
                latent_image_ids = torch.cat([transformer_image_ids, image_latent_ids], dim=1)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=encoder_hidden_states.to(dtype),
                timestep=timestep,
                guidance=guidance_vec,
                txt_ids=encoder_txt_ids.to(dtype),
                img_ids=latent_image_ids.to(dtype),
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, :latent_token_count]
            noise_pred = Flux2Pipeline._unpack_latents_with_ids(
                noise_pred,
                transformer_image_ids,
            )
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
