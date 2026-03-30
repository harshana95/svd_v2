# Note: 
# Harshana: Added the dtype mismatch workaround in the FluxKontextPipeline.step() method.
# Cast if the latents dtype is different from the transformer dtype.

import copy
import gc
import glob
import os
import random
import inspect
import einops
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    FluxTransformer2DModel,
)
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig
from torchvision import transforms
from ram import inference_ram as inference

from torch.utils.data import DataLoader, Subset
from models.base_model import BaseModel
from utils.dataset_utils import merge_patches
from utils import log_image, log_metrics

from safetensors.torch import load_file


def encode_images_flux(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    """Encode pixel images to Flux latent space (supports shift_factor if present)."""
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    shift = getattr(vae.config, "shift_factor", 0.0)
    scale = getattr(vae.config, "scaling_factor", vae.config.scaling_factor)
    pixel_latents = (pixel_latents - shift) * scale
    return pixel_latents.to(weight_dtype)


# def load_model_hook(models, input_dir):
#     saved = {}
#     while len(models) > 0:
#         # pop models so that they are not loaded again
#         model = models.pop()
#         class_name = model._get_name()
#         saved[class_name] = 1 if class_name not in saved.keys() else saved[class_name] + 1
#         print(f"Loading model {class_name}_{saved[class_name]} from {input_dir}")

#         # load diffusers style into model
#         folder = os.path.join(input_dir, f"{class_name}_{saved[class_name]}")
#         combined_state_dict = {}
    
#         for file_path in glob.glob(os.path.join(folder, "*.safetensors")):
#             state_dict_part = load_file(file_path)
#             combined_state_dict.update(state_dict_part)
#         try:
#             model.load_state_dict(combined_state_dict)
#         except Exception as e:
#             print(f"{'='*50} Failed to load {class_name} {'='*50} {model} {e}")


def initialize_vae(vae):
    l_target_modules_encoder = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
    
    lora_conf_encoder = LoraConfig(r=4, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    vae.set_adapter(['default_encoder'])
    return vae


def initialize_transformer_flux(transformer):
    """LoRA for FluxTransformer2DModel matching official train_control_lora_flux.py targets.

    Targets all attention projections in both image-token and text-token streams
    (double-stream blocks) plus feedforward layers and x_embedder.
    Previously the pattern-match approach missed add_*_proj and ff_context modules.
    """
    target_modules = [
        "x_embedder",
        # Image-token attention (all block types)
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
        # Text-token attention projections (double-stream MMDiT blocks)
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
        # Image-token feedforward
        "ff.net.0.proj", "ff.net.2",
        # Text-token feedforward (double-stream blocks)
        "ff_context.net.0.proj", "ff_context.net.2",
    ]
    # Filter to only modules that actually exist in this transformer
    existing = {n.replace(".weight", "") for n, _ in transformer.named_parameters()}
    target_modules = [m for m in target_modules if any(m in e for e in existing)]
    lora_conf = LoraConfig(r=32, init_lora_weights="gaussian", target_modules=target_modules)
    transformer.add_adapter(lora_conf, adapter_name="default")
    transformer.set_adapter(["default"])
    return transformer

def get_caption_generator(model_path, **kwargs):
    # init vlm model
    from ram.models.ram_lora import ram
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model_vlm = ram(pretrained=model_path,
            pretrained_condition=kwargs.get('dape_path'),
            image_size=384,
            vit='swin_l')
    return model_vlm, ram_transforms


class FluxKontextDeblur_model(BaseModel):

    def __init__(self, opt, logger):
        super(FluxKontextDeblur_model, self).__init__(opt, logger)
        # self.load_handle.remove()
        # self.load_handle = self.accelerator.register_load_state_pre_hook(load_model_hook)
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.guidance_scale = getattr(opt, "guidance_scale", 3.5)
        self.only_target_transformer_blocks = getattr(opt, "only_target_transformer_blocks", False)
        self.weighting_scheme = getattr(opt, "weighting_scheme", "none")
        # Use sharp (GT) image for caption during training so text matches target (avoids flat/generic outputs).
        # At inference we still use the blur image for caption since GT is not available.
        self.caption_from_gt = getattr(opt, "caption_from_gt", True)

        pretrained_model_name_or_path = self.opt.pretrained_model_name_or_path
        revision = getattr(self.opt, "revision", None)
        variant = getattr(self.opt, "variant", None)

        # Flux: VAE, transformer, flow-matching scheduler
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant
        )
        self.transformer = FluxTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer", revision=revision, variant=variant
        )
        self.transformer.train()
        
        # Enable gradient checkpointing to trade compute for memory
        # This reduces memory usage during backward pass by recomputing activations
        if hasattr(self.transformer, 'enable_gradient_checkpointing'):
            self.transformer.enable_gradient_checkpointing()
            self.logger.info("Gradient checkpointing enabled for transformer")

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # Text encoding via FluxKontextPipeline (no separate text encoders)
        self.text_encoding_pipeline = FluxKontextPipeline.from_pretrained(
            pretrained_model_name_or_path,
            transformer=None,
            vae=None,
            torch_dtype=weight_dtype,
        )

        if not self.only_target_transformer_blocks:
            self.transformer.requires_grad_(True)
        self.vae.requires_grad_(False)

        if self.only_target_transformer_blocks:
            self.transformer.x_embedder.requires_grad_(True)
            for name, module in self.transformer.named_modules():
                if "transformer_blocks" in name:
                    module.requires_grad_(True)
                else:
                    module.requires_grad_(False)

        if opt.train.get("lora_finetune", False):
            self.transformer.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.transformer = initialize_transformer_flux(self.transformer)
            # freeze VAE
            # self.vae = initialize_vae(self.vae)
            # self.vae.set_adapter(["default_encoder"])
            self.transformer.set_adapter(["default"])
            for n, _p in self.vae.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
            for n, _p in self.transformer.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
        else:
            self.vae.requires_grad_(False)

        # Convert modules to weight_dtype to reduce memory usage during optimizer step
        # Optimizer states (AdamW momentum/variance) will be in the same dtype as parameters
        self.transformer = self.transformer.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae = self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        self.models.append(self.vae)
        self.models.append(self.transformer)

        # Captioning model for prompts
        self.model_vlm, self.model_vlm_transforms = get_caption_generator(
            opt.vlm_model_path, dape_path=getattr(opt, "dape_path", None)
        )
        self.model_vlm.eval()
        self.model_vlm.to(self.accelerator.device, dtype=torch.float16)

        for m in self.models:
            print(m.__class__.__name__)
            all_param = sum(p.numel() for p in m.parameters())
            trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )
        
    def setup_optimizers(self):
        opt = self.opt.train.optim

        # Optimizer creation
        # Use 8-bit AdamW if available and requested (reduces optimizer state memory by ~4x)
        use_8bit_adam = getattr(opt, 'use_8bit_adam', False)
        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_class = bnb.optim.AdamW8bit
                self.logger.info("Using 8-bit AdamW optimizer for memory efficiency")
            except ImportError:
                self.logger.warning("bitsandbytes not available, falling back to standard AdamW")
                optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.AdamW
        
        self.gen_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.gen_params += [p for p in self.vae.parameters() if p.requires_grad]

        optimizer = optimizer_class(
            self.gen_params,
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2),
            weight_decay=opt.adam_weight_decay,
            eps=opt.adam_epsilon,
            )
        self.optimizers.append(optimizer)


    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        if self.opt.train.patched if is_train else self.opt.val.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.train if is_train else self.opt.val)
    
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler_copy.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        return model._orig_mod if is_compiled_module(model) else model

    def _to_device_safe(self, module, device):
        """Move module to device; no-op when module has meta tensors (e.g. under FSDP)."""
        try:
            return module.to(device)
        except NotImplementedError as e:
            if "meta tensor" in str(e).lower() or "to_empty" in str(e).lower():
                return module
            raise

    def optimize_parameters(self):
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        image_1 = self.sample[lq_key]  # conditioning (blur)
        gt_1 = self.sample[gt_key]    # target (sharp)
        bsz = image_1.shape[0]

        if image_1.shape[1] == 1:
            image_1 = einops.repeat(image_1, 'b 1 h w -> b 3 h w')
        if gt_1.shape[1] == 1:
            gt_1 = einops.repeat(gt_1, 'b 1 h w -> b 3 h w')

        image_1 = image_1.clip(-1, 1)
        gt_1 = gt_1.clip(-1, 1)

        # Encode sharp target and blurry conditioning to Flux latents.
        vae_dev = self._to_device_safe(self.vae, self.accelerator.device)
        pixel_latents = encode_images_flux(gt_1, vae_dev, self.weight_dtype)
        control_latents = encode_images_flux(image_1, vae_dev, self.weight_dtype)

        # Timestep sampling
        noise = torch.randn_like(pixel_latents, device=self.accelerator.device, dtype=self.weight_dtype)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=bsz,
            logit_mean=getattr(self.opt, "logit_mean", 0.0),
            logit_std=getattr(self.opt, "logit_std", 1.0),
            mode_scale=getattr(self.opt, "mode_scale", 1.29),
        )
        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

        # Flow matching: concatenate noisy target latents with control (blur) latents,
        # matching Flux Kontext's image-conditioning pattern (target + image_1).
        sigmas = self.get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
        noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
        
        # Pack latents for FluxKontext. `num_channels_latents` must be the base
        # latent channel count (from the target branch); packing expands to 4×
        # channels internally and expects this value, not the concatenated count.
        lat_h, lat_w = pixel_latents.shape[2], pixel_latents.shape[3]
        
        packed_noisy_model_input= FluxKontextPipeline._pack_latents(
            noisy_model_input,
            batch_size=bsz,
            num_channels_latents=self.transformer.config.in_channels // 4,
            height=lat_h,
            width=lat_w,
        )
        packed_control_latents = FluxKontextPipeline._pack_latents(
            control_latents,
            batch_size=bsz,
            num_channels_latents=self.transformer.config.in_channels // 4,
            height=lat_h,
            width=lat_w,
        )
        model_input = torch.cat([packed_noisy_model_input, packed_control_latents], dim=1)
        
        latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
            bsz,
            lat_h // 2,
            lat_w // 2,
            self.accelerator.device,
            self.weight_dtype,
        )
        image_ids = FluxKontextPipeline._prepare_latent_image_ids(bsz, lat_h // 2, lat_w // 2, self.accelerator.device, self.weight_dtype)

        latent_image_ids = torch.cat([latent_image_ids, image_ids], dim=0) # dim 0 is sequence dimension

        # Guidance
        flux_transformer = self._unwrap_model(self.transformer)
        if getattr(flux_transformer.config, "guidance_embeds", False):
            guidance_vec = torch.full(
                (bsz,), self.guidance_scale, device=noisy_model_input.device, dtype=self.weight_dtype
            )
        else:
            guidance_vec = None

        # Text encoding via pipeline (caption_from_gt: use sharp image so text matches target)
        caption_image = gt_1 if self.caption_from_gt else image_1
        gt_ram = self.model_vlm_transforms(caption_image * 0.5 + 0.5)
        caption = inference(gt_ram.to(dtype=torch.float16), self.model_vlm)
        captions = [c for c in caption]
        self.text_encoding_pipeline = self.text_encoding_pipeline.to(self.accelerator.device)
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = self.text_encoding_pipeline.encode_prompt(
                captions, prompt_2=None
            )

        # Predict
        model_pred = self.transformer(
            hidden_states=model_input,
            timestep=timesteps / 1000.0,
            guidance=guidance_vec,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
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

        # Flow-matching loss: target = noise - pixel_latents
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme, sigmas=sigmas
        )
        target = noise - pixel_latents
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            params_to_clip = self.transformer.parameters()
            self.accelerator.clip_grad_norm_(params_to_clip, getattr(self.opt.train.optim, "max_grad_norm", 1.0))
        
        # Clear cache before optimizer step to free up memory
        torch.cuda.empty_cache()
        
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()
        
        # Clear cache after optimizer step as well
        torch.cuda.empty_cache()

        return {"all": loss}
    
    @torch.no_grad()
    def validation(self):
        gc.collect()
        torch.cuda.empty_cache()
        # return
        idx = 0
        for model in self.models:
            model.eval()
        inferece_dtype = self.weight_dtype
        # For validation, run the FluxKontextPipeline in float32 end-to-end to avoid
        # mixed-dtype issues inside the VAE decoder.
        for model in self.models:
            self._to_device_safe(model, self.accelerator.device)
            model.to(dtype=inferece_dtype)

        transformer = self._unwrap_model(self.transformer)
        noise_scheduler_tmp = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.opt.pretrained_model_name_or_path, subfolder="scheduler"
        )
        print(f"Using dtype: {inferece_dtype}")
        
        # Use FluxKontextPipeline for validation/inference (image-to-image editing)
        self.pipeline = FluxKontextPipeline.from_pretrained(
            self.opt.pretrained_model_name_or_path,
            transformer=transformer,
            vae=self.vae,
            torch_dtype=inferece_dtype,
        )
        self.pipeline.scheduler = noise_scheduler_tmp
        self.pipeline.set_progress_bar_config(disable=False)

        # Memory optimizations to avoid OOM on 80GB A100 (Flux is ~12B params + VAE + text encoders)
        use_cpu_offload = getattr(self.opt.val, "use_cpu_offload", True)
        if use_cpu_offload:
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline.to(self.accelerator.device)
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
            if hasattr(self.pipeline.vae, "enable_tiling"):
                self.pipeline.vae.enable_tiling()
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
        if use_cpu_offload:
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
            if hasattr(self.pipeline.vae, "enable_tiling"):
                self.pipeline.vae.enable_tiling()

        # dataloader = DataLoader(
        #     Subset(self.dataloader.dataset, np.arange(min(1, len(self.dataloader.dataset)))),
        #     shuffle=False,
        #     batch_size=1,
        # )
        # dataloader = self.accelerator.prepare(dataloader)
        # for batch in dataloader:
        #     idx = self.validate_step(batch, idx, self.dataloader.dataset.lq_key, self.dataloader.dataset.gt_key)
        # if hasattr(self.accelerator, "_dataloaders") and dataloader in self.accelerator._dataloaders:
        #     self.accelerator._dataloaders.remove(dataloader)
        for batch in tqdm(self.test_dataloader):
            idx = self.validate_step(batch, idx, self.test_dataloader.dataset.lq_key, self.test_dataloader.dataset.gt_key)
            if idx >= self.max_val_steps:
                break
        del self.pipeline
        gc.collect()
        torch.cuda.empty_cache()
        # After validation, move models back to training dtype/device.
        for model in self.models:
            self._to_device_safe(model, self.accelerator.device)
            model.to(dtype=self.weight_dtype)
            model.train()

    def forwardpass(self, ctrl_pil, prompt, height, width, num_inference_steps):
        """
        Single forward pass through FluxKontextPipeline for validation.

        Returns a tensor in [-1, 1] with shape (1, C, H, W).
        """
        # we are not using image conditioning in the forward pass
        common_kwargs = dict(
            prompt=prompt,
            image=ctrl_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=height,
            width=width,
        )
        try:
            sig = inspect.signature(self.pipeline.__call__)
            valid_keys = set(sig.parameters.keys())
            filtered_kwargs = {k: v for k, v in common_kwargs.items() if k in valid_keys}
        except (TypeError, ValueError):
            filtered_kwargs = common_kwargs

        img = self.pipeline(**filtered_kwargs).images[0]
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
        return img

    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        num_inference_steps = getattr(self.opt.val, "num_inference_steps", 50)
        use_cpu_offload = getattr(self.opt.val, "use_cpu_offload", True)
        
        if self.opt.val.patched:
            b, c, h, w = self.original_size[lq_key]
            pred = []
            for _ in self.setup_patches():
                image_1 = self.sample[lq_key]
                if image_1.shape[1] == 1:
                    image_1 = einops.repeat(image_1, "b 1 h w -> b 3 h w")
                bsz = image_1.shape[0]
                lq_ram = self.model_vlm_transforms(image_1 * 0.5 + 0.5)
                caption = inference(lq_ram.to(dtype=torch.float16), self.model_vlm)
                captions = [c for c in caption]
                out_list = []
                for i in range(bsz):
                    ctrl = (image_1[i].float() * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                    ctrl_pil = Image.fromarray((ctrl * 255).astype(np.uint8))
                    img = self.forwardpass(
                        ctrl_pil,
                        captions[i] if i < len(captions) else "",
                        image_1[i].shape[-2],
                        image_1[i].shape[-1],
                        num_inference_steps,
                    )
                    out_list.append(img)
                    if use_cpu_offload:
                        gc.collect()
                        torch.cuda.empty_cache()
                out = torch.cat(out_list, dim=0).to(image_1.device)
                pred.append(out)
            pred = torch.cat(pred, dim=0)
            pred = einops.rearrange(pred, "(b n) c h w -> b n c h w", b=b)
            out = []
            for i in range(len(pred)):
                merged = merge_patches(pred[i], self.sample[lq_key + "_patched_pos"])
                out.append(merged[..., :h, :w])
            lq1 = self.sample[lq_key + "_original"]
            gt = self.sample[gt_key + "_original"]
            out = torch.stack(out)
        else:
            lq1 = self.sample[lq_key]
            gt = self.sample[gt_key]
            bsz = lq1.shape[0]
            if lq1.shape[1] == 1:
                lq1 = einops.repeat(lq1, "b 1 h w -> b 3 h w")
            lq_ram = self.model_vlm_transforms(lq1 * 0.5 + 0.5)
            caption = inference(lq_ram.to(dtype=torch.float16), self.model_vlm)
            captions = [c for c in caption]
            out_list = []
            for i in range(bsz):
                ctrl = (lq1[i].float() * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                ctrl_pil = Image.fromarray((ctrl * 255).astype(np.uint8))
                img = self.forwardpass(
                    ctrl_pil,
                    captions[i] if i < len(captions) else "",
                    lq1[i].shape[-2],
                    lq1[i].shape[-1],
                    num_inference_steps,
                )
                out_list.append(img)
                if use_cpu_offload:
                    gc.collect()
                    torch.cuda.empty_cache()
            out = torch.cat(out_list, dim=0).to(lq1.device)
        
        lq1 = lq1.cpu().numpy() * 0.5 + 0.5
        gt = gt.cpu().numpy() * 0.5 + 0.5
        out = out.cpu().numpy() * 0.5 + 0.5
        for i in range(len(gt)):
            idx += 1
            image1 = [lq1[i], gt[i], out[i]]
            for j in range(len(image1)):
                if image1[j].shape[0] == 1:
                    image1[j] = einops.repeat(image1[j], "1 h w -> 3 h w")
            # breakpoint()
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f"{idx:04d}", self.global_step)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0, 1), f"out_{idx:04d}", self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
    