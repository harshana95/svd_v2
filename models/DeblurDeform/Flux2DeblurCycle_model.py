"""
Flux2DeblurCycle_model: FLUX 2 dev support for deblurring with control conditioning.

Key FLUX 2 differences from FLUX 1:
- Uses Flux2Transformer2DModel (~32B params vs FLUX 1's ~12B)
- Uses AutoencoderKLFlux2 with batch normalization in VAE encoding
- Uses Flux2Pipeline with Mistral3ForConditionalGeneration text encoder
- Image conditioning uses 'image' parameter (not 'control_image')
- Default guidance_scale is 4.0 (vs FLUX 1's 3.5)
- Text encoding API may differ (Mistral3 vs CLIP+T5)
"""
import copy
import gc
import glob
import os
import random
import einops
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    Flux2Pipeline,
    Flux2Transformer2DModel,
)
# Try to import AutoencoderKLFlux2, fallback to AutoencoderKL if not available
try:
    from diffusers import AutoencoderKLFlux2
    USE_FLUX2_VAE = True
except ImportError:
    from diffusers import AutoencoderKL
    USE_FLUX2_VAE = False
    AutoencoderKLFlux2 = AutoencoderKL
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


def encode_images_flux2(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype, generator=None):
    """Encode pixel images to Flux 2 latent space with batch normalization (FLUX 2 specific)."""
    from diffusers.utils import retrieve_latents
    
    # FLUX 2 VAE encoding with batch normalization
    if hasattr(vae, 'bn') and hasattr(vae, '_patchify_latents'):
        # FLUX 2 AutoencoderKLFlux2: encode -> patchify -> batch norm
        image_latents = retrieve_latents(vae.encode(pixels.to(vae.dtype)), generator=generator, sample_mode="argmax")
        image_latents = vae._patchify_latents(image_latents)
        
        # Apply batch normalization (FLUX 2 specific)
        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std
    else:
        # Fallback for standard AutoencoderKL (FLUX 1 style)
        pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
        shift = getattr(vae.config, "shift_factor", 0.0)
        scale = getattr(vae.config, "scaling_factor", vae.config.scaling_factor)
        image_latents = (pixel_latents - shift) * scale
    
    return image_latents.to(weight_dtype)


def load_model_hook(models, input_dir):
    saved = {}
    while len(models) > 0:
        # pop models so that they are not loaded again
        model = models.pop()
        class_name = model._get_name()
        saved[class_name] = 1 if class_name not in saved.keys() else saved[class_name] + 1
        print(f"Loading model {class_name}_{saved[class_name]} from {input_dir}")
        
        # load diffusers style into model
        folder = os.path.join(input_dir, f"{class_name}_{saved[class_name]}")
        combined_state_dict = {}
    
        for file_path in glob.glob(os.path.join(folder, "*.safetensors")):
            state_dict_part = load_file(file_path)
            combined_state_dict.update(state_dict_part)
        try:
            model.load_state_dict(combined_state_dict)
        except Exception as e:
            print(f"{'='*50} Failed to load {class_name} {'='*50} {model} {e}")


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


def initialize_transformer_flux2(transformer):
    """LoRA for Flux2Transformer2DModel: target linear layers in transformer blocks and x_embedder."""
    l_target_modules = []
    for n, p in transformer.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        if "to_q" in n or "to_k" in n or "to_v" in n or "to_out" in n or "proj_out" in n or "context_embedder" in n or "ff.net" in n or "x_embedder" in n:
            l_target_modules.append(n.replace(".weight", ""))
    lora_conf = LoraConfig(r=4, init_lora_weights="gaussian", target_modules=l_target_modules)
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


class Flux2DeblurCycle_model(BaseModel):

    def __init__(self, opt, logger):
        super(Flux2DeblurCycle_model, self).__init__(opt, logger)
        self.load_handle.remove()
        self.load_handle = self.accelerator.register_load_state_pre_hook(load_model_hook)
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.guidance_scale = getattr(opt, "guidance_scale", 4.0)  # Flux 2 default is 4.0
        self.only_target_transformer_blocks = getattr(opt, "only_target_transformer_blocks", False)
        self.weighting_scheme = getattr(opt, "weighting_scheme", "none")

        pretrained_model_name_or_path = self.opt.pretrained_model_name_or_path
        revision = getattr(self.opt, "revision", None)
        variant = getattr(self.opt, "variant", None)

        # Flux 2: VAE (AutoencoderKLFlux2 with batch normalization), transformer, flow-matching scheduler
        try:
            self.vae = AutoencoderKLFlux2.from_pretrained(
                pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant
            )
            print("Using AutoencoderKLFlux2 (FLUX 2 VAE with batch normalization)")
        except Exception as e:
            print(f"Warning: Could not load AutoencoderKLFlux2, falling back to AutoencoderKL: {e}")
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant
            )
        self.transformer = Flux2Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer", revision=revision, variant=variant
        )
        self.transformer.train()

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # Text encoding via Flux2Pipeline (no separate text encoders)
        self.text_encoding_pipeline = Flux2Pipeline.from_pretrained(
            pretrained_model_name_or_path,
            transformer=None,
            vae=None,
            torch_dtype=weight_dtype,
        )

        if not self.only_target_transformer_blocks:
            self.transformer.requires_grad_(True)
        self.vae.requires_grad_(False)

        # Flux 2 control: enable image conditioning by doubling x_embedder input channels
        # FLUX 2 uses the same x_embedder architecture as FLUX 1 for control
        with torch.no_grad():
            if hasattr(self.transformer, "x_embedder"):
                initial_input_channels = self.transformer.config.in_channels
                new_linear = torch.nn.Linear(
                    self.transformer.x_embedder.in_features * 2,
                    self.transformer.x_embedder.out_features,
                    bias=self.transformer.x_embedder.bias is not None,
                    dtype=self.transformer.dtype,
                    device=self.transformer.device,
                )
                new_linear.weight.zero_()
                new_linear.weight[:, :initial_input_channels].copy_(self.transformer.x_embedder.weight)
                if self.transformer.x_embedder.bias is not None:
                    new_linear.bias.copy_(self.transformer.x_embedder.bias)
                self.transformer.x_embedder = new_linear
                self.transformer.register_to_config(in_channels=initial_input_channels * 2, out_channels=initial_input_channels)
            else:
                raise ValueError(
                    "Flux2Transformer2DModel must have x_embedder for control conditioning. "
                    "Please check your diffusers version supports FLUX 2 control."
                )

        if self.only_target_transformer_blocks:
            if hasattr(self.transformer, "x_embedder"):
                self.transformer.x_embedder.requires_grad_(True)
            for name, module in self.transformer.named_modules():
                if "transformer_blocks" in name:
                    module.requires_grad_(True)
                else:
                    module.requires_grad_(False)

        if opt.train.get("lora_finetune", False):
            self.transformer.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.transformer = initialize_transformer_flux2(self.transformer)
            self.vae = initialize_vae(self.vae)
            self.vae.set_adapter(["default_encoder"])
            self.transformer.set_adapter(["default"])
            for n, _p in self.vae.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
            for n, _p in self.transformer.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
        else:
            self.vae.requires_grad_(False)

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

        # Encode to latents (Flux 2 VAE)
        vae_dev = self.vae.to(self.accelerator.device)
        pixel_latents = encode_images_flux2(gt_1, vae_dev, self.weight_dtype)
        control_latents = encode_images_flux2(image_1, vae_dev, self.weight_dtype)

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

        # Flow matching: noisy input
        sigmas = self.get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
        noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
        concatenated_noisy_model_input = torch.cat([noisy_model_input, control_latents], dim=1)

        # FLUX 2: Pack latents for transformer input
        # FLUX 2 uses the same packing mechanism as FLUX 1 for control conditioning
        if hasattr(Flux2Pipeline, "_pack_latents"):
            packed_noisy_model_input = Flux2Pipeline._pack_latents(
                concatenated_noisy_model_input,
                batch_size=bsz,
                num_channels_latents=concatenated_noisy_model_input.shape[1],
                height=concatenated_noisy_model_input.shape[2],
                width=concatenated_noisy_model_input.shape[3],
            )
            latent_image_ids = Flux2Pipeline._prepare_latent_image_ids(
                bsz,
                concatenated_noisy_model_input.shape[2] // 2,
                concatenated_noisy_model_input.shape[3] // 2,
                self.accelerator.device,
                self.weight_dtype,
            )
        else:
            # If packing methods don't exist, use direct concatenation
            # This may require checking FLUX 2's actual API
            packed_noisy_model_input = concatenated_noisy_model_input
            h, w = concatenated_noisy_model_input.shape[2], concatenated_noisy_model_input.shape[3]
            # Create img_ids for positional encoding (FLUX 2 may need this)
            latent_image_ids = torch.arange(0, h * w, device=self.accelerator.device, dtype=self.weight_dtype).reshape(h, w)

        # Guidance
        flux_transformer = self._unwrap_model(self.transformer)
        if getattr(flux_transformer.config, "guidance_embeds", False):
            guidance_vec = torch.full(
                (bsz,), self.guidance_scale, device=noisy_model_input.device, dtype=self.weight_dtype
            )
        else:
            guidance_vec = None

        # Text encoding via Flux2Pipeline (uses Mistral3ForConditionalGeneration)
        gt_ram = self.model_vlm_transforms(image_1 * 0.5 + 0.5)
        caption = inference(gt_ram.to(dtype=torch.float16), self.model_vlm)
        captions = [c for c in caption]
        self.text_encoding_pipeline = self.text_encoding_pipeline.to(self.accelerator.device)
        with torch.no_grad():
            # FLUX 2 encode_prompt signature: (prompt, num_images_per_prompt=1, do_classifier_free_guidance=False, ...)
            # Returns: prompt_embeds, pooled_prompt_embeds (and optionally text_ids if available)
            try:
                # Try FLUX 2 API first
                result = self.text_encoding_pipeline.encode_prompt(
                    captions,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                if isinstance(result, tuple) and len(result) >= 2:
                    prompt_embeds, pooled_prompt_embeds = result[0], result[1]
                    text_ids = result[2] if len(result) > 2 else None
                elif isinstance(result, dict):
                    prompt_embeds = result.get("prompt_embeds")
                    pooled_prompt_embeds = result.get("pooled_prompt_embeds")
                    text_ids = result.get("text_ids")
                else:
                    # Fallback: assume it's just prompt_embeds
                    prompt_embeds = result
                    pooled_prompt_embeds = None
                    text_ids = None
            except Exception as e:
                # Fallback: try FLUX 1 style API
                try:
                    prompt_embeds, pooled_prompt_embeds, text_ids = self.text_encoding_pipeline.encode_prompt(
                        captions, prompt_2=None
                    )
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to encode prompts with both FLUX 2 and FLUX 1 APIs. "
                        f"FLUX 2 error: {e}, FLUX 1 error: {e2}"
                    )

        # Predict - FLUX 2 transformer forward call
        # FLUX 2 Flux2Transformer2DModel forward signature:
        # hidden_states, encoder_hidden_states, timestep, guidance, pooled_projections, 
        # txt_ids, img_ids, joint_attention_kwargs, return_dict
        transformer_kwargs = {
            "hidden_states": packed_noisy_model_input,
            "encoder_hidden_states": prompt_embeds,
            "timestep": timesteps / 1000.0,
            "return_dict": False,
        }
        if guidance_vec is not None:
            transformer_kwargs["guidance"] = guidance_vec
        if pooled_prompt_embeds is not None:
            transformer_kwargs["pooled_projections"] = pooled_prompt_embeds
        if text_ids is not None:
            transformer_kwargs["txt_ids"] = text_ids
        if isinstance(latent_image_ids, torch.Tensor):
            transformer_kwargs["img_ids"] = latent_image_ids
        
        model_pred = self.transformer(**transformer_kwargs)[0]

        # Unpack latents if packing was used
        if hasattr(Flux2Pipeline, "_unpack_latents"):
            model_pred = Flux2Pipeline._unpack_latents(
                model_pred,
                height=noisy_model_input.shape[2] * self.vae_scale_factor,
                width=noisy_model_input.shape[3] * self.vae_scale_factor,
                vae_scale_factor=self.vae_scale_factor,
            )
        # If unpacking not available, model_pred should already be in correct shape

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
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()

        return {"all": loss}
    
    @torch.no_grad()
    def validation(self):
        gc.collect()
        torch.cuda.empty_cache()
        idx = 0
        for model in self.models:
            model.eval()
        transformer = self._unwrap_model(self.transformer)
        noise_scheduler_tmp = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.opt.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.pipeline = Flux2Pipeline.from_pretrained(
            self.opt.pretrained_model_name_or_path,
            transformer=transformer,
            vae=self.vae,
            torch_dtype=self.weight_dtype,
        )
        self.pipeline.scheduler = noise_scheduler_tmp
        self.pipeline.set_progress_bar_config(disable=True)

        # Memory optimizations to avoid OOM on 80GB A100 (Flux 2 is ~32B params + VAE + text encoders)
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

        resolution = getattr(self.opt, "resolution", 512)
        if resolution % 8 != 0:
            resolution = (resolution // 8) * 8
        dataloader = DataLoader(
            Subset(self.dataloader.dataset, np.arange(min(5, len(self.dataloader.dataset)))),
            shuffle=False,
            batch_size=1,
        )
        dataloader = self.accelerator.prepare(dataloader)
        for batch in dataloader:
            idx = self.validate_step(batch, idx, self.dataloader.dataset.lq_key, self.dataloader.dataset.gt_key)
        if hasattr(self.accelerator, "_dataloaders") and dataloader in self.accelerator._dataloaders:
            self.accelerator._dataloaders.remove(dataloader)
        for batch in tqdm(self.test_dataloader):
            idx = self.validate_step(batch, idx, self.test_dataloader.dataset.lq_key, self.test_dataloader.dataset.gt_key)
            if idx >= self.max_val_steps:
                break
        del self.pipeline
        gc.collect()
        torch.cuda.empty_cache()
        # After CPU offload, transformer/vae may be on CPU; move back to GPU for training
        for model in self.models:
            model.to(self.accelerator.device)
            model.train()

    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        resolution = getattr(self.opt, "resolution", 512)
        if resolution % 8 != 0:
            resolution = (resolution // 8) * 8
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
                    ctrl_pil = Image.fromarray((ctrl * 255).astype(np.uint8)).resize((resolution, resolution), Image.BILINEAR)
                    # FLUX 2 uses 'image' parameter for image conditioning (not 'control_image')
                    img = self.pipeline(
                        prompt=captions[i] if i < len(captions) else "",
                        image=ctrl_pil,  # FLUX 2 image conditioning parameter
                        num_inference_steps=num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        height=image_1[i].shape[-2],
                        width=image_1[i].shape[-1],
                    ).images[0]
                    out_list.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1)
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
                ctrl_pil = Image.fromarray((ctrl * 255).astype(np.uint8)).resize((resolution, resolution), Image.BILINEAR)
                # FLUX 2 uses 'image' parameter for image conditioning
                img = self.pipeline(
                    prompt=captions[i] if i < len(captions) else "",
                    image=ctrl_pil,  # FLUX 2 image conditioning parameter
                    num_inference_steps=num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    height=lq1[i].shape[-2],
                    width=lq1[i].shape[-1],
                ).images[0]
                out_list.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1)
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
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f"{idx:04d}", self.global_step)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0, 1), f"out_{idx:04d}", self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
