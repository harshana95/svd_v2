import functools
import gc
import glob
import importlib
import os
import random
import einops
import numpy as np
import torch
import lpips
from tqdm import tqdm
import torch.nn.functional as F
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers import (
    AutoencoderKL,
    SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    EMAModel,
)
from peft import LoraConfig
from torchvision import transforms
from pipelines.OSEDiffPipeline_SD3 import OSEDiffPipelineSD3
from ram import inference_ram as inference

# SD3 uses flow matching: x0 from (z_t, model_output, sigma) is x0 = z_t - sigma * model_output
def flow_to_x0(scheduler, model_output, sample, timesteps):
    """Convert flow-matching model output to predicted x0 (denoised sample). Batched timesteps supported."""
    schedule_timesteps = scheduler.timesteps.to(device=sample.device)
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    step_indices = [scheduler.index_for_timestep(t.item(), schedule_timesteps) for t in timesteps]
    sigmas = scheduler.sigmas.to(device=sample.device, dtype=sample.dtype)
    sigma = sigmas[step_indices]
    if sigma.dim() == 1:
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)
    return sample - sigma * model_output

from torch.utils.data import DataLoader, Subset
from models.base_model import BaseModel
from utils.dataset_utils import merge_patches
from utils import log_image, log_metrics

from safetensors.torch import load_file

def load_model_hook(models, input_dir):
    saved = {}
    while len(models) > 0:
        # pop models so that they are not loaded again
        model = models.pop()
        class_name = model._get_name()
        saved[class_name] = 1 if class_name not in saved.keys() else saved[class_name] + 1
        print(f"Loading model {class_name}_{saved[class_name]} from {input_dir}")
        # try:
        #     c = find_attr(_arch_modules, class_name)
        #     assert c is not None
        # except ValueError as e:  # class is not written by us. Try to load from diffusers
        #     print(f"Class {class_name} not found in archs. Trying to load from diffusers...")
        #     m = importlib.import_module('diffusers') # load the module, will raise ImportError if module cannot be loaded
        #     c = getattr(m, class_name)  # get the class, will raise AttributeError if class cannot be found    
        
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
        
        # load_model = c.from_pretrained(os.path.join(input_dir, f"{class_name}_{saved[class_name]}"))
        # model.load_state_dict(load_model.state_dict())
        # del load_model
        

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


def initialize_transformer_sd3(transformer):
    """LoRA for SD3Transformer2DModel: target linear layers in transformer blocks and projections."""
    l_target_modules = []
    for n, p in transformer.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        if "to_q" in n or "to_k" in n or "to_v" in n or "to_out" in n or "proj_out" in n or "context_embedder" in n or "ff.net" in n:
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


def encode_prompt_sd3(
    prompt_batch,
    tokenizer,
    tokenizer_2,
    tokenizer_3,
    text_encoder,
    text_encoder_2,
    text_encoder_3,
    device,
    max_sequence_length=256,
    tokenizer_max_length=77,
):
    """Encode prompts using SD3's three text encoders (2x CLIP + T5). Returns (prompt_embeds, pooled_prompt_embeds)."""
    batch_size = len(prompt_batch)
    prompt = [p for p in prompt_batch]

    # CLIP 1
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embed_1 = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
    pooled_1 = prompt_embed_1[0]
    prompt_embed_1 = prompt_embed_1.hidden_states[-2].to(dtype=text_encoder.dtype)

    # CLIP 2
    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embed_2 = text_encoder_2(text_inputs_2.input_ids.to(device), output_hidden_states=True)
    pooled_2 = prompt_embed_2[0]
    prompt_embed_2 = prompt_embed_2.hidden_states[-2].to(dtype=text_encoder_2.dtype)

    clip_prompt_embeds = torch.cat([prompt_embed_1, prompt_embed_2], dim=-1)

    # T5
    text_inputs_3 = tokenizer_3(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    t5_embeds = text_encoder_3(text_inputs_3.input_ids.to(device))[0]

    clip_prompt_embeds = F.pad(
        clip_prompt_embeds, (0, t5_embeds.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_embeds], dim=-2)
    pooled_prompt_embeds = torch.cat([pooled_1, pooled_2], dim=-1)

    return prompt_embeds, pooled_prompt_embeds

class OSEDiff_SD3_onedataset_model(BaseModel):
    def __init__(self, opt, logger):
        super(OSEDiff_SD3_onedataset_model, self).__init__(opt, logger)
        self.load_handle.remove()
        self.load_handle = self.accelerator.register_load_state_pre_hook(load_model_hook)
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.concatenate_images = self.opt.train.get('concatenate_images', False)
        self.use_image1 = self.opt.train.get('use_image1', True)
        self.use_hf_loss = self.opt.train.get('use_hf_loss', False)
        self.use_adapter = self.opt.train.get('use_adapter', False)

        if self.use_hf_loss:
            self.MASK_CLIP = 5e-1
            hf_noise_ratio = 1.0
            hf_noise_power = 1.0
            hf_noise_piston = 0.0
            # build weight map for FT transformed image
            res = opt.image_resolution[0] // 8 # //8 because we are adding noise in latent space
            nr = hf_noise_ratio
            x = torch.linspace(-1, 1, res, device='cuda') * nr
            y = torch.linspace(-1, 1, res, device='cuda') * nr
            grid_x, grid_y = torch.meshgrid(x, y) # Create 2D coordinate grids
            radial_distances = torch.sqrt(grid_x**2 + grid_y**2) # Calculate radial distances
            radial_distances = radial_distances**hf_noise_power
            radial_distances = (radial_distances + hf_noise_piston).clip(0, 1)  # more than 1 should be clipped
            self.HPF = radial_distances.unsqueeze(0).unsqueeze(0)
            
        self.neg_caption = "painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"

        pretrained_model_name_or_path = self.opt.pretrained_model_name_or_path
        revision = self.opt.revision
        variant = self.opt.variant
        print(f"Using {pretrained_model_name_or_path} {revision} {variant}")

        self.lambda_l2 = self.opt.train.lambda_l2
        self.lambda_lpips = self.opt.train.lambda_lpips
        self.lambda_kl = self.opt.train.get('lambda_kl', 1.0)
        self.cfg_vsd = self.opt.train.cfg_vsd

        self.net_lpips = lpips.LPIPS(net='vgg').cuda()
        self.net_lpips.requires_grad_(False)

        # SD3 does not use T2IAdapter in the same way; adapter support skipped for SD3
        self.use_adapter = False
        # if self.use_adapter and opt.train.hpf_adapter_input:
        #     self.adapter_preprocess = hpf_adapter_input
        # else:
        #     self.adapter_preprocess = lambda x: x

        # Stable Diffusion 3 Medium: 3 text encoders (2x CLIP + T5), transformer, flow-matching scheduler
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
        self.tokenizer_3 = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_3")
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2").cuda()
        self.text_encoder_3 = T5EncoderModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_3").cuda()

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(timesteps=[float(self.opt.train.timestep)], device="cuda")
        self.timesteps = self.noise_scheduler.timesteps
        print(f"Timesteps (SD3 flow matching): {self.timesteps}")

        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant)
        self.transformer = SD3Transformer2DModel.from_pretrained(pretrained_model_name_or_path, subfolder="transformer", revision=revision, variant=variant)
        self.transformer.train()

        self.tokenizer_max_length = getattr(self.tokenizer, "model_max_length", 77)


        # for VSD loss (flow matching)
        self.noise_scheduler_reg = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.transformer_fix = SD3Transformer2DModel.from_pretrained(pretrained_model_name_or_path, subfolder="transformer", revision=revision, variant=variant)
        self.transformer_update = SD3Transformer2DModel.from_pretrained(pretrained_model_name_or_path, subfolder="transformer", revision=revision, variant=variant)
        self.transformer_update.train()

        self.transformer_fix.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.text_encoder_3.requires_grad_(False)

        # Create EMA for the transformer.
        if opt.use_ema:
            print(f"{'='*50} NO EMA")

        if opt.train.lora_finetune:
            self.transformer.requires_grad_(False)
            self.transformer_update.requires_grad_(False)
            self.vae.requires_grad_(False)

            self.transformer = initialize_transformer_sd3(self.transformer)
            self.transformer_update = initialize_transformer_sd3(self.transformer_update)
            self.vae = initialize_vae(self.vae)

            self.vae.set_adapter(['default_encoder'])
            self.transformer.set_adapter(['default'])
            self.transformer_update.set_adapter(['default'])

            for n, _p in self.vae.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
            for n, _p in self.transformer.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
            for n, _p in self.transformer_update.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
        else:
            self.vae.requires_grad_(True)
            self.transformer.requires_grad_(True)
            self.transformer_update.requires_grad_(True)

        if self.concatenate_images:
            self.vae.encoder.conv_in = torch.nn.Conv2d(6, 128, kernel_size=3, stride=1, padding=1)
            self.vae.encoder.conv_in.requires_grad = True

        self.models.append(self.vae)
        self.models.append(self.transformer)
        self.models.append(self.transformer_update)

        # get captioning model
        self.model_vlm, self.model_vlm_transforms = get_caption_generator(opt.vlm_model_path,
                                                                          dape_path=opt.dape_path)
        self.model_vlm.eval()

        # move fixed models to gpu
        self.model_vlm.to(self.accelerator.device, dtype=torch.float16)
        self.transformer_fix.to(self.accelerator.device, dtype=torch.float16)
        
        for m in self.models:
            print(m.__class__.__name__)
            all_param = 0
            trainable_params = 0
            for name, param in m.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    # print(name)
            print(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )
        
    def setup_optimizers(self):
        opt = self.opt.train.optim

        # Optimizer creation for generator
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

        # Optimizer creation for VSD
        optimizer_class = torch.optim.AdamW
        self.reg_params = [p for p in self.transformer_update.parameters() if p.requires_grad]
        optimizer_vsd = optimizer_class(
            self.reg_params,
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2),
            weight_decay=opt.adam_weight_decay,
            eps=opt.adam_epsilon,
            )
        self.optimizers.append(optimizer_vsd)


    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        
        # # make SR for testing
        # gt_1 = self.sample[gt_key+"_1"]
        # image_1 = F.interpolate(gt_1, size=(128,128), mode='bicubic')
        # image_1 = F.interpolate(image_1, size=(512,512), mode='bicubic')
        # self.sample[lq_key+"_1"] = image_1

        if self.opt.train.patched if is_train else self.opt.val.patched:
            self.grids(keys=[lq_key, gt_key], 
                       opt=self.opt.train if is_train else self.opt.val)
    
    def _decode_latents(self, latents):
        """Decode latents with VAE; SD3 may use shift_factor."""
        latents = latents / self.vae.config.scaling_factor
        if getattr(self.vae.config, "shift_factor", None) is not None:
            latents = latents + self.vae.config.shift_factor
        return self.vae.decode(latents, return_dict=False)[0].clamp(-1, 1)

    def _optimize_parameters(self, denoised_latents, gt_1, prompt_embeds, pooled_prompt_embeds, neg_prompt_embeds, neg_pooled_prompt_embeds):
        bsz = denoised_latents.shape[0]
        output_image = self._decode_latents(denoised_latents)

        # loss data
        loss_l2 = F.mse_loss(output_image.float(), gt_1.float(), reduction="mean") * self.lambda_l2
        loss_lpips = self.net_lpips(output_image.float(), gt_1.float()).mean() * self.lambda_lpips
        loss_data = self.lambda_l2*loss_l2 + self.lambda_lpips * loss_lpips

        if self.lambda_kl == 0.0:
            self.accelerator.backward(loss_data)
            if self.accelerator.sync_gradients:
                # Use stricter gradient clipping for stability
                self.accelerator.clip_grad_norm_(self.gen_params, 0.5)
            self.optimizers[0].step()
            self.optimizers[0].zero_grad()
            return {'all': loss_data, 
                'l2': loss_l2, 
                'lpips': loss_lpips, 
            }
        
        # loss distribution KL (flow matching: scale_noise, flow_to_x0)
        schedule_timesteps = self.noise_scheduler_reg.timesteps.to(denoised_latents.device)
        step_indices = torch.randint(20, len(schedule_timesteps) - 20, (bsz,), device=denoised_latents.device)
        timesteps_t = schedule_timesteps[step_indices]
        noise = torch.randn_like(denoised_latents)
        noisy_latents = self.noise_scheduler_reg.scale_noise(denoised_latents, timesteps_t, noise)

        with torch.no_grad():
            noise_pred_update = self.transformer_update(
                hidden_states=noisy_latents,
                timestep=timesteps_t,
                encoder_hidden_states=prompt_embeds.float(),
                pooled_projections=pooled_prompt_embeds.float(),
                return_dict=False,
            )[0]

            x0_pred_update = flow_to_x0(self.noise_scheduler_reg, noise_pred_update, noisy_latents, timesteps_t)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([timesteps_t, timesteps_t], dim=0)
            prompt_embeds_concat = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            pooled_concat = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

            noise_pred_fix = self.transformer_fix(
                hidden_states=noisy_latents_input.to(dtype=torch.float16),
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds_concat.to(dtype=torch.float16),
                pooled_projections=pooled_concat.to(dtype=torch.float16),
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
            noise_pred_fix = noise_pred_uncond + self.cfg_vsd * (noise_pred_text - noise_pred_uncond)
            noise_pred_fix = noise_pred_fix.to(dtype=torch.float32)

            x0_pred_fix = flow_to_x0(self.noise_scheduler_reg, noise_pred_fix, noisy_latents, timesteps_t)

            # update_err = F.mse_loss(denoised_latents, x0_pred_update).mean()
            # fix_err = F.mse_loss(denoised_latents, x0_pred_fix).mean()

        weighting_factor = torch.abs(denoised_latents - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True)
        # Clip weighting factor to prevent division by very small values and gradient explosion
        weighting_factor = torch.clamp(weighting_factor, min=1e-4)
        grad = -(x0_pred_update - x0_pred_fix) / weighting_factor
        # Clip grad to prevent extreme values that can cause training instability
        grad = torch.clamp(grad, min=-10.0, max=10.0)
        # Check for NaN/Inf and replace with zeros to prevent training collapse
        grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad)) 

        # mask HF in the loss
        if self.use_hf_loss:  
            with torch.no_grad():
                # zt_ft = torch.fft.fftshift(torch.fft.fft2(denoised_latents), dim=(-2,-1))
                # zt_ft = zt_ft * self.HPF # remove low frequencies
                # zt_hp = torch.real(torch.fft.ifft2(torch.fft.ifftshift(zt_ft))).to(denoised_latents.dtype)
                # mask = zt_hp
                # mask = abs(mask).clip(0,self.MASK_CLIP)/self.MASK_CLIP
                # grad = grad * mask  # this should be convolution, not .*

                grad_ft = torch.fft.fftshift(torch.fft.fft2(grad), dim=(-2,-1))
                grad_ft = grad_ft * self.HPF # remove low frequencies
                grad_hp = torch.real(torch.fft.ifft2(torch.fft.ifftshift(grad_ft))).to(denoised_latents.dtype)
                grad = grad_hp

        # we use this method to calculate loss because grad is computed inside torch.no_grad to save memory.
        # if not we have to compute the gradients of transformer_fix and transformer_update
        target_latents = (denoised_latents - grad).detach()
        # Ensure target latents are finite to prevent NaN loss
        target_latents = torch.where(torch.isfinite(target_latents), target_latents, denoised_latents.detach())
        loss_kl = F.mse_loss(denoised_latents, target_latents)

        # calculate total gen loss and update parameters
        loss_gen = loss_data + self.lambda_kl*loss_kl

        self.accelerator.backward(loss_gen)
        if self.accelerator.sync_gradients:
            # Use stricter gradient clipping for stability
            self.accelerator.clip_grad_norm_(self.gen_params, 0.5)
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()
        
        # loss diff for vsd transformer update (flow matching)
        denoised_latents = denoised_latents.detach()
        prompt_embeds = prompt_embeds.detach()
        pooled_prompt_embeds = pooled_prompt_embeds.detach()
        noise = torch.randn_like(denoised_latents)
        schedule_timesteps = self.noise_scheduler_reg.timesteps.to(denoised_latents.device)
        step_indices = torch.randint(0, len(schedule_timesteps), (bsz,), device=denoised_latents.device)
        timesteps_t = schedule_timesteps[step_indices]
        noisy_latents = self.noise_scheduler_reg.scale_noise(denoised_latents, timesteps_t, noise)

        noise_pred = self.transformer_update(
            hidden_states=noisy_latents,
            timestep=timesteps_t,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

        loss_d = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        self.accelerator.backward(loss_d)
        if self.accelerator.sync_gradients:
            # Use stricter gradient clipping for VSD loss stability
            self.accelerator.clip_grad_norm_(self.reg_params, 0.5)
        self.optimizers[1].step()
        self.optimizers[1].zero_grad()
        return {'all': loss_gen + loss_d, 
                'kl': loss_kl, 
                'l2': loss_l2, 
                'lpips': loss_lpips, 
                'diff': loss_d,
                # 'update_err': update_err,
                # 'fix_err': fix_err,
                # 'output_min': output_image.min(),
                # 'output_max': output_image.max(),
                # 'output_mean': output_image.mean(),
                }
    
    def optimize_parameters(self):
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        image_1 = self.sample[lq_key]
        gt_1 = self.sample[gt_key]
        
        bsz = image_1.shape[0]

        image_1 = image_1.clip(-1, 1)
        gt_1 = gt_1.clip(-1, 1)

        if image_1.shape[1] == 1:
            image_1 = einops.repeat(image_1, 'b 1 h w -> b 3 h w')
        if gt_1.shape[1] == 1:
            gt_1 = einops.repeat(gt_1, 'b 1 h w -> b 3 h w')
 
        gt_ram = self.model_vlm_transforms(image_1*0.5+0.5)
        caption = inference(gt_ram.to(dtype=torch.float16), self.model_vlm)
        prompt_embeds, pooled_prompt_embeds = encode_prompt_sd3(
            [c for c in caption],
            self.tokenizer,
            self.tokenizer_2,
            self.tokenizer_3,
            self.text_encoder,
            self.text_encoder_2,
            self.text_encoder_3,
            self.accelerator.device,
            tokenizer_max_length=self.tokenizer_max_length,
        )
        neg_prompt_embeds, neg_pooled_prompt_embeds = encode_prompt_sd3(
            [self.neg_caption] * bsz,
            self.tokenizer,
            self.tokenizer_2,
            self.tokenizer_3,
            self.text_encoder,
            self.text_encoder_2,
            self.text_encoder_3,
            self.accelerator.device,
            tokenizer_max_length=self.tokenizer_max_length,
        )

        vae_input = image_1
        latents = self.vae.encode(vae_input).latent_dist.sample() * self.vae.config.scaling_factor

        t = self.timesteps[0].expand(latents.shape[0]).to(latents.device)
        model_pred = self.transformer(
            hidden_states=latents,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        
        denoised_latents = self.noise_scheduler.step(model_pred, self.timesteps[0], latents, return_dict=True).prev_sample
        self.noise_scheduler._step_index = 0 # roll back scheduler step_index

        return self._optimize_parameters(
            denoised_latents, gt_1,
            prompt_embeds, pooled_prompt_embeds,
            neg_prompt_embeds, neg_pooled_prompt_embeds,
        )

    def forwardpass(self, lq1):
        if lq1.shape[1] == 1:
            lq1 = einops.repeat(lq1, 'b 1 h w -> b 3 h w')

        bsz = lq1.shape[0]
        lq_ram = self.model_vlm_transforms(lq1 * 0.5 + 0.5)
        caption = inference(lq_ram.to(dtype=torch.float16), self.model_vlm)
        prompt_embeds, pooled_prompt_embeds = encode_prompt_sd3(
            [c for c in caption],
            self.tokenizer, self.tokenizer_2, self.tokenizer_3,
            self.text_encoder, self.text_encoder_2, self.text_encoder_3,
            self.accelerator.device,
            tokenizer_max_length=self.tokenizer_max_length,
        )
        output = self.pipeline(
            lq1,
            None,
            prompt_embeds=prompt_embeds[:bsz],
            pooled_prompt_embeds=pooled_prompt_embeds[:bsz],
            timesteps=[float(self.opt.train.timestep)],
        )
        return output.images
        
    
    @torch.no_grad()
    def validation(self):
        gc.collect()
        torch.cuda.empty_cache()
        
        idx = 0
        for model in self.models:
            model.eval()
        noise_scheduler_tmp = FlowMatchEulerDiscreteScheduler.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="scheduler")
        noise_scheduler_tmp.set_timesteps(timesteps=[float(self.opt.train.timestep)], device="cuda")
        self.pipeline = OSEDiffPipelineSD3(
            self.vae,
            self.transformer,
            noise_scheduler_tmp,
            concatenate_images=self.concatenate_images
        )
        dataloader = DataLoader(Subset(self.dataloader.dataset, np.arange(5)), 
                                shuffle=False, 
                                batch_size=1)
        print(f"Tesing using {len(dataloader)} training data...")
        dataloader = self.accelerator.prepare(dataloader)
        for batch in dataloader:
            idx = self.validate_step(batch, idx, self.dataloader.dataset.lq_key, self.dataloader.dataset.gt_key)
        self.accelerator._dataloaders.remove(dataloader)
        for batch in tqdm(self.test_dataloader):
            idx = self.validate_step(batch, idx, self.test_dataloader.dataset.lq_key, self.test_dataloader.dataset.gt_key)
            if idx >= self.max_val_steps:
                break

        for model in self.models:
            model.train()
            
    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        
        if self.opt.val.patched:
            b,c,h,w = self.original_size[lq_key]
            pred = []
            for _ in self.setup_patches():    
                image_1 = self.sample[lq_key]
                out = self.forwardpass(image_1)
                pred.append(out)
            pred = torch.cat(pred, dim=0)
            pred = einops.rearrange(pred, '(b n) c h w -> b n c h w', b=b)
            out = []
            for i in range(len(pred)):
                merged = merge_patches(pred[i], self.sample[lq_key+'_patched_pos'])
                out.append(merged[..., :h, :w])
            lq1 = self.sample[lq_key+'_original']
            gt = self.sample[gt_key+'_original']
            out = torch.stack(out)
        else: 
            lq1 = self.sample[lq_key]
            gt = self.sample[gt_key]
            out = self.forwardpass(lq1)

        lq1 = lq1.cpu().numpy()*0.5+0.5
        gt = gt.cpu().numpy()*0.5+0.5
        out = out.cpu().numpy()*0.5+0.5
        for i in range(len(gt)):
            idx += 1
            image1 = [lq1[i], gt[i], out[i]]
            for j in range(len(image1)):
                if image1[j].shape[0] == 1:
                    image1[j] = einops.repeat(image1[j], '1 h w -> 3 h w')
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx:04d}', self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
    