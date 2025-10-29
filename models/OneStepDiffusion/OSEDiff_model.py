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
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, EMAModel, T2IAdapter
from peft import LoraConfig
from torchvision import transforms
from pipelines.OSEDiffPipeline import OSEDiffPipeline
from ram import inference_ram as inference

from torch.utils.data import DataLoader, Subset
from models.two_dataset_model import TwoDatasetBasemodel
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

        model.load_state_dict(combined_state_dict)
        # load_model = c.from_pretrained(os.path.join(input_dir, f"{class_name}_{saved[class_name]}"))
        # model.load_state_dict(load_model.state_dict())
        # del load_model
        
def initialize_unet(unet):
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    lora_conf_encoder = LoraConfig(r=4, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=4, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=4, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")        
    return unet

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
    return vae

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

def eps_to_mu(scheduler, model_output, sample, timesteps):
    alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
    alpha_prod_t = alphas_cumprod[timesteps]
    while len(alpha_prod_t.shape) < len(sample.shape):
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample

def encode_prompt(prompt_batch, tokenizer, text_encoder):
    prompt_embeds_list = []
    with torch.no_grad():
        for caption in prompt_batch:
            text_input_ids = tokenizer(
                caption, max_length=tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
            )[0]
            prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
    return prompt_embeds

# todo Extend from OSEDiff_model
class OSEDiff_model(TwoDatasetBasemodel):
    def __init__(self, opt, logger):
        super(OSEDiff_model, self).__init__(opt, logger)
        self.load_handle.remove()
        self.load_handle = self.accelerator.register_load_state_pre_hook(load_model_hook)
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.concatenate_images = self.opt.train.get('concatenate_images', True)
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

        self.lambda_l2 = self.opt.train.lambda_l2
        self.lambda_lpips = self.opt.train.lambda_lpips
        self.cfg_vsd = self.opt.train.cfg_vsd

        self.net_lpips = lpips.LPIPS(net='vgg').cuda()
        self.net_lpips.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(timesteps=[self.opt.train.timestep], device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        # one step prediction. choose t=T
        self.timesteps = self.noise_scheduler.timesteps
        print(f"Timesteps : {self.timesteps}")
        
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision=revision, variant=variant)
        self.unet.train()

        # use adapter for high-frequency injection
        if self.use_adapter:

            t2iadapter_1 = T2IAdapter(
                in_channels=3,
                channels=(320, 640, 1280, 1280),
                num_res_blocks=2,
                downscale_factor=16,
                adapter_type="full_adapter",
            )


        # for VSD loss
        self.noise_scheduler_reg = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.unet_fix = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision=revision, variant=variant)
        self.unet_update = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision=revision, variant=variant)
        self.unet_update.train()
        
        self.unet_fix.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Create EMA for the unet.
        if opt.use_ema:
            #self.ema_unet = EMAModel(self.unet.parameters(), model_cls=UNet2DConditionModel, model_config=self.unet.config)
            print(f"{'='*50} NO EMA")

        if opt.train.lora_finetune:
            self.unet.requires_grad_(False)
            self.unet_update.requires_grad_(False)
            self.vae.requires_grad_(False)

            self.unet = initialize_unet(self.unet)
            self.unet_update = initialize_unet(self.unet_update)
            self.vae = initialize_vae(self.vae)
            
            self.vae.set_adapter(['default_encoder'])
            self.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])
            self.unet_update.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

            for n, _p in self.vae.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
            self.unet.conv_in.requires_grad_(True)
            for n, _p in self.unet.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
            for n, _p in self.unet_update.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
        else:
            self.vae.requires_grad_(True)
            self.unet.requires_grad_(True)
            self.unet_update.requires_grad_(True)
        
        if self.concatenate_images:
            self.vae.encoder.conv_in = torch.nn.Conv2d(4, 128, kernel_size=3, stride=1, padding=1)
            self.vae.encoder.conv_in.requires_grad = True

            
        self.models.append(self.vae)
        self.models.append(self.unet)
        self.models.append(self.unet_update)

        # get captioning model
        self.model_vlm, self.model_vlm_transforms = get_caption_generator(opt.vlm_model_path,
                                                                          dape_path=opt.dape_path)
        self.model_vlm.eval()

        # move fixed models to gpu
        self.model_vlm.to(self.accelerator.device, dtype=torch.float16)
        self.unet_fix.to(self.accelerator.device, dtype=torch.float16)
        
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
        self.gen_params = [p for p in self.unet.parameters() if p.requires_grad]
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
        self.reg_params = [p for p in self.unet_update.parameters() if p.requires_grad]
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

        if self.opt.train.patched:
            self.grids(keys=[lq_key+"_1",lq_key+"_2", gt_key+"_1",gt_key+"_2",], 
                       opt=self.opt.train if is_train else self.opt.val)
    

    def optimize_parameters(self):
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        image_1 = self.sample[lq_key+"_1"]
        image_2 = self.sample[lq_key+"_2"]

        if self.concatenate_images:
            if image_2.shape[1] == 3:
                image_2 = image_2[:, 1:2]
        else:
            image_2 = einops.repeat(image_2, 'b 1 h w -> b 3 h w')
        
        
        bsz = image_1.shape[0]

        gt_1 = self.sample[gt_key+"_1"] if self.use_image1 else self.sample[gt_key+"_2"]
        # gt_2 = self.sample[gt_key+"_2"] 
        image_1 = image_1.clip(-1, 1)
        image_2 = image_2.clip(-1, 1)
        gt_1 = gt_1.clip(-1, 1)
        # gt_2 = gt_2.clip(-1, 1)
        if gt_1.shape[1] == 1:
            gt_1 = einops.repeat(gt_1, 'b 1 h w -> b 3 h w')
 
        gt_ram = self.model_vlm_transforms(gt_1*0.5+0.5)
        caption = inference(gt_ram.to(dtype=torch.float16), self.model_vlm)
        prompt_embeds = encode_prompt([c for c in caption], self.tokenizer, self.text_encoder)
        neg_prompt_embeds = encode_prompt([self.neg_caption]*bsz, self.tokenizer, self.text_encoder)
        # print(image_1.min(), image_1.max(), gt_1.min(), gt_1.max())
        
        # use color cue as starting latent        
        if self.concatenate_images:
            vae_input = torch.cat([image_1, image_2], dim=1)
        else:
            vae_input = image_1 if self.use_image1 else image_2
        latents = self.vae.encode(vae_input).latent_dist.sample() * self.vae.config.scaling_factor

        # Predict the noise residual
        model_pred = self.unet(latents, self.timesteps, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

        # Denoise the latents
        denoised_latents = self.noise_scheduler.step(model_pred, self.timesteps[0], latents, return_dict=True).prev_sample
        output_image = (self.vae.decode(denoised_latents / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        # loss data
        loss_l2 = F.mse_loss(output_image.float(), gt_1.float(), reduction="mean") * self.lambda_l2
        loss_lpips = self.net_lpips(output_image.float(), gt_1.float()).mean() * self.lambda_lpips
        loss_data = loss_l2 + loss_lpips
        
        # loss distribution KL
        timesteps = torch.randint(20, 980, (bsz,), device=denoised_latents.device).long()
        noise = torch.randn_like(denoised_latents)
        noisy_latents = self.noise_scheduler_reg.add_noise(denoised_latents, noise, timesteps)
        with torch.no_grad():
            noise_pred_update = self.unet_update(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.float(),
                ).sample

            x0_pred_update = eps_to_mu(self.noise_scheduler_reg, noise_pred_update, noisy_latents, timesteps)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([timesteps] * 2)
            prompt_embeds_concat = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)

            noise_pred_fix = self.unet_fix(
                noisy_latents_input.to(dtype=torch.float16),
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds_concat.to(dtype=torch.float16),
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
            noise_pred_fix = noise_pred_uncond + self.cfg_vsd * (noise_pred_text - noise_pred_uncond)
            noise_pred_fix.to(dtype=torch.float32)

            x0_pred_fix = eps_to_mu(self.noise_scheduler_reg, noise_pred_fix, noisy_latents, timesteps)

            update_err = F.mse_loss(denoised_latents, x0_pred_update).mean()
            fix_err = F.mse_loss(denoised_latents, x0_pred_fix).mean()

        weighting_factor = torch.abs(denoised_latents - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True)
        grad = -(x0_pred_update - x0_pred_fix) / (weighting_factor + 1e-7) 

        # mask HF in the loss
        if self.use_hf_loss:  
            with torch.no_grad():
                zt_ft = torch.fft.fftshift(torch.fft.fft2(denoised_latents), dim=(-2,-1))
                zt_ft = zt_ft * self.HPF # remove low frequencies
                zt_hp = torch.real(torch.fft.ifft2(torch.fft.ifftshift(zt_ft))).to(latents.dtype)
                mask = zt_hp
                mask = abs(mask).clip(0,self.MASK_CLIP)/self.MASK_CLIP
                grad = grad * mask

        loss_kl = F.mse_loss(denoised_latents, (denoised_latents - grad).detach())

        # calculate total gen loss and update parameters
        loss_gen = loss_data + loss_kl

        self.accelerator.backward(loss_gen)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.gen_params, 1.0)
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()
        
        # loss diff for vsd unet update
        denoised_latents, prompt_embeds = denoised_latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(denoised_latents)
        timesteps = torch.randint(0, self.noise_scheduler_reg.config.num_train_timesteps, (bsz,), device=denoised_latents.device).long()
        noisy_latents = self.noise_scheduler_reg.add_noise(denoised_latents, noise, timesteps)

        noise_pred = self.unet_update(
            noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            ).sample

        loss_d = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        self.accelerator.backward(loss_d)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.reg_params, 1.0)
        self.optimizers[1].step()
        self.optimizers[1].zero_grad()
        return {'all': loss_gen + loss_d, 
                'kl': loss_kl, 
                'l2': loss_l2, 
                'lpips': loss_lpips, 
                'diff': loss_d,
                'update_err': update_err,
                'fix_err': fix_err,
                'output_min': output_image.min(),
                'output_max': output_image.max(),
                'output_mean': output_image.mean(),
                }
    
    @torch.no_grad()
    def validation(self):
        gc.collect()
        torch.cuda.empty_cache()
        idx = 0
        for model in self.models:
            model.eval()
        noise_scheduler_tmp = DDPMScheduler.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="scheduler")
        noise_scheduler_tmp.set_timesteps(timesteps=[self.opt.train.timestep], device='cuda')
        self.pipeline = OSEDiffPipeline(
            self.vae,
            self.unet,
            noise_scheduler_tmp,
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
            # if idx >= 1:
            #     break

        for model in self.models:
            model.train()
            
    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        
        if self.opt.val.patched:
            b,c,h,w = self.original_size[lq_key+"_1"]
            pred = []
            for _ in self.setup_patches():    
                image_1 = self.sample[lq_key+"_1"]
                image_2 = self.sample[lq_key+"_2"]
                if image_2.shape[1] == 3:
                    image_2 = image_2[:, 1:2]
                if not self.use_image1 and not self.concatenate_images:
                    image_2 = einops.repeat(image_2, 'b 1 h w -> b 3 h w')
                bsz = image_1.shape[0]  
                lq_ram = self.model_vlm_transforms(image_1*0.5+0.5)
                caption = inference(lq_ram.to(dtype=torch.float16), self.model_vlm)
                prompt_embeds = encode_prompt(caption, self.tokenizer, self.text_encoder)
                out = self.pipeline(
                    image_1 if self.use_image1 else image_2,
                    image_2 if self.concatenate_images else None,
                    prompt_embeds=prompt_embeds[:bsz],
                    # added_cond_kwargs={k: v[:bsz] for k, v in self.unet_added_conditions.items()},
                    timesteps=[self.opt.train.timestep],
                )
                pred.append(out.images)
            pred = torch.cat(pred, dim=0)
            pred = einops.rearrange(pred, '(b n) c h w -> b n c h w', b=b)
            out = []
            for i in range(len(pred)):
                merged = merge_patches(pred[i], self.sample[lq_key+'_1_patched_pos'])
                out.append(merged[..., :h, :w])
            lq1 = self.sample[lq_key+'_1_original']
            lq2 = self.sample[lq_key+'_2_original']
            gt = self.sample[gt_key+'_1_original'] if self.use_image1 else self.sample[gt_key+'_2_original']
            out = torch.stack(out)
        else: 
            lq1 = self.sample[lq_key+"_1"]
            lq2 = self.sample[lq_key+"_2"]
            if lq2.shape[1] == 3:
                lq2 = lq2[:, 1:2]
            if not self.use_image1 and not self.concatenate_images:
                lq2 = einops.repeat(lq2, 'b 1 h w -> b 3 h w')
            gt = self.sample[gt_key+"_1"] if self.use_image1 else self.sample[gt_key+"_2"]
            bsz = lq1.shape[0]  
            lq_ram = self.model_vlm_transforms(lq1*0.5+0.5)
            caption = inference(lq_ram.to(dtype=torch.float16), self.model_vlm)
            prompt_embeds = encode_prompt([c for c in caption], self.tokenizer, self.text_encoder)
        
            output = self.pipeline(
                lq1 if self.use_image1 else lq2,
                lq2 if self.concatenate_images else None,
                prompt_embeds=prompt_embeds[:bsz],
                # added_cond_kwargs={"text_embeds": prompt_embeds[:bsz], "time_ids": torch.tensor([self.timesteps]*bsz).to(prompt_embeds.device)},
                timesteps=[self.opt.train.timestep],
            )
            out = output.images

        lq1 = lq1.cpu().numpy()*0.5+0.5
        lq2 = lq2.cpu().numpy()*0.5+0.5
        gt = gt.cpu().numpy()*0.5+0.5
        out = out.cpu().numpy()*0.5+0.5
        for i in range(len(gt)):
            idx += 1
            if self.concatenate_images:
                image1 = [lq1[i], lq2[i], gt[i], out[i]]
            else:
                image1 = [lq1[i] if self.use_image1 else lq2[i], gt[i], out[i]]
            for j in range(len(image1)):
                if image1[j].shape[0] == 1:
                    image1[j] = einops.repeat(image1[j], '1 h w -> 3 h w')
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx:04d}', self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
    