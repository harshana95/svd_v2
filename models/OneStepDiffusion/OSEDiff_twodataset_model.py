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
import kornia.augmentation as K
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, EMAModel, T2IAdapter
from peft import LoraConfig
from torchvision import transforms
import torchvision.transforms.functional as TF
from models.OneStepDiffusion.OSEDiff_onedataset_model import OSEDiff_onedataset_model, encode_prompt, eps_to_mu
from pipelines.OSEDiffPipeline import OSEDiffPipeline
from ram import inference_ram as inference

from torch.utils.data import DataLoader, Subset
from models.two_dataset_model import TwoDatasetBasemodel
from utils.dataset_utils import merge_patches
from utils import log_image, log_metrics

from safetensors.torch import load_file

# todo Extend from OSEDiff_model
class OSEDiff_twodataset_model(OSEDiff_onedataset_model, TwoDatasetBasemodel):
    def __init__(self, opt, logger):
        super(OSEDiff_twodataset_model, self).__init__(opt, logger)
        
    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        
        if not is_train:
            permute_channels = self.opt.val.get("permute_channels", None)
            saturation_factor = self.opt.val.get("saturation_factor", 1.0)
            hue_factor = self.opt.val.get("hue_factor", 0.0)
            pixelate_factor = self.opt.val.get("pixelate_factor", 1.0)
            gaussian_blur_pixels = self.opt.val.get("gaussian_blur_pixels", 0.0)

            lq1 = self.sample[lq_key+"_1"]
            lq2 = self.sample[lq_key+"_2"]
            if saturation_factor != 1.0:
                # 0 = Grayscale, 1 = Original, >1 = More saturated
                lq1 = TF.adjust_saturation(lq1*0.5+0.5, saturation_factor=saturation_factor)*2-1
            if hue_factor != 0.0:
                # Range is [-0.5, 0.5]. 
                # 0 = Original, 0.5/-0.5 = Complementary colors (180 degree shift)
                lq1 = TF.adjust_hue(lq1*0.5+0.5, hue_factor=hue_factor)*2-1
            if permute_channels:
                lq1 = lq1[:, permute_channels, :, :]
            self.sample[lq_key+"_1"] = lq1

            if self.opt.val.get("random_elastic_transform", False):
                aug = K.RandomElasticTransform(alpha=(2.0, 2.0), sigma=(10.0, 10.0), p=1.0)
                lq2 = aug(lq2)
            if pixelate_factor != 1.0:
                # Pixelation (Downsample -> Upsample)
                h, w = lq2.shape[-2:]
                small = F.interpolate(lq2, scale_factor=1/pixelate_factor, mode='nearest')
                lq2 = F.interpolate(small, size=(h, w), mode='nearest')
            if gaussian_blur_pixels != 0.0:
                lq2 = TF.gaussian_blur(lq2, kernel_size=[gaussian_blur_pixels, gaussian_blur_pixels])
            self.sample[lq_key+"_2"] = lq2
            
        # # make SR for testing
        # gt_1 = self.sample[gt_key+"_1"]
        # image_1 = F.interpolate(gt_1, size=(128,128), mode='bicubic')
        # image_1 = F.interpolate(image_1, size=(512,512), mode='bicubic')
        # self.sample[lq_key+"_1"] = image_1

        if self.opt.train.patched if is_train else self.opt.val.patched:
            self.grids(keys=[lq_key+"_1",lq_key+"_2", gt_key+"_1",gt_key+"_2",], 
                       opt=self.opt.train if is_train else self.opt.val)
    
    def optimize_parameters(self):
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        image_1 = self.sample[lq_key+"_1"]
        image_2 = self.sample[lq_key+"_2"]
        gt_1 = self.sample[gt_key+"_1"]
        # gt_2 = self.sample[gt_key+"_2"] 

        if image_1.shape[1] == 1:
            image_1 = einops.repeat(image_1, 'b 1 h w -> b 3 h w')
        if image_2.shape[1] == 1:
            image_2 = einops.repeat(image_2, 'b 1 h w -> b 3 h w')
        if gt_1.shape[1] == 1:
            gt_1 = einops.repeat(gt_1, 'b 1 h w -> b 3 h w')
        
        
        bsz = image_1.shape[0]

        image_1 = image_1.clip(-1, 1)
        image_2 = image_2.clip(-1, 1)
        gt_1 = gt_1.clip(-1, 1)
        # gt_2 = gt_2.clip(-1, 1)
 
        gt_ram = self.model_vlm_transforms(image_1*0.5+0.5)
        caption = inference(gt_ram.to(dtype=torch.float16), self.model_vlm)
        prompt_embeds = encode_prompt([c for c in caption], self.tokenizer, self.text_encoder)
        neg_prompt_embeds = encode_prompt([self.neg_caption]*bsz, self.tokenizer, self.text_encoder)
        # print(image_1.min(), image_1.max(), gt_1.min(), gt_1.max())
        
        # use color cue as starting latent        
        if self.concatenate_images:
            vae_input = torch.cat([image_1, image_2], dim=1)
        else:
            vae_input = image_1

        latents = self.vae.encode(vae_input).latent_dist.sample() * self.vae.config.scaling_factor
        if self.use_adapter:
            down_block_additional_residuals = self.t2iadapter(self.adapter_preprocess(image_1, image_2))    
        else:
            down_block_additional_residuals = None

        # Predict the noise residual
        model_pred = self.unet(latents, 
                               self.timesteps, 
                               encoder_hidden_states=prompt_embeds, 
                               down_block_additional_residuals=down_block_additional_residuals, 
                               return_dict=False)[0]

        # Denoise the latents
        denoised_latents = self.noise_scheduler.step(model_pred, self.timesteps[0], latents, return_dict=True).prev_sample
        
        return self._optimize_parameters(denoised_latents, gt_1, prompt_embeds, neg_prompt_embeds)
    
    def forwardpass(self, lq1, lq2):
        if lq1.shape[1] == 1:
            lq1 = einops.repeat(lq1, 'b 1 h w -> b 3 h w')
        if lq2.shape[1] == 1:
            lq2 = einops.repeat(lq2, 'b 1 h w -> b 3 h w')

        bsz = lq1.shape[0]
        lq_ram = self.model_vlm_transforms(lq1 * 0.5 + 0.5)
        caption = inference(lq_ram.to(dtype=torch.float16), self.model_vlm)
        prompt_embeds = encode_prompt([c for c in caption], self.tokenizer, self.text_encoder)

        output = self.pipeline(
            lq1,
            lq2,
            prompt_embeds=prompt_embeds[:bsz],
            timesteps=[self.opt.train.timestep],
        )
        return output.images
    
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
            adapter=self.t2iadapter if self.use_adapter else None,
            adapter_preprocess=self.adapter_preprocess,
            concatenate_images=self.concatenate_images
        )

        self.calculate_flops(torch.randn(1, 3, 512, 512).to(self.device), 
                             torch.randn(1, 3, 512, 512).to(self.device),
                             n=10)
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
            b,c,h,w = self.original_size[lq_key+"_1"]
            pred = []
            for _ in self.setup_patches():    
                image_1 = self.sample[lq_key+"_1"]
                image_2 = self.sample[lq_key+"_2"]
                out = self.forwardpass(image_1, image_2)
                pred.append(out)
            pred = torch.cat(pred, dim=0)
            pred = einops.rearrange(pred, '(b n) c h w -> b n c h w', b=b)
            out = []
            for i in range(len(pred)):
                merged = merge_patches(pred[i], self.sample[lq_key+'_1_patched_pos'])
                out.append(merged[..., :h, :w])
            lq1 = self.sample[lq_key+'_1_original']
            lq2 = self.sample[lq_key+'_2_original']
            gt1 = self.sample[gt_key+'_1_original']
            gt2 = self.sample[gt_key+'_2_original']
            out = torch.stack(out)
        else: 
            lq1 = self.sample[lq_key+"_1"]
            lq2 = self.sample[lq_key+"_2"]
            gt1 = self.sample[gt_key+"_1"]
            gt2 = self.sample[gt_key+"_2"]
            out = self.forwardpass(lq1, lq2)
        adapter_input = None
        if self.use_adapter:
            adapter_input = self.adapter_preprocess(lq1, lq2).cpu().numpy()*0.5+0.5
            if adapter_input.shape[1] == 6:
                adapter_input = einops.rearrange(adapter_input, 'n (c1 c2) h w -> n c1 (h c2) w', c1=3, c2=2)
            if adapter_input.shape[1] == 1:
                adapter_input = einops.repeat(adapter_input, 'n 1 h w -> n 3 h w')
            
        lq1 = lq1.cpu().numpy()*0.5+0.5
        lq2 = lq2.cpu().numpy()*0.5+0.5
        gt1 = gt1.cpu().numpy()*0.5+0.5
        gt2 = gt2.cpu().numpy()*0.5+0.5
        out = out.cpu().numpy()*0.5+0.5
        for i in range(len(gt1)):
            idx += 1
            image1 = [lq1[i], lq2[i], gt1[i], gt2[i], out[i]]
            for j in range(len(image1)):
                if image1[j].shape[0] == 1:
                    image1[j] = einops.repeat(image1[j], '1 h w -> 3 h w')
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(out[i:i+1], 0, 1), f'out_{idx:04d}', self.global_step)
            if adapter_input is not None:
                log_image(self.opt, self.accelerator, np.clip(adapter_input[i:i+1], 0, 1), f'adapter_input_{idx:04d}', self.global_step)
            log_metrics(gt1[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)

            if not self.is_train:
                log_image(self.opt, self.accelerator, np.clip(lq1[i:i+1],0,1),f'lq1_{idx:04d}', self.global_step)
                log_image(self.opt, self.accelerator, np.clip(lq2[i:i+1],0,1),f'lq2_{idx:04d}', self.global_step)
                log_image(self.opt, self.accelerator, np.clip(gt1[i:i+1],0,1),f'gt_{idx:04d}', self.global_step)
        return idx
    