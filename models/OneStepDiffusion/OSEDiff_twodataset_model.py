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
from models.OneStepDiffusion.OSEDiff_onedataset_model import OSEDiff_onedataset_model, encode_prompt, eps_to_mu
from pipelines.OSEDiffPipeline import OSEDiffPipeline, preprocess_adapter_input
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
        if self.use_adapter:
            down_block_additional_residuals = self.t2iadapter(preprocess_adapter_input(image_2))
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
        adapter_input = preprocess_adapter_input(lq2).cpu().numpy()*0.5+0.5
        adapter_input = einops.repeat(adapter_input, 'n 1 h w -> n 3 h w')
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
            log_image(self.opt, self.accelerator, np.clip(out[i:i+1], 0, 1), f'out_{idx:04d}', self.global_step)
            log_image(self.opt, self.accelerator, np.clip(adapter_input[i:i+1], 0, 1), f'adapter_input_{idx:04d}', self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
    