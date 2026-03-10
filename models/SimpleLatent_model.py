from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import einops
import numpy as np
import torch
from models.Simple_model import Simple_model
from utils.dataset_utils import merge_patches
from utils import log_image, log_metrics

class SimpleLatent_model(Simple_model):
    def __init__(self, opt, logger):
        super(SimpleLatent_model, self).__init__(opt, logger)
        pretrained_model_name_or_path = self.opt.pretrained_model_name_or_path
        revision = self.opt.revision
        variant = self.opt.variant
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(self.accelerator.device)

        
    def optimize_parameters(self):
        lq = self.sample[self.dataloader.dataset.lq_key]
        gt = self.sample[self.dataloader.dataset.gt_key]

        self.sample[self.dataloader.dataset.lq_key] = self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor
        self.sample[self.dataloader.dataset.gt_key] = self.vae.encode(gt).latent_dist.sample() * self.vae.config.scaling_factor

        return super(SimpleLatent_model, self).optimize_parameters()
    
    def forwardpass(self, lq):
        return self.net_g(self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor)

    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        self.calculate_flops(torch.randn(1, 3, 512, 512).to(self.device))
        
        if self.opt.val.patched:
            b,c,h,w = self.original_size[lq_key]
            pred = []
            for _ in self.setup_patches():
                _out = self.forwardpass(self.sample[lq_key])
                _out = self.vae.decode(_out/self.vae.config.scaling_factor).sample
                pred.append(_out)
            pred = torch.cat(pred, dim=0)
            pred = einops.rearrange(pred, '(b n) c h w -> b n c h w', b=b)
            out = []
            for i in range(len(pred)):
                merged = merge_patches(pred[i], self.sample[lq_key+'_patched_pos'])
                out.append(merged[..., :h, :w])
            lq = self.sample[lq_key+'_original']
            gt = self.sample[gt_key+'_original']
            out = torch.stack(out)
        else: 
            lq = self.sample[lq_key]
            gt = self.sample[gt_key]
            out = self.forwardpass(lq)
            out = self.vae.decode(out/self.vae.config.scaling_factor).sample
            
            
        lq = lq.cpu().numpy()*0.5+0.5
        gt = gt.cpu().numpy()*0.5+0.5
        out = out.cpu().numpy()*0.5+0.5
        for i in range(len(gt)):
            idx += 1
            image1 = [lq[i, :3],gt[i], out[i]]
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            # log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([gt[i]]), 0,1), f'gt_{idx:04d}', self.global_step)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx:04d}', self.global_step)
            log_image(self.opt, self.accelerator, np.clip(np.stack([lq[i, :3]]), 0,1), f'lq_{idx:04d}', self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
    
    
        