# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import einops
import numpy as np
import torch

from models.SimpleLatent_model import SimpleLatent_model
from models.two_dataset_model import TwoDatasetBasemodel
from utils.dataset_utils import merge_patches
from utils.loss import Loss
from utils import log_image, log_metrics


class SimpleLatent2Dataset_model(SimpleLatent_model, TwoDatasetBasemodel):
    
    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        img1 = self.sample[lq_key+"_1"]
        img2 = self.sample[lq_key+"_2"]
        
        if img1.shape[1] == 1:
            img1 = einops.repeat(img1, 'b 1 h w -> b 3 h w')
        if img2.shape[1] == 1:
            img2 = einops.repeat(img2, 'b 1 h w -> b 3 h w')
        self.sample[lq_key+"_1"] = img1
        self.sample[lq_key+"_2"] = img2
        if self.opt.train.patched if is_train else self.opt.val.patched:
            self.grids(keys=[lq_key+"_1",lq_key+"_2", gt_key+"_1",gt_key+"_2"], 
                       opt=self.opt.train if is_train else self.opt.val)

    def optimize_parameters(self):
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key
        img1 = self.sample[lq_key+"_1"]
        img2 = self.sample[lq_key+"_2"]
        gt = self.sample[gt_key+"_1"]
        
        img1 = self.vae.encode(img1).latent_dist.sample() * self.vae.config.scaling_factor
        img2 = self.vae.encode(img2).latent_dist.sample() * self.vae.config.scaling_factor
        gt = self.vae.encode(gt).latent_dist.sample() * self.vae.config.scaling_factor

        self.sample[lq_key] = torch.concat([img1, img2], dim=1)
        self.sample[gt_key] = gt
        
        return super(SimpleLatent_model, self).optimize_parameters()
    
    def forwardpass(self, lq1, lq2):
        return self.net_g(torch.concat([self.vae.encode(lq1).latent_dist.sample() * self.vae.config.scaling_factor, 
                                        self.vae.encode(lq2).latent_dist.sample() * self.vae.config.scaling_factor], dim=1)/self.vae.config.scaling_factor)
        
    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        self.calculate_flops(torch.randn(1, 3, 512, 512).to(self.device), torch.randn(1, 3, 512, 512).to(self.device))
        
        if self.opt.val.patched:
            b,c,h,w = self.original_size[lq_key + "_1"]
            pred = []
            for _ in self.setup_patches():
                _out = self.forwardpass(self.sample[lq_key+"_1"], self.sample[lq_key+"_2"])
                _out = self.vae.decode(_out).sample
                pred.append(_out)
            pred = torch.cat(pred, dim=0)
            pred = einops.rearrange(pred, '(b n) c h w -> b n c h w', b=b)
            out = []
            for i in range(len(pred)):
                merged = merge_patches(pred[i], self.sample[lq_key+'_1_patched_pos'])
                out.append(merged[..., :h, :w])
            img1 = self.sample[lq_key+'_1_original']
            img2 = self.sample[lq_key+'_2_original']
            gt = self.sample[gt_key+'_1_original']
            out = torch.stack(out)
        else: 
            img1 = self.sample[lq_key+"_1"]
            img2 = self.sample[lq_key+"_2"]
            gt = self.sample[gt_key+"_1"]
            out = self.forwardpass(img1, img2)
            out = self.vae.decode(out).sample

        img1 = img1.cpu().numpy()
        img2 = img2.cpu().numpy()
        gt = gt.cpu().numpy()
        out = out.cpu().numpy()
        for i in range(len(gt)):
            idx += 1
            image1 = [img1[i], img2[i], gt[i], out[i]]
            for j in range(len(image1)):
                if image1[j].shape[0] == 1:
                    image1[j] = einops.repeat(image1[j], '1 h w -> 3 h w')
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx:04d}', self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
    
    
        