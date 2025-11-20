# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import einops
import numpy as np
import torch
from tqdm import tqdm
import gc
from torch.utils.data import DataLoader, Subset
from deeplens.geolens import GeoLens

from models.archs import define_network
from models.base_model import BaseModel
from utils.dataset_utils import merge_patches
from utils.loss import Loss
from utils import log_image, log_metrics

class SimpleSynthetic_model(BaseModel):
    def __init__(self, opt, logger):
        super(SimpleSynthetic_model, self).__init__(opt, logger)

        self.m_scale = 1e3  # This converts depth to mm
        self.psf_rescale_factor = 1

        # define network
        self.net_g = define_network(opt.network)
        self.models.append(self.net_g)

        if self.is_train:
            self.init_training_settings()

        # setup other parameters
        settings = opt.deeplens
        self.single_wavelength = settings.get("single_wavelength", False)
        self.spp = settings.get("spp", 100000)
        self.kernel_size = settings.get("kernel_size", 65)

        self.depth_min = -settings.depth_min * self.m_scale
        self.depth_max = -settings.depth_max * self.m_scale
        self.fov = settings.fov
        self.foc_d_arr = np.array(
            [
                -400,
                -425,
                -450,
                -500,
                -550,
                -600,
                -650,
                -700,
                -800,
                -900,
                -1000,
                -1250,
                -1500,
                -1750,
                -2000,
                -2500,
                -3000,
                -4000,
                -5000,
                -6000,
                -8000,
                -10000,
                -12000,
                -15000,
                -20000,
            ]
        )
        # normalize focal distance [0, 1]
        self.foc_z_arr = (self.foc_d_arr - self.depth_min) / (self.depth_max - self.depth_min)

        # initialize optics
        self.wavelength_set_m = np.array(settings.wavelength_set_m)  # if we are using hsi/rgb images, this should change
        self.lens = GeoLens(filename=settings.lens_file, device=self.accelerator.device)
        # self.lens = HybridLens(filename=settings.lens_file, device=self.accelerator.device)
        
        # lens.refocus(foc_dist=-1000)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)    

    def depth2z(self, depth):
        z = (depth - self.depth_min) / (self.depth_max - self.depth_min)
        z = torch.clamp(z, min=0, max=1)
        return z

    def z2depth(self, z):
        depth = z * (self.depth_max - self.depth_min) + self.depth_min
        return depth
    
    def feed_data(self, data, is_train=True):
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key

        # foc_z = float(np.random.choice(self.foc_z_arr))
        # foc_dist = self.z2depth(foc_z) # mm
        # self.lens.refocus(foc_dist)

        with torch.no_grad():
            img_rendered = self.lens.render(
                data[gt_key]*0.5+0.5,
                depth=10000.0,          # 10 meter
                spp=512,                # Samples per pixel
                method='psf_patch'
            )
        data[lq_key] = img_rendered*2-1

        self.sample = data
        if self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.train if is_train else self.opt.val)

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
            
        preds = self.net_g(self.sample[self.dataloader.dataset.lq_key])
        losses = self.criterion(preds, self.sample[self.dataloader.dataset.gt_key])
        self.accelerator.backward(losses['all'])
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        return losses

    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        if self.opt.val.patched:
            b,c,h,w = self.original_size[lq_key]
            pred = []
            for _ in self.setup_patches():        
                out = self.net_g(self.sample[lq_key])
                pred.append(out)
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
            out = self.net_g(lq)
            
        lq = lq.cpu().numpy()*0.5+0.5
        gt = gt.cpu().numpy()*0.5+0.5
        out = out.cpu().numpy()*0.5+0.5
        for i in range(len(gt)):
            idx += 1
            image1 = [lq[i], gt[i], out[i]]
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx:04d}', self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
    
    
        