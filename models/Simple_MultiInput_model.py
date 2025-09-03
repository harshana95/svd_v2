# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import einops
import numpy as np
import torch

from models.archs import define_network
from models.two_dataset_model import TwoDatasetBasemodel
from utils.dataset_utils import merge_patches
from utils.loss import Loss
from utils import log_image, log_metrics

class Simple2Dataset_model(TwoDatasetBasemodel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt, logger):
        super(Simple2Dataset_model, self).__init__(opt, logger)

        # define network
        self.net_g = define_network(opt.network)
        self.models.append(self.net_g)

        if self.is_train:
            self.init_training_settings()

        # setup other parameters

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)  

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        if self.opt.train.patched:
            self.grids(keys=[lq_key+"_1",lq_key+"_2", gt_key+"_1",gt_key+"_2"], 
                       opt=self.opt.train if is_train else self.opt.val)

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        img1 = self.sample[lq_key+"_1"]
        img2 = self.sample[lq_key+"_2"]
        gt = self.sample[gt_key+"_1"]

        preds = self.net_g(img1, img2)
        losses = self.criterion(preds, gt)
        self.accelerator.backward(losses['all'])
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        return losses

    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        if self.opt.val.patched:
            b,c,h,w = self.original_size[lq_key + "_1"]
            pred = []
            for _ in self.setup_patches():
                img1 = self.sample[lq_key+"_1"]
                img2 = self.sample[lq_key+"_2"]
                out = self.net_g(img1, img2)
                pred.append(out)
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
            out = self.net_g(img1, img2)

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
            log_image(self.opt, self.accelerator, image1, f'{idx}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx}', self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
    
    
        