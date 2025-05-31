import einops
import numpy as np
import torch
from tqdm import tqdm

from models.Simple_model import Simple_model
from utils.dataset_utils import merge_patches
from utils.loss import Loss
from utils import log_image, log_metrics

class HF_model(Simple_model):
    """Base Deblur model for single image deblur with High-Frequency information"""

    def __init__(self, opt, logger):
        super(HF_model, self).__init__(opt, logger)

    def feed_data(self, data, is_train=True):
        self.sample = data
        if is_train:
            if self.opt.train.patched:
                lq_key = self.dataloader.dataset.lq_key 
                gt_key = self.dataloader.dataset.gt_key
                self.grids(keys=[lq_key, gt_key, lq_key + "_hf"], opt=self.opt.train)
        else:
            if self.opt.val.patched:
                lq_key = self.test_dataloader.dataset.lq_key 
                gt_key = self.test_dataloader.dataset.gt_key
                self.grids(keys=[lq_key, gt_key, lq_key + "_hf"], opt=self.opt.val)

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        lq = self.sample[self.dataloader.dataset.lq_key]
        lqhf = self.sample[self.dataloader.dataset.lq_key + "_hf"]
        preds = self.net_g(torch.cat([lq, lqhf], dim=1))
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
                lq = self.sample[lq_key]
                lqhf = self.sample[lq_key + "_hf"]
                out = self.net_g(torch.cat([lq, lqhf], dim=1))
                pred.append(out)
            pred = torch.cat(pred, dim=0)
            pred = einops.rearrange(pred, '(b n) c h w -> b n c h w', b=b)
            out = []
            for i in range(len(pred)):
                merged = merge_patches(pred[i], self.sample[lq_key+'_patched_pos'])
                out.append(merged[..., :h, :w])
            lq = self.sample[lq_key+'_original']
            lqhf = self.sample[lq_key+"_hf_original"]
            gt = self.sample[gt_key+'_original']
            out = torch.stack(out)
        else: 
            lq = self.sample[lq_key]
            lqhf = self.sample[lq_key + "_hf"]
            gt = self.sample[gt_key]
            out = self.net_g(torch.cat([lq, lqhf], dim=1))

        lq = lq.cpu().numpy()
        lqhf = lqhf.cpu().numpy()
        gt = gt.cpu().numpy()
        out = out.cpu().numpy()
        for i in range(len(gt)):
            idx += 1
            image1 = [lq[i], lqhf[i], gt[i], out[i]]
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx}', self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
