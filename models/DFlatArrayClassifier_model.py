import einops
import numpy as np
import torch
from models.DFlatArrayIR_model import DFlatArrayIR_model
from utils import log_image, log_metrics


class DFlatArrayClassifier_model(DFlatArrayIR_model):

    def __init__(self, opt, logger):
        super(DFlatArrayClassifier_model, self).__init__(opt, logger)
        self.label_key = self.opt.datasets.train.label_key
        self.n_classes = self.opt.network.n_classes
    
    def prepare_label(self, label):
        return torch.nn.functional.one_hot(label, num_classes=self.n_classes).to(torch.float32)

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key
        label = self.sample[self.label_key]
        label = self.prepare_label(label)

        pred = self.net_g(self.sample[lq_key])
        losses = self.criterion(pred, label)
        
        self.accelerator.backward(losses['all'])
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        return losses

    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        if idx==0:
            for i in range(len(self.all_psf_intensity)):
                psf_intensity = self.all_psf_intensity[i]
                for j in range(psf_intensity.shape[2]):
                    image1 = [psf_intensity[0,0,j].detach().cpu().numpy()]
                    image1 = np.stack(image1)
                    image1 = image1/image1.max()
                    image1 = np.clip(image1, 0, 1)
                    ps_loc = self.ps_locs[j]
                    log_image(self.opt, self.accelerator, image1, f"MS{i}_psf_{j}_{ps_loc}_{ps_loc + self.MS_pos[i]}", self.global_step)
                

        if self.opt.val.patched:
            raise Exception("Patched validation not supported for DFlatArrayClassifier_model")
        else: 
            lq = self.sample[lq_key]
            gt = self.sample[gt_key]
            pred = self.net_g(lq)
            label = self.sample[self.label_key]
            label = self.prepare_label(label)

        lq = lq.cpu().numpy()
        gt = gt.cpu().numpy()
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()

        log_metrics(label, pred, self.opt.val.metrics, self.accelerator, self.global_step)
        for i in range(len(gt)):
            idx += 1
            lq_i = einops.rearrange(lq[i], '(n1 n2) c h w -> c (n1 h) (n2 w)', n1=self.array_size[0], n2=self.array_size[1])
            image1 = [lq_i, gt[i]]

            for j in range(len(image1)):
                if image1[j].shape[0] == 1:
                    image1[j] = einops.repeat(image1[j], '1 h w -> 3 h w')
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx}', self.global_step)  # image format (N,C,H,W)
            
            if self.opt.val.max_images is not None and idx > self.opt.val.max_images:
                break
        return idx
