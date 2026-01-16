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
from models.base_model import BaseModel
from models.archs.related.LD.loss import GANLoss
from utils.dataset_utils import merge_patches
from utils.loss import Loss
from utils import log_image, log_metrics

class Simple_model(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt, logger):
        super(Simple_model, self).__init__(opt, logger)

        # define network
        self.net_g = define_network(opt.network)
        self.models.append(self.net_g)

        # define discriminator if defined in yml
        self.discriminator = None
        if opt.discriminator is not None:
            self.discriminator = define_network(opt.discriminator)
            self.models.append(self.discriminator)
            
        if self.is_train:
            self.init_training_settings()

        # setup other parameters

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)    
        if self.discriminator is not None:
            # initialize GAN loss for discriminator and generator
            self.criterionGAN = GANLoss('hinge').to(self.accelerator.device)

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        if self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.train if is_train else self.opt.val)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # If there is no discriminator, use base implementation
        if self.discriminator is None:
            super(Simple_model, self).setup_optimizers()
            return

        # Generator optim params
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                self.logger.warning(f'Params {k} will not be optimized.')
        self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
            lr=train_opt.optim.learning_rate,
            betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
            weight_decay=train_opt.optim.adam_weight_decay,
            eps=train_opt.optim.adam_epsilon,
        )
        self.optimizers.append(self.optimizer_g)

        # Discriminator optim params
        d_optim_params = []
        for k, v in self.discriminator.named_parameters():
            if v.requires_grad:
                d_optim_params.append(v)
            else:
                self.logger.warning(f'Params {k} will not be optimized.')
        self.optimizer_d = torch.optim.Adam([{'params': d_optim_params}],
            lr=train_opt.optim.learning_rate,
            betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
            weight_decay=train_opt.optim.adam_weight_decay,
            eps=train_opt.optim.adam_epsilon,
        )
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self):
        # If no discriminator, follow base flow
        if self.discriminator is None:
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            preds = self.net_g(self.sample[self.dataloader.dataset.lq_key])
            losses = self.criterion(preds, self.sample[self.dataloader.dataset.gt_key])
            self.accelerator.backward(losses['all'])
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            for i in range(len(self.optimizers)):
                self.optimizers[i].step()
            return losses

        # With discriminator -> separate G and D updates
        # Generator update
        lq = self.sample[self.dataloader.dataset.lq_key]
        gt = self.sample[self.dataloader.dataset.gt_key]
        self.net_g.train()
        self.optimizer_g.zero_grad()
        preds = self.net_g(lq)
        losses = self.criterion(preds, gt)
        pred_fake, pred_real = self.discriminate(lq, preds, gt)
        G_gan_loss = self.criterionGAN(pred_fake, True, for_discriminator=False)
        losses['GAN'] = G_gan_loss
        l_total = losses['all'] + G_gan_loss
        self.accelerator.backward(l_total)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        # Discriminator update
        self.optimizer_d.zero_grad()
        # generate fake image without computing gradients for generator
        with torch.no_grad():
            lq = lq.detach()
            gt = gt.detach()
            fake_image = self.net_g(lq)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        # compute discriminator outputs for fake and real pairs
        pred_fake, pred_real = self.discriminate(lq, fake_image, gt)
        losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)
        D_loss = (losses['D_Fake'] + losses['D_real']).mean()
        self.accelerator.backward(D_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.01)
        self.optimizer_d.step()

        # merge losses for logging
        losses['G_loss'] = l_total
        losses['D_loss'] = D_loss
        losses['all'] = l_total + D_loss
        return losses

    def divide_pred(self, pred):
        # split predictions into fake and real for multiscale outputs
        fake = pred[:pred.size(0) // 2]
        real = pred[pred.size(0) // 2:]
        return fake, real

    def discriminate(self, input_semantics, fake_image, real_image):
        # concatenate semantics and fake/real image and forward through discriminator
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.discriminator(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

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
            
        lq = lq.cpu().numpy()
        gt = gt.cpu().numpy()
        out = out.cpu().numpy()
        for i in range(len(gt)):
            idx += 1
            image1 = [lq[i], gt[i], out[i]]
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx:04d}', self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
    
    
        