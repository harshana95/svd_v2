# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import einops
import numpy as np
import torch
from torch import nn

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
            self.lambda_adv = opt.train.get("lambda_adv", 1)
            
        if self.is_train:
            self.init_training_settings()

        # setup other parameters

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)    
        if self.discriminator is not None:
            # initialize GAN loss for discriminator and generator
            self.criterionGAN = nn.BCELoss() # GANLoss('original').to(self.accelerator.device)

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        if self.opt.train.patched if is_train else self.opt.val.patched:
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
        optimizer_g = torch.optim.Adam([{'params': optim_params}],
            lr=train_opt.optim.learning_rate,
            betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
            weight_decay=train_opt.optim.adam_weight_decay,
            eps=train_opt.optim.adam_epsilon,
        )
        self.optimizers.append(optimizer_g)

        # Discriminator optim params
        d_optim_params = []
        for k, v in self.discriminator.named_parameters():
            if v.requires_grad:
                d_optim_params.append(v)
            else:
                self.logger.warning(f'Params {k} will not be optimized.')
        optimizer_d = torch.optim.Adam([{'params': d_optim_params}],
            lr=train_opt.optim.learning_rate,
            betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
            weight_decay=train_opt.optim.adam_weight_decay,
            eps=train_opt.optim.adam_epsilon,
        )
        self.optimizers.append(optimizer_d)

    def optimize_parameters(self):
        lq = self.sample[self.dataloader.dataset.lq_key]
        gt = self.sample[self.dataloader.dataset.gt_key]

        # If no discriminator, follow base flow
        if self.discriminator is None:
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            preds = self.net_g(lq)
            losses = self.criterion(preds, gt)
            self.accelerator.backward(losses['all'])
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            for i in range(len(self.optimizers)):
                self.optimizers[i].step()
            return losses

        ### ---------------------------
        ### Step 1: Train Discriminator
        ### ---------------------------
        # 1a. Train on Real Data
        # We want the Discriminator to output 1 (Real) for real images
        self.optimizers[1].zero_grad() # dicriminator optimizer is at 1 index
        real_preds = self.discriminator(torch.cat([lq, gt], dim=1)) 
        loss_d_real = self.criterionGAN(real_preds, torch.ones_like(real_preds)) # Labels = 1

        # 1b. Train on Fake Data
        fake_images = self.net_g(lq)

        # We want the Discriminator to output 0 (Fake) for generated images
        # NOTE: .detach() is crucial here! We don't want to update Generator weights yet.
        fake_preds = self.discriminator(torch.cat([lq, fake_images.detach()], dim=1))
        loss_d_fake = self.criterionGAN(fake_preds, torch.zeros_like(fake_preds)) # Labels = 0

        # 1c. Backprop and Update Discriminator
        loss_d = (loss_d_real + loss_d_fake) / 2
        self.accelerator.backward(loss_d)
        # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.01)
        self.optimizers[1].step() 

        ### ---------------------------
        ### Step 2: Train Generator
        ### ---------------------------

        # We want the Generator to generate images that the Discriminator thinks are Real (1).
        self.optimizers[0].zero_grad()
                
        # Re-run discriminator on the *same* fake images (but this time, we keep the gradients connected)
        preds = self.discriminator(torch.cat([lq, fake_images], dim=1))
        
        # TRICK: We use 'real_targets' (1) as the label. 
        # If D says "1", G has succeeded.
        loss_g = self.criterionGAN(preds, torch.ones_like(preds))  # labels = 1
        
        losses = self.criterion(fake_images, gt)
        loss_pix = losses['all']
        loss_gen = loss_pix + loss_g*self.lambda_adv
        self.accelerator.backward(loss_gen)
        # torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizers[0].step()

        # # With discriminator -> separate G and D updates
        # # Generator update
        # lq = self.sample[self.dataloader.dataset.lq_key]
        # gt = self.sample[self.dataloader.dataset.gt_key]
        # self.net_g.train()
        # self.optimizer_g.zero_grad()
        # preds = self.net_g(lq)
        # losses = self.criterion(preds, gt)

        # pred_fake, pred_real = self.discriminate(lq, preds, gt)
        # G_gan_loss = self.criterionGAN(pred_fake, target_is_real=True, for_discriminator=False)
        # losses['GAN'] = G_gan_loss
        # l_total = losses['all'] + G_gan_loss
        # self.accelerator.backward(l_total)
        # torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        # self.optimizer_g.step()

        # # Discriminator update
        # self.optimizer_d.zero_grad()
        # # generate fake image without computing gradients for generator
        # with torch.no_grad():
        #     lq = lq.detach()
        #     gt = gt.detach()
        #     fake_image = self.net_g(lq)
        #     fake_image = fake_image.detach()
        #     fake_image.requires_grad_()
        # # compute discriminator outputs for fake and real pairs
        # pred_fake, pred_real = self.discriminate(lq, fake_image, gt)
        # losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        # losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)
        # D_loss = (losses['D_Fake'] + losses['D_real']).mean()
        # self.accelerator.backward(D_loss)
        # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.01)
        # self.optimizer_d.step()

        # merge losses for logging
        losses['gen_pix'] = loss_pix
        losses['gen_adv'] = loss_g*self.lambda_adv
        losses['gen'] = loss_gen
        losses['disc'] = loss_d
        losses['disc_fake'] = loss_d_fake
        losses['disc_real'] = loss_d_real
        losses['all'] = loss_g + loss_d
        return losses

    # def divide_pred(self, pred):
    #     # split predictions into fake and real for multiscale outputs
    #     fake = pred[:pred.size(0) // 2]
    #     real = pred[pred.size(0) // 2:]
    #     return fake, real

    # def discriminate(self, input_semantics, fake_image, real_image):
    #     # concatenate semantics and fake/real image and forward through discriminator
    #     fake_concat = torch.cat([input_semantics, fake_image], dim=1)
    #     real_concat = torch.cat([input_semantics, real_image], dim=1)
    #     fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
    #     discriminator_out = self.discriminator(fake_and_real)
    #     pred_fake, pred_real = self.divide_pred(discriminator_out)
    #     return pred_fake, pred_real

    def forwardpass(self, lq):
        return self.net_g(lq)

    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        self.calculate_flops(torch.randn(1, 3, 512, 512).to(self.device))
        
        if self.opt.val.patched:
            b,c,h,w = self.original_size[lq_key]
            pred = []
            for _ in self.setup_patches():        
                out = self.forwardpass(self.sample[lq_key])
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
            out = self.forwardpass(lq)
            
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
    
    
        