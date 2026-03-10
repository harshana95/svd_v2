import einops
import numpy as np
import torch
from torch import nn

from models.archs import define_network
from models.Simple_model import Simple_model
from utils.dataset_utils import merge_patches
from utils.loss import Loss
from utils import log_image, log_metrics

class SimpleTurb_model(Simple_model):
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
        real_preds = self.discriminator(torch.cat([gt], dim=1)) 
        loss_d_real = self.criterionGAN(real_preds, torch.ones_like(real_preds)) # Labels = 1

        # 1b. Train on Fake Data
        fake_images = self.net_g(lq)

        # We want the Discriminator to output 0 (Fake) for generated images
        # NOTE: .detach() is crucial here! We don't want to update Generator weights yet.
        fake_preds = self.discriminator(torch.cat([fake_images.detach()], dim=1))
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
        preds = self.discriminator(torch.cat([fake_images], dim=1))
        
        # TRICK: We use 'real_targets' (1) as the label. 
        # If D says "1", G has succeeded.
        loss_g = self.criterionGAN(preds, torch.ones_like(preds))  # labels = 1
        
        losses = self.criterion(fake_images, gt)
        loss_pix = losses['all']
        loss_gen = loss_pix + loss_g*self.lambda_adv
        self.accelerator.backward(loss_gen)
        # torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizers[0].step()

        # merge losses for logging
        losses['gen_pix'] = loss_pix
        losses['gen_adv'] = loss_g*self.lambda_adv
        losses['gen'] = loss_gen
        losses['disc'] = loss_d
        losses['disc_fake'] = loss_d_fake
        losses['disc_real'] = loss_d_real
        losses['all'] = loss_g + loss_d
        return losses
