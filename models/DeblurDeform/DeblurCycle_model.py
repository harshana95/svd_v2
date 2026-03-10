from diffusers import AutoencoderKL, T2IAdapter
import einops
import numpy as np
import torch
from torch import nn

from models.archs import define_network
from models.archs.Encoder_arch import Encoder_arch, Encoder_config
from models.base_model import BaseModel
from utils.dataset_utils import merge_patches
from utils.loss import Loss
from utils import log_image, log_metrics
from utils.dataset_utils import crop_arr



class DeblurCycle_model(BaseModel):
    def __init__(self, opt, logger):
        super(DeblurCycle_model, self).__init__(opt, logger)
        self.n_channels = 3
        self.vae = None
        if opt.vae_name_or_path:
            self.vae = AutoencoderKL.from_pretrained(opt.vae_name_or_path, subfolder=opt.vae_subfolder)
            self.vae.requires_grad_(False)
            self.vae.eval()
            self.vae.to(self.accelerator.device)

            self.n_channels = int(self.vae.latent_channels)
        
        # define network
        self.reblur_net = define_network(opt.reblur_network)
        self.deblur_network = define_network(opt.deblur_network)
        self.models.append(self.reblur_net)
        self.models.append(self.deblur_network)
        
        # define discriminator if defined in yml
        self.discriminator = None
        if opt.discriminator is not None:
            self.discriminator = define_network(opt.discriminator)
            self.models.append(self.discriminator)
            self.lambda_adv = opt.train.get("lambda_adv", 1)

        self.lambda_deblur = opt.train.get("lambda_deblur", 1)
        self.lambda_reblur = opt.train.get("lambda_reblur", 1)
        
        if self.is_train:
            self.init_training_settings()


    def init_training_settings(self):
        train_opt = self.opt['train']
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)    
        
        self.deblur_network.train()
        self.reblur_net.train()
                
        if self.discriminator is not None:
            self.discriminator.train()
            self.criterionGAN = nn.BCELoss() # GANLoss('original').to(self.accelerator.device)

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        
        if self.opt.train.patched if is_train else self.opt.val.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.train if is_train else self.opt.val)

    def setup_optimizers(self):
        train_opt = self.opt['train']
    
        # Generator optim params 0
        optim_params = []
        for k, v in self.deblur_network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)  
        optimizer_g = torch.optim.Adam([{'params': optim_params}],
            lr=train_opt.optim.learning_rate,
            betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
            weight_decay=train_opt.optim.adam_weight_decay,
            eps=train_opt.optim.adam_epsilon,
        )
        self.optimizers.append(optimizer_g)
        
        
        # Reblur network optim params 1
        optim_params = []
        for k, v in self.reblur_net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
        optimizer_g = torch.optim.Adam([{'params': optim_params}],
            lr=train_opt.optim.learning_rate,
            betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
            weight_decay=train_opt.optim.adam_weight_decay,
            eps=train_opt.optim.adam_epsilon,
        )
        self.optimizers.append(optimizer_g)

        # discriminator optim params 2
        if self.discriminator is not None:
            d_optim_params = []
            for k, v in self.discriminator.named_parameters():
                if v.requires_grad:
                    d_optim_params.append(v)
            optimizer_d = torch.optim.Adam([{'params': d_optim_params}],
                lr=train_opt.optim.learning_rate,
                betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
                weight_decay=train_opt.optim.adam_weight_decay,
                eps=train_opt.optim.adam_epsilon,
            )
            self.optimizers.append(optimizer_d)
        

    def optimize_parameters(self):
        _lq = self.sample[self.dataloader.dataset.lq_key]
        gt = self.sample[self.dataloader.dataset.gt_key]

        if self.vae is not None:
            lq = self.vae.encode(_lq).latent_dist.sample() * self.vae.config.scaling_factor
            gt = self.vae.encode(gt).latent_dist.sample() * self.vae.config.scaling_factor
        else:
            lq = _lq
        
        opt_g, opt_reblur, opt_d = self.optimizers
        
        ### ---------------------------
        ### Step 1: Train Discriminator
        ### ---------------------------
        # 1a. Train on Real Data
        # We want the Discriminator to output 1 (Real) for real images
        opt_d.zero_grad() # dicriminator optimizer is at 1 index
        real_preds = self.discriminator(torch.cat([lq, gt], dim=1)) 
        loss_d_real = self.criterionGAN(real_preds, torch.ones_like(real_preds)) # Labels = 1

        # 1b. Train on Fake Data
        fake_images = self.deblur_network(lq)
        
        # We want the Discriminator to output 0 (Fake) for generated images
        # NOTE: .detach() is crucial here! We don't want to update Generator weights yet.
        fake_preds = self.discriminator(torch.cat([lq, fake_images.detach()], dim=1))
        loss_d_fake = self.criterionGAN(fake_preds, torch.zeros_like(fake_preds)) # Labels = 0

        # 1c. Backprop and Update Discriminator
        loss_d = (loss_d_real + loss_d_fake) / 2
        self.accelerator.backward(loss_d)
        # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.01)
        opt_d.step() 

        ### ---------------------------
        ### Step 2: Train Generator
        ### ---------------------------

        # We want the Generator to generate images that the Discriminator thinks are Real (1).
        opt_g.zero_grad()
                
        # Re-run discriminator on the *same* fake images (but this time, we keep the gradients connected)
        preds = self.discriminator(torch.cat([lq, fake_images], dim=1))
        
        # TRICK: We use 'real_targets' (1) as the label. 
        # If D says "1", G has succeeded.
        loss_g = self.criterionGAN(preds, torch.ones_like(preds))  # labels = 1
        
        # calculate loss for deblurred image 
        losses_pix = self.criterion(fake_images, gt)
        loss_pix = losses_pix['all']
        loss_deblur = (loss_pix + loss_g*self.lambda_adv)*self.lambda_deblur
        
        
        loss_gen = loss_deblur
        self.accelerator.backward(loss_gen)
        opt_g.step()
        
        opt_reblur.zero_grad()
        # calculate loss for reblurring
        losses_reblur = self.criterion(self.reblur_net(fake_images.detach()), lq)
        loss_reblur = losses_reblur['all']*self.lambda_reblur
        self.accelerator.backward(loss_reblur)
        opt_reblur.step()
        

        # torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        
        # merge losses for logging
        losses = {}
        losses['gen_pix'] = loss_pix
        losses['gen_adv'] = loss_g*self.lambda_adv
        losses['disc_fake'] = loss_d_fake
        losses['disc_real'] = loss_d_real
        losses['reblur'] = loss_reblur
        losses['deblur'] = loss_deblur
        losses['gen'] = loss_gen
        losses['disc'] = loss_d
        losses['all'] = loss_gen + loss_d
        return losses

    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        if self.opt.val.patched:
            b,c,h,w = self.original_size[lq_key]
            out_patch = []
            reblur_patch = []
            for _ in self.setup_patches():
                lq = self.sample[lq_key]
                if self.vae is not None:
                    lq = self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor
                _out = self.deblur_network(lq)
                _reblur = self.reblur_net(_out)
                if self.vae is not None:
                    _out = self.vae.decode(_out/self.vae.config.scaling_factor).sample
                    _reblur = self.vae.decode(_reblur/self.vae.config.scaling_factor).sample
                out_patch.append(_out)
                reblur_patch.append(_reblur)
            out_patch = torch.cat(out_patch, dim=0)
            out_patch = einops.rearrange(out_patch, '(b n) c h w -> b n c h w', b=b)
            reblur_patch = torch.cat(reblur_patch, dim=0)
            reblur_patch = einops.rearrange(reblur_patch, '(b n) c h w -> b n c h w', b=b)
            
            out = []
            reblur = []
            for i in range(len(out_patch)):
                merged = merge_patches(out_patch[i], self.sample[lq_key+'_patched_pos'])
                out.append(merged[..., :h, :w])
                merged = merge_patches(reblur_patch[i], self.sample[lq_key+'_patched_pos'])
                reblur.append(merged[..., :h, :w])

            lq = self.sample[lq_key+'_original']
            gt = self.sample[gt_key+'_original']
            out = torch.stack(out)
            reblur = torch.stack(reblur)
        else: 
            lq = self.sample[lq_key]
            gt = self.sample[gt_key]
            if self.vae is not None:
                _lq = self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor
            else:
                _lq = lq
            out  = self.deblur_network(_lq)
            reblur = self.reblur_net(out)
            if self.vae is not None:
                out = self.vae.decode(out/self.vae.config.scaling_factor).sample
                reblur = self.vae.decode(reblur/self.vae.config.scaling_factor).sample
            
        lq = lq.cpu().numpy()*0.5+0.5
        gt = gt.cpu().numpy()*0.5+0.5
        out = out.cpu().numpy()*0.5+0.5
        reblur = reblur.cpu().numpy()*0.5+0.5
        for i in range(len(gt)):
            idx += 1
            image1 = [einops.rearrange([lq[i, :3], reblur[i]], 'n c h w -> c h (n w)'), 
                      einops.rearrange([gt[i],     out[i]], 'n c h w -> c h (n w)')]
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx:04d}', self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
    
    
        