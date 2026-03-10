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



class DeblurDeform_model(BaseModel):
    def __init__(self, opt, logger):
        super(DeblurDeform_model, self).__init__(opt, logger)
        self.n_channels = 3
        self.vae = None
        self.meta_vae = None
        if opt.vae_name_or_path:
            self.vae = AutoencoderKL.from_pretrained(opt.vae_name_or_path, subfolder=opt.vae_subfolder)
            self.vae.requires_grad_(False)
            self.vae.eval()
            self.vae.to(self.accelerator.device)

            self.n_channels = int(self.vae.latent_channels)
            # self.meta_vae = T2IAdapter(
            #     in_channels=opt.deform_network.in_channels,
            #     channels=(320, 640, 1280, 1280),
            #     num_res_blocks=2,
            #     downscale_factor=8,
            #     adapter_type="full_adapter",
            # )
        if opt.meta_vae:
            self.meta_vae = Encoder_arch(Encoder_config(**opt.meta_vae.__dict__))
            if opt.meta_vae.weight_path:
                self.meta_vae.from_pretrained(opt.meta_vae.weight_path)
                self.meta_vae.requires_grad_(False)
                self.meta_vae.eval()
                self.meta_vae.to(self.accelerator.device)
            self.models.append(self.meta_vae)

        
        # define network
        self.reblur_net = define_network(opt.reblur_network)
        self.deform_network = define_network(opt.deform_network)
        self.deblur_network = define_network(opt.deblur_network)
        if opt.reblur_network.weight_path:
            self.reblur_net.from_pretrained(opt.reblur_network.weight_path)
            self.reblur_net.requires_grad_(False)
            self.reblur_net.eval()
            self.reblur_net.to(self.accelerator.device)
        self.models.append(self.reblur_net)
        self.models.append(self.deform_network)
        self.models.append(self.deblur_network)
        
        # define discriminator if defined in yml
        self.discriminator = None
        if opt.discriminator is not None:
            self.discriminator = define_network(opt.discriminator)
            self.models.append(self.discriminator)
            self.lambda_adv = opt.train.get("lambda_adv", 1)

        self.lambda_deblur = opt.train.get("lambda_deblur", 1)
        self.lambda_reblur = opt.train.get("lambda_reblur", 1)
        self.lambda_deform = opt.train.get("lambda_deform", 1)

        if self.is_train:
            self.init_training_settings()

        # setup other parameters

    def init_training_settings(self):
        train_opt = self.opt['train']
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)    
        
        self.deblur_network.train()
        self.deform_network.train()
                
        if self.discriminator is not None:
            self.discriminator.train()
            self.criterionGAN = nn.BCELoss() # GANLoss('original').to(self.accelerator.device)

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        meta_key1 = self.opt.meta_key1
        meta_key2 = self.opt.meta_key2
        if self.opt.train.patched if is_train else self.opt.val.patched:
            self.grids(keys=[lq_key, gt_key, meta_key1], opt=self.opt.train if is_train else self.opt.val)

    def setup_optimizers(self):
        train_opt = self.opt['train']
    
        # Generator optim params 0
        optim_params = []
        for k, v in self.deblur_network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)  
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
        
        # Deform network optim params 1
        optim_params = []
        if self.meta_vae is not None:
            for k, v in self.meta_vae.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)  
        for k, v in self.deform_network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
        optimizer_g = torch.optim.Adam([{'params': optim_params}],
            lr=train_opt.optim.learning_rate,
            betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
            weight_decay=train_opt.optim.adam_weight_decay,
            eps=train_opt.optim.adam_epsilon,
        )
        self.optimizers.append(optimizer_g)

        # # Reblur network optim params 2
        # optim_params = []
        # for k, v in self.reblur_net.named_parameters():
        #     if v.requires_grad:
        #         optim_params.append(v)
        # optimizer_g = torch.optim.Adam([{'params': optim_params}],
        #     lr=train_opt.optim.learning_rate,
        #     betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
        #     weight_decay=train_opt.optim.adam_weight_decay,
        #     eps=train_opt.optim.adam_epsilon,
        # )
        # self.optimizers.append(optimizer_g)

        # discriminator optim params 3
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

    def get_meta_data(self):
        raise NotImplementedError()

    def _deblur_forward(self, lq):
        """Unify deblur_network output: (img, deform) or single tensor [img_channels, deform_channels]."""
        out = self.deblur_network(lq)
        if isinstance(out, tuple):
            return out[0], out[1]
        return out[:, :self.n_channels], out[:, self.n_channels:]

    def optimize_parameters(self):
        _lq = self.sample[self.dataloader.dataset.lq_key]
        gt = self.sample[self.dataloader.dataset.gt_key]
        meta = self.get_meta_data()

        if self.vae is not None:
            lq = self.vae.encode(_lq).latent_dist.sample() * self.vae.config.scaling_factor
            gt = self.vae.encode(gt).latent_dist.sample() * self.vae.config.scaling_factor
        else:
            lq = _lq
        if self.meta_vae is not None:
            meta = self.meta_vae(meta) 
        
        opt_g, opt_deform = self.optimizers[0], self.optimizers[1]
        opt_d = self.optimizers[2] if self.discriminator is not None else None

        fake_images, deform_hat = self._deblur_forward(lq)

        ### ---------------------------
        ### Step 1: Train Discriminator (if present)
        ### ---------------------------
        if self.discriminator is not None:
            opt_d.zero_grad()
            real_preds = self.discriminator(torch.cat([lq, gt], dim=1))
            loss_d_real = self.criterionGAN(real_preds, torch.ones_like(real_preds))
            fake_preds = self.discriminator(torch.cat([lq, fake_images.detach()], dim=1))
            loss_d_fake = self.criterionGAN(fake_preds, torch.zeros_like(fake_preds))
            loss_d = (loss_d_real + loss_d_fake) / 2
            self.accelerator.backward(loss_d)
            opt_d.step()

        ### ---------------------------
        ### Step 2: Train Generator
        ### ---------------------------
        opt_g.zero_grad()
        if self.discriminator is not None:
            preds = self.discriminator(torch.cat([lq, fake_images], dim=1))
            loss_g = self.criterionGAN(preds, torch.ones_like(preds))
        else:
            loss_g = torch.tensor(0.0, device=lq.device)

        losses_pix = self.criterion(fake_images, gt)
        loss_pix = losses_pix['all']
        loss_deblur = (loss_pix + loss_g * self.lambda_adv) * self.lambda_deblur
        
        # calculate loss for deform map
        deform_map = self.deform_network(torch.cat([lq, meta], dim=1))
        losses_deform = self.criterion(deform_hat, deform_map.detach())
        loss_deform = losses_deform['all']*self.lambda_deform

        # opt_reblur.zero_grad()
        # calculate loss for reblurring
        losses_reblur = self.criterion(self.reblur_net(torch.cat([fake_images, deform_hat], dim=1)), lq)
        loss_reblur = losses_reblur['all']*self.lambda_reblur
        # self.accelerator.backward(loss_reblur)
        # opt_reblur.step()
        
        loss_gen = loss_deblur + loss_deform + loss_reblur
        self.accelerator.backward(loss_gen)
        opt_g.step()

        opt_deform.zero_grad()
        # calculate loss for deform map
        losses_deform = self.criterion(deform_hat.detach(), deform_map)
        loss_deform = losses_deform['all']*self.lambda_deform
        self.accelerator.backward(loss_deform)
        opt_deform.step()
        
        
        

        # torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        
        # merge losses for logging
        losses = {}
        losses['gen_pix'] = loss_pix
        losses['gen_adv'] = loss_g * self.lambda_adv
        losses['deform'] = loss_deform
        losses['reblur'] = loss_reblur
        losses['deblur'] = loss_deblur
        losses['gen'] = loss_gen
        if self.discriminator is not None:
            losses['disc_fake'] = loss_d_fake
            losses['disc_real'] = loss_d_real
            losses['disc'] = loss_d
            losses['all'] = loss_gen + loss_d
        else:
            losses['all'] = loss_gen
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
                _out, deform_hat = self._deblur_forward(lq)
                _reblur = self.reblur_net(torch.cat([_out, deform_hat], dim=1))
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
            out, deform_hat = self._deblur_forward(_lq)
            reblur = self.reblur_net(torch.cat([out, deform_hat], dim=1))
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
    
    
        