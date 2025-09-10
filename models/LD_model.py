# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import gc
import einops
import numpy as np
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from models.archs import define_network
from models.base_model import BaseModel

from models.archs.related.LD.loss import GANLoss, VGGLoss
from utils.dataset_utils import merge_patches
from utils.misc import log_image, log_metrics

class LD_model(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt, logger):
        super(LD_model, self).__init__(opt, logger)

        # define network
        self.net_g = define_network(opt.network)
        self.net_d = define_network({'type':"MultiscaleDiscriminator"})
        self.models.append(self.net_g)
        self.models.append(self.net_d)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.criterionGAN = GANLoss('hinge')
        self.criterionVGG = VGGLoss(0)
        self.criterionFeat = torch.nn.L1Loss()
        self.L1 = torch.nn.L1Loss()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_lowlr = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    optim_params_lowlr.append(v)
                else:
                    optim_params.append(v)
            else:
                self.logger.warning(f'Params {k} will not be optimized.')
        ratio = 0.1
        self.optimizer_g = torch.optim.Adam(
            [{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim']['learning_rate'] * ratio}],
            lr=train_opt.optim.learning_rate,
            betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
            weight_decay=train_opt.optim.adam_weight_decay,
            eps=train_opt.optim.adam_epsilon,
            )
        
        self.optimizers.append(self.optimizer_g)

        d_optim_params = []
        d_optim_params_lowlr = []
        for k, v in self.net_d.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    d_optim_params_lowlr.append(v)
                else:
                    d_optim_params.append(v)
            else:
                self.logger.warning(f'Params {k} will not be optimized.')
        ratio = 0.1
        self.optimizer_d = torch.optim.Adam(
                [{'params': d_optim_params}, {'params': d_optim_params_lowlr, 'lr': train_opt['optim']['learning_rate'] * ratio}],
                lr=train_opt.optim.learning_rate,
                betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
                weight_decay=train_opt.optim.adam_weight_decay,
                eps=train_opt.optim.adam_epsilon,
                )
        self.optimizers.append(self.optimizer_d)
        
    def feed_data(self, data, is_train=True):
        self.sample = data
        if is_train:
            if self.opt.train.patched:
                self.grids(keys=[self.dataloader.dataset.lq_key, self.dataloader.dataset.gt_key], opt=self.opt.train)
        else:
            if self.opt.val.patched:
                self.grids(keys=[self.test_dataloader.dataset.lq_key, self.test_dataloader.dataset.gt_key], opt=self.opt.val)
      
    def compute_generator_loss(self):
        
        lq_key = self.dataloader.dataset.lq_key
        gt_key = self.dataloader.dataset.gt_key
        G_losses = OrderedDict() #{}
        # fake_image, KLD_loss = self.generate_fake(input_semantics, real_image, compute_kld_loss=self.opt.use_vae)
        pred, fake_image = self.net_g(self.sample[lq_key], self.sample[gt_key])
        pred_fake, pred_real = self.discriminate(self.sample[gt_key], fake_image, self.sample[lq_key])
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)
        num_D = len(pred_fake)
        GAN_Feat_loss = torch.FloatTensor(1).cuda().fill_(0)
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss * 10 / num_D
        G_losses['GAN_Feat'] = GAN_Feat_loss
        G_losses['VGG'] = self.criterionVGG(fake_image, self.sample[lq_key])*30
        
        l_pix = 0.
        # for pred in preds:
        l_pix += 10 *self.L1(pred, self.sample[gt_key])
            # print('l pix ... ', l_pix)
        G_losses['l_pix'] = l_pix
        
        # blur_list1 = self.net_loss(self.sample[gt_key])
        # blur_list2 = self.net_loss(preds[-1])
        # l_blur = 0.
        #for kk in range(len(blur_list1)):
        #    l_blur_ele = self.L1(blur_list1[kk], blur_list2[kk])
        #    l_blur += l_blur_ele
            
        # G_losses['l_blur'] = l_blur
        return pred, fake_image, G_losses
        
    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.net_d(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real
    
    def compute_discriminator_loss(self):
        lq_key = self.dataloader.dataset.lq_key
        gt_key = self.dataloader.dataset.gt_key
        D_losses = OrderedDict()
        with torch.no_grad():
            _, fake_image = self.net_g(self.sample[lq_key], self.sample[gt_key])
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        pred_fake, pred_real = self.discriminate(self.sample[gt_key], fake_image, self.sample[lq_key])
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)
        return D_losses

    def optimize_parameters(self):
        self.optimizer_g.zero_grad()
        preds, inverse_preds, G_losses = self.compute_generator_loss()
        self.output = preds
        self.output1 = inverse_preds
        l_total = (G_losses['GAN'] + G_losses['GAN_Feat'] + G_losses['VGG']) + G_losses['l_pix']
        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        
        self.optimizer_d.zero_grad()
        D_losses = self.compute_discriminator_loss()
        D_loss = sum(D_losses.values()).mean()
        D_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), 0.01)
        self.optimizer_d.step()
    
        losses = {}
        for k in G_losses.keys():
            losses[k] = G_losses[k]
        for k in D_losses.keys():
            losses[k] = D_losses[k]
        losses['G_loss'] = l_total
        losses['D_loss'] = D_loss
        losses['all'] = l_total + D_loss
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
   
        
