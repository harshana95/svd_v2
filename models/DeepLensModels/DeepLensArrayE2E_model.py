import os
import einops
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms

from dataset import create_dataset
from dataset.ZipDatasetWrapper import ZipDatasetWrapper
from models.archs import define_network
from models.base_model import BaseModel
from models.DeepLensModels.DeepLensArray_model import DeepLensArray_model
from utils.dataset_utils import crop_arr, grayscale, merge_patches, sv_convolution
from utils.image_utils import save_images_as_zip
from utils.loss import Loss
from utils import log_image, log_metrics

from deeplens.hybridlens import HybridLens
from deeplens.diffraclens import DiffractiveLens
from deeplens.optics.psf import conv_psf

class DeepLensArrayE2E_model(DeepLensArray_model):
    def __init__(self, opt, logger):
        super().__init__(opt, logger)
        
        # ========================== define generator network
        network_opt = opt.network
        
        network_opt['img_channel'] = network_opt['img_channel'] if not self.single_wavelength else 1
        network_opt['array_size'] = self.array_size
        network_opt['downscale_factor'] = 1
        self.net_g = define_network(network_opt).to(torch.float32)
        self.models.append(self.net_g)
        
        if self.is_train:
            self.net_g.train()

        # setup loss function
        train_opt = self.opt['train']
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)

    def setup_optimizers(self):
        opt = self.opt.train.optim

        # Optimizer creation
        optimizer_class = torch.optim.AdamW
        optim_params = []
        for model in self.models:
            for k, v in model.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print(f"Parameter {k} is not optimized.")
        def disable_grad(parameters):
            for p in parameters:
                if isinstance(p, dict):
                    if isinstance(p['params'],list):
                        for _p in p['params']:
                            _p.requies_grad = False
                    else:
                        p['params'].requires_grad = False
                else:
                    p.requires_grad = False
        lens_params = {}
        for i in range(len(self.MO)):

            params = []
            if self.opt.train.optimize_lens_param:
                params += self.MO[i].geolens.get_optimizer_params(lrs=[1e-4, 1e-4, 1e-1, 1e-5], decay=0.01)
            # else:
            #     _params = self.MO[i].geolens.get_optimizer_params(lrs=[1e-4, 1e-4, 1e-1, 1e-5], decay=0.01)
            #     disable_grad(_params)
            if self.opt.train.optimize_shape_param:
                params += self.MO[i].doe.get_optimizer_params(lr=0.1)
            # else:
            #     _params = self.MO[i].doe.get_optimizer_params(lr=0.1)
            #     disable_grad(_params)
            
            for param_group in params:
                if param_group['lr'] in lens_params:
                    lens_params[param_group['lr']]['params'] += param_group['params']
                else:
                    lens_params[param_group['lr']] = param_group

        # gather params with same lr
        params = []
        for lr in lens_params.keys():
            params.append(lens_params[lr])

        # add shape parameters as a parameter to optimize
        params_to_optimize = [{'params': optim_params}] + params 
        optimizer = optimizer_class(
            params_to_optimize,
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2),
            weight_decay=opt.adam_weight_decay,
            eps=opt.adam_epsilon,
            )
        self.optimizers.append(optimizer)          

    def setup_dataloaders(self):
        # create train and validation dataloaders
        
        train_set1 = create_dataset(self.opt.datasets.train1)
        train_set2 = create_dataset(self.opt.datasets.train2)
        self.dataloader = DataLoader(
            ZipDatasetWrapper({'1': train_set1, '2': train_set2}, transforms=transforms.Compose([]), random=False),
            shuffle=self.opt.datasets.train1.use_shuffle,
            batch_size=self.opt.train.batch_size*(self.depth_levels - 1),
            num_workers=self.opt.datasets.train1.get('num_worker_per_gpu', 1),
            drop_last=True
        )
        self.dataloader.dataset.gt_key = train_set1.gt_key
        self.dataloader.dataset.lq_key = train_set1.lq_key
                
        val_set1 = create_dataset(self.opt.datasets.val1)
        val_set2 = create_dataset(self.opt.datasets.val2)
        self.test_dataloader = DataLoader(
            ZipDatasetWrapper({'1': val_set1, '2': val_set2}, transforms=transforms.Compose([]), random=False),
            shuffle=self.opt.datasets.val1.use_shuffle,
            batch_size=self.opt.val.batch_size*(self.depth_levels - 1),
            num_workers=self.opt.datasets.val1.get('num_worker_per_gpu', 1),
            drop_last=True
        )
        self.test_dataloader.dataset.gt_key = val_set1.gt_key
        self.test_dataloader.dataset.lq_key = val_set1.lq_key

    def feed_data(self, data, is_train=True):
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        fg = data['foreground_1'].to(torch.float32) # batch size * (depth_levels -1)
        bg = data['background_2'].to(torch.float32) # batch size * (depth_levels -1)
        mask = data['mask_1'].to(torch.float32)  # batch size * (depth_levels -1)
        # print(bg.shape, fg.shape, mask.shape)

        bg = bg[:len(bg)//(self.depth_levels - 1)]  # pick first batch_size images. it will load batch_size*(depth_levels -1) images
        bg = einops.rearrange(bg, 'b c h w -> b c h w')
        fg = einops.rearrange(fg, '(b n) c h w -> b n c h w', n=self.depth_levels - 1)
        mask = einops.rearrange(mask, '(b n) c h w -> b n c h w', n=self.depth_levels - 1)
        
        
        if is_train:
            mask = torch.clamp(mask, 1e-6, 1 - 1e-6)  # if we don't clamp, we get NaNs.   
        device = mask.device
        
        # generate the ground truth images [all-in-focus image, depth map]
        depth = torch.rand(self.depth_levels)
        depth = depth*(self.depth_max - self.depth_min) + self.depth_min
        depth = torch.sort(depth, descending=True)[0] # from far to near
        depth *= self.m_scale

        _ps_locs = [torch.cat([self.ps_locs, -torch.ones(len(self.ps_locs), 1)*depth[_i]], dim=-1) for _i in range(self.depth_levels)]
        _ps_locs = torch.cat(_ps_locs, dim=0)  # [P, 3] P = n_pos * depth_levels (bg, fg1, fg2, ...)
        
        self.all_psf_intensity = []
        all_meas = []
        # simulate PSF for current p
        for i in range(len(self.MO)):
            if self.single_wavelength:
                wv_idx = (i//self.array_size[0])%len(self.wavelength_set_m)
                wv = self.wavelength_set_m[wv_idx].item()*1e6 # um
                _bg = bg[..., wv_idx:wv_idx+1,:,:]
                _fg = fg[..., wv_idx:wv_idx+1,:,:]
                _mask = mask[..., wv_idx:wv_idx+1,:,:]
            else:
                raise Exception()
                wv = self.wavelength_set_m
                _bg = bg
                _fg = fg
                _mask = mask
            ks = 256
            ps_locs = _ps_locs - self.MS_pos[i] # MS at center, translate obj
            if not (self.opt.train.optimize_shape_param or self.opt.train.optimize_lens_param):
                with torch.no_grad():
                    psf = self.MO[i].psf(points=ps_locs[0], ks=ks, wvln=wv, spp=1_000_000).to(torch.float32)
            else:
                psf = self.MO[i].psf(points=ps_locs[0], ks=ks, wvln=wv, spp=1_000_000).to(torch.float32)
            self.all_psf_intensity.append(psf)
            meas = conv_psf(_bg, psf[None])[..., :_bg.shape[-2], :_bg.shape[-1]]
            # print(f"====== {i}")
            # print(_bg.shape, meas.shape, psf.min(), psf.max(), ps_locs[0], wv)
            for ps_i in range(1, ps_locs.shape[0]):
                psf = self.MO[i].psf(points=ps_locs[ps_i], ks=ks, wvln=wv, spp=1_000_000).to(torch.float32)
                self.all_psf_intensity.append(psf)
                fg_meas = conv_psf(_fg[:, ps_i-1], psf[None])[..., :_bg.shape[-2], :_bg.shape[-1]]
                mask_meas = conv_psf(_mask[:, ps_i-1], psf[None])[..., :_bg.shape[-2], :_bg.shape[-1]]
                # print(psf.min(), psf.max(), ps_locs[ps_i])

                # alpha clipping and merging with previous measurement
                mask_meas = mask_meas/mask_meas.max()
                meas = meas*(1 - mask_meas) + fg_meas*mask_meas
            all_meas.append(meas)  # no polarization

        # this is not correct for all MS in the array. there can be parallax effect for GT
        gt = bg
        for d in range(self.depth_levels -1):
            gt = gt*(1 - mask[:, d]) + fg[:, d]*mask[:, d]

        # calculate depth
        depth_map = depth[0].to(device)
        for d in range(self.depth_levels - 1):
            depth_map = depth_map*(1-mask[:, d]) + depth[d+1].to(device)*mask[:, d]

        all_meas = torch.stack(all_meas)
        all_meas = einops.rearrange(all_meas, "n b c h w -> b n c h w")
        data[gt_key] = gt.to(torch.float32)
        data[lq_key] = all_meas.to(torch.float32)
        data['depth'] = depth_map.to(torch.float32)[:, 0:1, :, :]
        data['depth'] = (data['depth'] - self.depth_min*self.m_scale)/self.m_scale/(self.depth_max - self.depth_min) # normalize depth to 0-1
        data['mask'] = mask
        
        self.sample = data
        if self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key, 'depth'], opt=self.opt.train if is_train else self.opt.val)
            
    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        preds_depth = self.net_g(self.sample[lq_key]).to(torch.float32)

        losses = self.criterion(preds_depth, self.sample['depth'])
        self.accelerator.backward(losses['all'])
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        return losses

    @torch.no_grad()
    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        if idx == 0:
            psfs = einops.rearrange(torch.stack(self.all_psf_intensity).detach().cpu().numpy(), '(n d) h w -> d n 1 h w', d=self.depth_levels)
            for d in range(self.depth_levels):
                image1 = einops.rearrange(psfs[d], '(n2 n1) c h w -> c (n2 h) (n1 w)', n1=self.array_size[0], n2=self.array_size[1])
                image1 = np.stack([image1])
                image1 = image1/image1.max()
                image1 = np.clip(image1, 0, 1)
                log_image(self.opt, self.accelerator, image1, f"psfs_depth_{d}", self.global_step)
        
        if self.opt.val.patched:
            b,c,h,w = self.original_size[lq_key]
            pred = []
            for _ in self.setup_patches():
                pred.append(self.net_g(self.sample[lq_key]))
            pred = torch.cat(pred, dim=0)
            pred = einops.rearrange(pred, '(b n) c h w -> b n c h w', b=b)
            out = []
            for i in range(len(pred)):
                merged = merge_patches(pred[i], self.sample[lq_key+'_patched_pos'])
                out.append(merged[..., :h, :w])
            lq = self.sample[lq_key+'_original']
            gt = self.sample[gt_key+'_original']
            gt_depth = self.sample['depth_original']
            out = torch.stack(out)
        else: 
            lq = self.sample[lq_key]
            gt = self.sample[gt_key]
            gt_depth = self.sample['depth']
            out = self.net_g(lq)
            
        lq = lq.cpu().numpy()
        gt = gt.cpu().numpy()
        gt_depth = gt_depth.cpu().numpy()
        out = out.cpu().numpy()
        for i in range(len(gt)):
            idx += 1
            # save_images_as_zip(lq[i], self.opt.path.experiments_root, f'lq_{idx:04d}.zip')
            lq_i = einops.rearrange(lq[i], '(n2 n1) c h w -> c (n2 h) (n1 w)', n1=self.array_size[0], n2=self.array_size[1])
            lq_i /= lq_i.max()
            lq_i = np.stack([lq_i])
            lq_i = np.clip(lq_i, 0, 1)
            log_image(self.opt, self.accelerator, lq_i, f'lq_{idx:04d}', self.global_step)

            image1 = [gt[i], out[i], gt_depth[i]]
            for j in range(len(image1)):
                print(image1[j].shape)
                if image1[j].shape[0] == 1:
                    image1[j] = einops.repeat(image1[j], '1 h w -> 3 h w')
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx:04d}', self.global_step)
            log_metrics(gt_depth[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
