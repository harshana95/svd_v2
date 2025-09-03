"""
Training a reconstruction neural network, PSF basis and coefficients without any constraints for 
simultaneous depth estimation and image reconstruction.
"""
import gc
import einops
from matplotlib import pyplot as plt
import numpy as np
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from models.archs import define_network
from models.base_model import BaseModel
from utils.image_utils import display_images
from utils.zernike import phase2psf, zernike_poly
from utils.loss import Loss

from utils.dataset_utils import merge_patches
from utils.misc import fig_to_array, log_image, log_metrics

class MetaDepthPhase1_model(BaseModel):
    def __init__(self, opt, logger):
        super(MetaDepthPhase1_model, self).__init__(opt, logger)
        self.image_size = opt.image_size
        Nb = 36 # number of basis functions
        C = 1 # number of channels
        self.Nd = 16 # number of depth levels
        self.psf_init = 'zernike'
        # initialize psfs and coeffs
        if self.psf_init == 'random':
            self.psf_basis = torch.rand((1, Nb, C, opt.image_size[0], opt.image_size[1]), dtype=torch.float32)
            self.psf_basis /= self.psf_basis.sum(dim=(-2, -1), keepdim=True)  # normalize psf basis
            self.psf_basis = torch.nn.Parameter(self.psf_basis, requires_grad=True)
        elif self.psf_init == 'zernike':
            H, W = self.image_size
            zH,zW = H//1, W//1
            zernike_phase = zernike_poly(Nb, H, W, zH,zW)
            self.psf_basis = phase2psf(torch.from_numpy(zernike_phase), D=zH/H, pad=0)
            self.psf_basis /= self.psf_basis.sum(dim=(-2, -1), keepdim=True)  # normalize psf basis
            self.psf_basis = einops.repeat(self.psf_basis, 'n h w -> 1 n c h w', c=C).to(torch.float32)
            self.psf_basis = torch.nn.Parameter(self.psf_basis, requires_grad=False)
        self.psf_coeff = torch.rand((1, self.Nd, Nb, C, self.image_size[0], self.image_size[1]), dtype=torch.float32)
        self.psf_coeff /= self.psf_coeff.sum(dim=(2,), keepdim=True)
        self.psf_coeff = torch.nn.Parameter(self.psf_coeff, requires_grad=True)

        # define network
        self.net = define_network(opt.network)
        self.models.append(self.net)
        self.criterion = Loss(opt['train'].loss).to(self.accelerator.device)   

    def plot_psfs(self):
        basis_psfs = einops.rearrange(self.psf_basis.detach().cpu().numpy(), '1 n c h w -> n h w c')
        Nb, h, w, c = basis_psfs.shape
        # plotting basis psfs        
        fig = display_images({"Basis":basis_psfs}, independent_colorbar=True, size=2)
        img = fig_to_array(fig)
        plt.close(fig)
        log_image(self.opt, self.accelerator, img[None], 'Basis PSFs', self.global_step)

        # plotting psfs at given positions and all depths
        crop_size = [32,32]
        crop = [h//2-crop_size[0]//2, h//2+crop_size[0]//2,
                w//2-crop_size[1]//2, w//2+crop_size[1]//2]
        d = {}
        for pi in np.linspace(0, h-1, 4, dtype=int):
            for pj in np.linspace(0, w-1, 4, dtype=int):
                W = self.psf_coeff[0,:,:,:,pi, pj].detach().cpu().numpy()  # (Nd Nb C)
                W = einops.rearrange(W, 'Nd Nb c -> Nd Nb 1 1 c')
                psfs = (basis_psfs[None] * W).sum(axis=1)  # Nd h w c
                d[f'psfs({pi},{pj})'] = psfs[:, crop[0]:crop[1], crop[2]:crop[3],:]
        fig = display_images(d, independent_colorbar=False, size=2)
        img = fig_to_array(fig)
        plt.close(fig)
        log_image(self.opt, self.accelerator, img[None], 'PSFs', self.global_step)

        # plotting coeffs
        d = {}
        coeff = self.psf_coeff.detach().cpu().numpy()  # (1 Nd Nb C h w)
        coeff = einops.rearrange(coeff, '1 Nd Nb c h w -> Nd Nb h w c')
        for i in range(coeff.shape[1]):
            d[f"Basis {i}"] = coeff[:, i]
        fig = display_images(d, independent_colorbar=False, size=2)
        img = fig_to_array(fig)
        plt.close(fig)
        log_image(self.opt, self.accelerator, img[None], 'Coeff', self.global_step)


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
        
        # add shape parameters as a parameter to optimize
        params_to_optimize = [{'params': optim_params}, 
                              {'params': self.psf_basis, 'lr': opt.learning_rate * 0.00001},
                              {'params': self.psf_coeff, 'lr': opt.learning_rate * 0.00001}]
        optimizer = optimizer_class(
            params_to_optimize,
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2),
            weight_decay=opt.adam_weight_decay,
            eps=opt.adam_epsilon,
            )
        self.optimizers.append(optimizer)


    def feed_data(self, data, is_train=True):
        assert not self.opt.train.patched
        self.sample = data
        x = data['image']
        d = data['depth']
        b, c, h, w = x.shape
        assert h == self.image_size[0] and w == self.image_size[1], f"Image size mismatch: {h}x{w} vs {self.image_size[0]}x{self.image_size[1]}"
        if self.psf_coeff.device != x.device:
            print(f"Moving psf_coeff and psf_basis to {x.device} {self.psf_coeff.shape} {self.psf_basis.shape}")
            self.psf_coeff = self.psf_coeff.to(x.device)
            self.psf_basis = self.psf_basis.to(x.device)

        # discretize depth for Nd levels
        d = (d * (self.Nd - 1)).clamp(0, self.Nd-1).to(torch.long)
        d = d.squeeze(1) # remove 1st dimesion (b H W)
        batch_indices = torch.arange(b, device=d.device).view(b, 1, 1).expand(b, h, w)
        height_indices = torch.arange(h, device=d.device).view(1, h, 1).expand(b, h, w)
        width_indices = torch.arange(w, device=d.device).view(1, 1, w).expand(b, h, w)
        # psf coeff based on depth. psf_coeff=(1 Nd Nb C H W) d=(b H W)
        W = self.psf_coeff[batch_indices, d,:,:,height_indices, width_indices]  # (b Nb C H W)
        W = einops.rearrange(W, 'b h w Nb c -> b Nb c h w')  # :, : puts the dims at the end

        # take the fourier transform of the image
        X = torch.fft.fft2(x[:, None] * W, dim=(-2, -1))
        # take the fourier transform of the psf
        H = torch.fft.fft2(self.psf_basis, dim=(-2, -1))  # need to do this everytime because we are updating it
        y = torch.fft.ifftshift(torch.fft.ifft2(X * H, dim=(-2, -1)), dim=(-2, -1)).real.sum(1)
        
        data['meas'] = y
        self.sample = data

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        net_in = torch.cat([self.sample['meas']], dim=1)
        preds = self.net(net_in)
        losses1 = self.criterion(preds[:,:1,:,:], self.sample['image'])
        losses2 = self.criterion(preds[:,1:,:,:], self.sample['depth'])
        loss = losses1['all']+losses2['all']
        self.accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.01)
        
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        return {'loss_image': losses1['all'], 'loss_depth': losses2['all'], 'loss': loss}


    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        if idx == 0:
            self.plot_psfs()
        if idx > 10:
            return idx + 1
        if self.opt.val.patched:
            b,c,h,w = self.original_size['image']
            pred = []
            meas = []
            for _ in self.setup_patches(): 
                net_in = torch.cat([self.sample['meas']], dim=1)
                out = self.net(net_in)
                pred.append(out)
                meas.append(self.sample['meas'])
            pred = torch.cat(pred, dim=0)
            pred = einops.rearrange(pred, '(b n) c h w -> b n c h w', b=b)
            meas = torch.cat(meas, dim=0)
            meas = einops.rearrange(meas, '(b n) c h w -> b n c h w', b=b)
            out = []
            meas_arr = []
            for i in range(len(pred)):
                out.append(merge_patches(pred[i], self.sample['image_patched_pos'])[..., :h, :w])
                meas_arr.append(merge_patches(meas[i], self.sample['meas_patched_pos'])[..., :h, :w])
            image = self.sample['image_original']
            depth = self.sample['depth_original']
            out = torch.stack(out)
            meas = torch.stack(meas_arr)
        else: 
            image = self.sample['image']
            depth = self.sample['depth']
            net_in = torch.cat([self.sample['meas']], dim=1)
            out = self.net(net_in)
            meas = self.sample['meas']

        image = image.cpu().numpy()
        depth = depth.cpu().numpy()
        out = out.cpu().numpy()
        meas = meas.cpu().numpy()
        for i in range(len(image)):
            idx += 1
            image1 = [image[i], depth[i], meas[i], out[i, :1], out[i, 1:]]
            for j in range(len(image1)):
                if image1[j].shape[-3] == 1:
                    image1[j] = einops.repeat(image1[j], '1 h w -> 3 h w')
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx}', self.global_step)  # image format (N,C,H,W)
            log_metrics(out[i, :1], image[i], self.opt.val.metrics, self.accelerator, self.global_step, 'image')
            log_metrics(out[i, 1:], depth[i], self.opt.val.metrics, self.accelerator, self.global_step, 'depth')
        return idx
    