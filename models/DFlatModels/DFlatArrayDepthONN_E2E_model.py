import einops
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms

from dataset import create_dataset
from models.DFlatModels.DFlatArrayDepthONN_model import DFlatArrayDepthONN_model
from models.DFlatModels.DFlatArrayDepth_model import DFlatArrayDepth_model
from models.archs import define_network
from models.archs.related.AdaBinsMonoDepth.unet_adaptive_bins import UnetAdaptiveBins
from models.archs.related.MonoDepth2.monodepth2_arch import DepthDecoder, ResnetEncoder
from utils.dataset_utils import crop_arr, grayscale, merge_patches
from utils.image_utils import save_images_as_zip
from utils.loss import Loss
from utils import log_image, log_metrics


class DFlatArrayDepthONN_E2E_model(DFlatArrayDepthONN_model):
    def __init__(self, opt, logger):
        super().__init__(opt, logger)
        self.kernel_rescale_factor = 1

    def feed_data(self, data, is_train=True):
        assert not self.opt.train.patched
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        
        x = data['image']
        d = data['depth']
        b, c, h, w = x.shape
        device = self.accelerator.device

        FIXED_DEPTH = 10 # in meters. We can use depth map instead but fixed now for testing
        self.all_psf_intensity = []
        all_meas = []

        # simulate PSF for current p
        for i in range(len(self.p_norm)): # iterate over all metasurfaces
            ps_locs = torch.cat([self.ps_locs, torch.ones(len(self.ps_locs), 1)*FIXED_DEPTH], dim=-1)  # add depth to point spread locations
            ps_locs -= self.MS_pos[i] # MS at center, translate obj

            nan_check_sum = torch.isnan(self.p_norm[i]).sum()
            assert not nan_check_sum, f'{nan_check_sum} Nan in p_norm'

            est_amp, est_phase = self.optical_model(self.p_norm[i], [self.wavelength_set_m[i%len(self.wavelength_set_m)]], pre_normalized=True)
            psf_intensity, _ = self.PSF(
                est_amp.to(dtype=torch.float32, device=device),
                est_phase.to(dtype=torch.float32, device=device),
                [self.wavelength_set_m[i%len(self.wavelength_set_m)]],
                ps_locs,
                aperture=None,
                normalize_to_aperture=True)

            self.all_psf_intensity.append(psf_intensity)

            # # Need to pass inputs like
            # # psf has shape  [B P Z L H W]
            # # scene radiance [B P Z L H W]
            # # out shape      [B P Z L H W]
            # meas = self.renderer(psf_intensity, einops.rearrange(x, 'b c h w -> b 1 1 c h w'), rfft=True, crop_to_psf_dim=False)
            # meas = self.renderer.rgb_measurement(meas, np.array([self.wavelength_set_m[i%len(self.wavelength_set_m)]]), gamma=True, process='demosaic')

            # assert torch.isnan(meas).sum() == 0, f'{torch.isnan(meas).sum()} Nan in meas'
            # meas = meas[:, 0, 0]  # no polarization
            # all_meas.append(meas)
        
        # all_meas = torch.stack(all_meas)
        # all_meas = einops.rearrange(all_meas, "n b c h w -> b n c h w")
        # data[lq_key] = all_meas
        
        self.sample = data
        if self.opt.train.patched:
            raise Exception()

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key
        all_psfs = self.all_psf_intensity

        image = self.sample[gt_key]
        print(image.min(), image.max())

        # predict using frozen model
        with torch.no_grad():
            model_pred = self.depth_model(image)[("disp", 0)]
        
        # predict using ms weights
        all_psfs = self.all_psf_intensity
        all_psfs = einops.rearrange(torch.stack(all_psfs), '(pn N c) 1 1 1 1 h w -> pn N c h w', pn=2, N=64, c=3)  # hard coded
        kernels = crop_arr(all_psfs, 7, 7)  # hard coded
        kernels = kernels[0] - kernels[1]  # kernels are soo small, the gradients barely update
        kernels *= self.kernel_rescale_factor

        # todo denormalize kernels

        # update conv1 weights
        # self.encoder.encoder.conv1.weight.data = kernels  # check what is going on here

        # predict using updated kernels
        onn_pred = self.decoder(self.encoder(image, kernels))[("disp", 0)]
        
        # calculate loss
        total_loss = F.mse_loss(model_pred, onn_pred)*1e6

        self.accelerator.backward(total_loss)

        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        print(self.p_norm[0])
        return {"all": total_loss}

    def validate_step(self, batch, idx, lq_key, gt_key):
        if idx == 0:
            self.feed_data(batch, is_train=False)
            psfs = {}
            NORMALIZE_PSFS = True
            kh, kw = 7, 7
            for i in range(len(self.all_psf_intensity)):
                psf_intensity = self.all_psf_intensity[i]
                for j in range(psf_intensity.shape[2]):
                    psf = psf_intensity[0,0,j].detach().cpu().numpy()
                    if NORMALIZE_PSFS:
                        psf = psf/psf.max()
                    if i < 5:
                        log_image(self.opt, self.accelerator, psf[None], f"psf{i}_ch{j}", self.global_step)
                    psf = crop_arr(psf, kh, kw)
                    if j in psfs.keys():
                        psfs[j].append(psf)
                    else:
                        psfs[j] = [psf]
            for k in psfs.keys():
                psfs_k = np.stack(psfs[k])
                if NORMALIZE_PSFS:
                    psfs_k = psfs_k/psfs_k.sum(0, keepdims=True)
                image1 = einops.rearrange(psfs_k, '(n1 n2) c h w -> c (n2 h) (n1 w)', n1=self.array_size[0], n2=self.array_size[1])
                image1 = np.stack([image1])
                image1 = image1/image1.max()
                image1 = np.clip(image1, 0, 1)
                ps_loc = self.ps_locs[k%len(self.ps_locs)]
                log_image(self.opt, self.accelerator, image1, f"img{idx:04d}_psfs_ch{k}_{ps_loc}", self.global_step)
        
        image = self.sample['image']
        depth = self.sample['depth']

        # predict using frozen model
        with torch.no_grad():
            model_pred = self.depth_model(image)[("disp", 0)]
        
        # predict using ms weights
        all_psfs = self.all_psf_intensity
        all_psfs = einops.rearrange(torch.stack(all_psfs), '(pn N c) 1 1 1 1 h w -> pn N c h w', pn=2, N=64, c=3)  # hard coded
        kernels = crop_arr(all_psfs, 7, 7)  # hard coded
        kernels = kernels[0] - kernels[1]  # out_c in_c 7 7
        kernels *= self.kernel_rescale_factor

        # todo denormalize kernels

        # update conv1 weights
        # self.encoder.encoder.conv1.weight.data = kernels

        # predict using updated kernels
        onn_pred = self.decoder(self.encoder(image, kernels))[("disp", 0)]
        
        model_pred = model_pred.cpu().numpy()
        onn_pred = onn_pred.cpu().numpy()
        depth = depth.cpu().numpy()
        image = image.cpu().numpy()
        
        for i in range(len(model_pred)):
            idx += 1
            # model_pred_i = np.clip(np.stack([model_pred[i]]), 0, 1)
            # log_image(self.opt, self.accelerator, model_pred_i, f'model_pred_{idx:04d}', self.global_step)

            # onn_pred_i = np.clip(np.stack([onn_pred[i]]), 0, 1)
            # log_image(self.opt, self.accelerator, onn_pred_i, f'onn_pred_{idx:04d}', self.global_step)
            
            # depth_i = np.clip(np.stack([depth[i]]), 0, 1)
            # log_image(self.opt, self.accelerator, depth_i, f'depth_{idx:04d}', self.global_step)
            
            image1 = [image[i], model_pred[i], onn_pred[i], depth[i]]
            for j in range(len(image1)):
                if image1[j].shape[0] == 1:
                    image1[j] = einops.repeat(image1[j], '1 h w -> c h w', c=3)
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)
            
            log_metrics(model_pred[i], onn_pred[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx

