"""
Learns to predict the PSF of a given optical system.
"""
import einops
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dataset import create_dataset
from dataset.ZipDatasetWrapper import ZipDatasetWrapper
from models.archs import define_network
from models.base_model import BaseModel
from utils.dataset_utils import crop_arr, grayscale, merge_patches
from utils.image_utils import save_images_as_zip
from utils.lens_profiles import get_lens_profile
from utils.loss import Loss
from utils import log_image, log_metrics


from dflat.initialize import focusing_lens
from dflat.metasurface import reverse_lookup_optimize, load_optical_model
from dflat.propagation.propagators_legacy import PointSpreadFunction # Using the legacy version as it takes wavelength as a forward input instead of initialization input
from dflat.render import hsi_to_rgb
from dflat.plot_utilities import format_plot
from dflat.render import Fronto_Planar_Renderer_Incoherent

from utils.misc import fig_to_array


class PropLearn_model(BaseModel):
    def __init__(self, opt, logger):
        import matplotlib
        matplotlib.use("Agg")
        super(PropLearn_model, self).__init__(opt, logger)
        self.batch_size = opt.train.batch_size
        
        # ============ define DFlat PSF simulator and image generator
        settings = opt.dflat
        self.fov_rad = 2 * np.arctan(settings.out_dx_m[0]*settings.h/ (2*settings.f_distance_m))  # 2*arctan(s/2f)
        logger.info(f"FOV: {self.fov_rad/np.pi*180:.2f} degrees")

        # 1. initialize the target phase profile of the metasurface
        self.wavelength_set_m = np.array(settings.wavelength_set_m)  # if we are using hsi/rgb images, this should change
        self.depth_min = settings.depth_min
        self.depth_max = settings.depth_max
        
        self.out_size = [settings.h, settings.w]

        amp, phase, aperture, p_norm, p = get_lens_profile(
                        reverse_lookup=True, lens_type=settings.initialization.type,
                        h=settings.h, w=settings.w, 
                        in_dx_m=settings.in_dx_m, 
                        wavelength_set_m=self.wavelength_set_m, 
                        out_distance_m=settings.f_distance_m, 
                        aperture_radius_m=settings.aperture_radius_m, 
                        model_name=settings.model_name, 
                        method='nearest',
                        **settings.initialization,
        )
        
        # 3. load optical model
        self.optical_model = load_optical_model(settings.model_name).cuda()
        with torch.no_grad():
            est_amp, est_phase = self.optical_model(p, self.wavelength_set_m, pre_normalized=False)
        self.est_amp = est_amp.to(dtype=torch.float32, device='cuda')
        self.est_phase = est_phase.to(dtype=torch.float32, device='cuda')
        self.aperture = torch.from_numpy(aperture).to(dtype=torch.float32, device='cuda')


        # 4. setup PSF generators from phase, amp
        # Compute the point spread function given this broadband stack of field amplitude and phases
        self.PSF = PointSpreadFunction(
            in_size=[settings.h+1, settings.w+1],
            in_dx_m=settings.in_dx_m,
            out_distance_m=settings.sensor_distance_m,
            out_size=self.out_size,
            out_dx_m=settings.out_dx_m,
            out_resample_dx_m=None,
            radial_symmetry=False,
            diffraction_engine="ASM").cuda()

        # 5. renderer for image blurring
        self.renderer = Fronto_Planar_Renderer_Incoherent()
                
        # ========================== define generator network
        network_opt = opt.network
        network_opt['image_size'] = settings.h
        network_opt['output_channels'] = len(self.wavelength_set_m)
        self.net_g = define_network(network_opt)
        self.models.append(self.net_g)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)

    def feed_data(self, data, is_train=True):
        # get random angles and radial distance and convert them to cartesian coordinates
        polar = (torch.rand(self.batch_size) * 2  - 1) * self.fov_rad/2 * (self.global_step/self.opt.train.max_train_steps)
        alpha = (torch.rand(self.batch_size) * 2 - 1) * torch.pi
        r = torch.rand(self.batch_size)*(self.depth_max - self.depth_min) + self.depth_min

        x = r * torch.sin(polar) * torch.cos(alpha)
        y = r * torch.sin(polar) * torch.sin(alpha)
        z = r * torch.cos(polar)
        locations = torch.stack([x, y, z], dim=1).to(device=self.accelerator.device)
        
        psf_intensity, _ = self.PSF(
            self.est_amp,
            self.est_phase,
            self.wavelength_set_m,
            locations,
            aperture=self.aperture,
            normalize_to_aperture=True)
        data['psf_sum'] = psf_intensity.sum()
        data['psf_max'] = psf_intensity.max()
        assert data['psf_sum'] > 0, "No signal on sensor. Adjust FOV or sensor size"
        psf_intensity /= data['psf_max'] # normalize psf
        data['pos'] = locations # [Z 3]
        data['psf'] = psf_intensity[0,0] # [B=1 P=1 Z L H W]
        self.sample = data

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        psf = self.sample['psf']
        pred = self.net_g(self.sample['pos'])

        losses = self.criterion(pred, psf)

        # loss in Fourier space
        psf_ft = torch.fft.fftshift(torch.fft.fft2(psf, dim=(-2,-1)), dim=(-2,-1))
        pred_ft = torch.fft.fftshift(torch.fft.fft2(pred, dim=(-2,-1)), dim=(-2,-1))
        losses['ft_loss'] = F.mse_loss(((pred_ft.real/1e3)**2), 
                                       ((psf_ft.real/1e3)**2))
        # losses['ft_loss'] += F.mse_loss(((pred_ft.imag/1e3)**2), 
        #                                 ((psf_ft.imag/1e3)**2))
        losses['all'] = losses['all'] + 0.0001 * torch.log(losses['ft_loss'])

        self.accelerator.backward(losses['all'])
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        return losses

    def validate_step(self, batch, idx,lq_key,gt_key):
        if idx > 10:
            return idx+1
        self.feed_data(batch, is_train=False)
        if self.opt.val.patched:
            raise Exception()
        else: 
            pos = self.sample['pos']
            psf = self.sample['psf']
            out = self.net_g(pos)
            
        psf = psf.cpu().numpy()
        out = out.cpu().numpy()
        psf_ft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(psf, axes=(-2,-1)), axes=(-2,-1)).real))
        out_ft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(out, axes=(-2,-1)), axes=(-2,-1)).real))
        for i in range(len(psf)):
            idx += 1
            print(pos[i])
            fig, axes = plt.subplots(2, 2, figsize=(4*2, 4))
            ax = axes[0, 0]
            ax.set_title(f"GT")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(einops.rearrange(psf[i], 'c h w -> h w c'))
            fig.colorbar(im, cax=cax, orientation='vertical')


            ax = axes[1, 0]
            ax.set_title(f"GT FT")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(einops.rearrange(psf_ft[i], 'c h w -> h w c'))
            fig.colorbar(im, cax=cax, orientation='vertical')

            ax = axes[0, 1]
            ax.set_title(f"Pred")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(einops.rearrange(out[i], 'c h w -> h w c'))
            fig.colorbar(im, cax=cax, orientation='vertical')

            ax = axes[1, 1]
            ax.set_title(f"Pred FT")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(einops.rearrange(out_ft[i], 'c h w -> h w c'))
            fig.colorbar(im, cax=cax, orientation='vertical')

            img = fig_to_array(fig)
            plt.close(fig)
            log_image(self.opt, self.accelerator, img[None], f'{idx}', self.global_step)
            log_metrics(psf[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx

