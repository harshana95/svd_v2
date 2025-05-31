"""
Optimize Metasurface to get higher MTF for all depths (Designing all-in-focus lens)
"""
import os
import einops
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
import seaborn as sns

from dataset import create_dataset
from dataset.ZipDatasetWrapper import ZipDatasetWrapper

from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms

from models.archs import define_network
from models.base_model import BaseModel
from utils.lens_profiles import get_lens_profile
from utils.mtf import get_psf_cross_section, plot_1d_mtf, plot_psf_mtf, get_mtf, get_cutoff
from utils.dataset_utils import crop_arr, grayscale, merge_patches
from utils.image_utils import display_images
from utils.loss import Loss
from utils import log_image, log_metrics


from transformers import PreTrainedModel, PretrainedConfig
from dflat.initialize import focusing_lens
from dflat.metasurface import reverse_lookup_optimize, load_optical_model, optical_model
from dflat.propagation.propagators_legacy import PointSpreadFunction # Using the legacy version as it takes wavelength as a forward input instead of initialization input
from dflat.render import Fronto_Planar_Renderer_Incoherent

from utils.misc import fig_to_array
# Create a custom colormap
# White Blue Cyan Green Yellow Red
colors = ["#FFFFFF", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000"]  # Define custom colors
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)  # Create custom colormap


class MS_config(PretrainedConfig):
    def __init__(self, p_norm=None, **kwargs):
        super().__init__(**kwargs)
        self.p_norm = p_norm

class MS_arch(PreTrainedModel):
    def __init__(self, config):
        super(MS_arch, self).__init__(config)
        self.p_norm = nn.Parameter(torch.tensor(config.p_norm).to(torch.float32), requires_grad=True)


class DFlatAllFocus_model(BaseModel):
    def __init__(self, opt, logger):
        import matplotlib
        matplotlib.use("Agg")
        super(DFlatAllFocus_model, self).__init__(opt, logger)
        
        self.wavelength_val_set_m = np.array([400e-9, 550e-9, 700e-9])  # if we are using hsi/rgb images, this should change
        
        # ============ define DFlat PSF simulator and image generator
        # 1. initialize the target phase profile of the metasurface
        self.baseline = []
        settings = opt.dflat
        model_name = settings.model_name
        h, w = settings.h, settings.w
        self.wavelength_set_m = np.array(settings.wavelength_set_m)  # Design wavelengths
        self.ps_locs = torch.tensor([list(map(float, v)) for v in settings.ps_locs.values()])  # point source locations (we use a sparse grid)
        self.cutoff_mtf = float(settings.cutoff_mtf)
        self.depth_min = settings.depth_min
        self.depth_max = settings.depth_max
        settings['in_dx_m'] = [settings.aperture_radius_m/(h/2),settings.aperture_radius_m/(w/2)]
        settings['out_dx_m'] = [x*0.7 for x in settings['in_dx_m']]
        self.L = settings.in_dx_m[0]*h/2
        self.F_number = settings.out_distance_m/(2*self.L)  # f/D
        self.diff_spot_size = 1.22 * self.wavelength_set_m * settings.out_distance_m / (2*self.L)  # 1.22*lam*f/D
        self.fc = 1/(self.wavelength_set_m*1e3)/self.F_number  # cutoff frequency 1/lam/F#  in cycles/mm
        self.pixel_pitch = settings.out_dx_m

        logger.info(f"F#=f/D {self.F_number} Diff spot r=1.22*lam*f/D {self.diff_spot_size*1e6} um")
        logger.info(f"Sensor Nyquist freq = 0.5/pixelpitch {0.5/(settings['out_dx_m'][0]*1e3)} cycles/mm")
        logger.info(f"Optics cutoff freq = 1/lam/F# {self.fc} cycles/mm")
        logger.info(f"Aperture Radius {self.L*1e6}um out_distance {settings.out_distance_m*1e6}um")
        logger.info(f"in_dx_m {settings.in_dx_m} out_dx_m {settings.out_dx_m}")
        logger.info(f"Propagation matrix size ~{self.wavelength_set_m[None,:]/1e-6 * settings.out_distance_m/1e-6 /(np.array(settings.out_dx_m)/1e-6)[:,None]/(np.array(settings.in_dx_m)/1e-6)[:,None]}")
        

        # find the MS center positions of the array
        self.array_size = settings.array_size  # col, row
        self.array_spacing = settings.array_spacing # col, row
        self.MS_size = [(w+1)*settings.in_dx_m[0], (h+1)*settings.in_dx_m[1]]  # width, height
        self.MSArray_size = [self.array_size[0]*self.array_spacing[0], self.array_size[1]*self.array_spacing[1]] # width, height
        logger.info(f"MS size (width, height): {self.MS_size} MS Array size (width, height): {self.MSArray_size}")
        self.MS_pos = []
        for i in range(self.array_size[0]): # cols
            for j in range(self.array_size[1]): # rows
                self.MS_pos.append([i*self.array_spacing[0] - self.MSArray_size[0]/2,   # x
                                    j*self.array_spacing[1] - self.MSArray_size[1]/2,   # y
                                    0.0])                                               # z
                logger.info(f"MS {i},{j} center: {self.MS_pos[-1]}")
        self.MS_pos = torch.tensor(self.MS_pos)

        # find the out_size of the MS. Downscale to preserve memory
        downscale_factor = settings.downscale_factor
        self.out_size = [h//downscale_factor[0], w//downscale_factor[1]]

        # initialize shape parameters
        amp, phase, self.aperture, p_norm, p = get_lens_profile(True, 
            settings.initialization.type, h, w, settings.in_dx_m, self.wavelength_set_m, settings.out_distance_m,
            aperture_radius_m=self.L, model_name=model_name, **settings.initialization
        )
        
        # initialize shape parameters with noise
        self.ms_models = []
        self.logger.info(f"Shape parameters min: {p.min()} max: {p.max()}")
        self.logger.info(f"Shape parameters min: {p_norm.min()} max: {p_norm.max()}")
        for _ in range(len(self.MS_pos)):
            p_norm = p_norm #+ np.random.rand(*p_norm.shape) * p_norm.max() * 0.1  
            ms_model = MS_arch(MS_config(p_norm.tolist()))
            self.models.append(ms_model)
            self.ms_models.append(ms_model)
            # [B, H, W, D] where D = model.dim_in - 1 is the number of shape parameters
        
        
        # 3. load optical model
        self.optical_model = load_optical_model(model_name, verbose=True).cuda()
        
        # 4. setup PSF generators from phase, amp
        # Compute the point spread function given this broadband stack of field amplitude and phases
        self.PSF = PointSpreadFunction(
            in_size=[h+1, w+1],
            in_dx_m=settings.in_dx_m,
            out_distance_m=settings.out_distance_m,
            out_size=self.out_size,
            out_dx_m=settings.out_dx_m,
            out_resample_dx_m=None,
            radial_symmetry=False,
            diffraction_engine=settings.diffraction_engine).cuda()
        # self.baseline.append({'name':'Start', 'amp':amp, 'phase':phase, 'aperture':self.aperture,'p_norm':p_norm, 'p':p, 'gen':self.PSF})  # add staring state for comparison
        # with torch.no_grad():
        #     est_amp, est_phase = self.optical_model(torch.tensor(p).cuda(), self.wavelength_set_m, pre_normalized=False)
        # self.baseline.append({'name':'Start est', 'amp':est_amp.cpu().numpy(), 'phase':est_phase.cpu().numpy(), 'aperture':self.aperture, 'p_norm':p_norm, 'p':p, 'gen':self.PSF})  # add staring state for comparison

        # 5. renderer for image blurring
        self.renderer = Fronto_Planar_Renderer_Incoherent()
        
        # ================================================================ baseline amp, phase
        out_distance_m = settings.out_distance_m
        # # baseline with lens focusing objects at different depths
        # for depth in np.logspace(np.log10(1e-2), np.log10(1e0), 3):
        #     amp, phase, aperture, p_norm, p = get_lens_profile(True, 'focusing_lens', h, w, settings.in_dx_m, self.wavelength_set_m, settings.out_distance_m,
        #         aperture_radius_m=self.L, model_name=model_name, depth_set_m=[depth]*len(self.wavelength_set_m)
        #     )
        #     self.baseline.append({'name':f'{depth:.4f}',
        #                           'amp': amp, 'phase': phase, 'aperture': aperture, 
        #                           'p_norm': p_norm, 'p': p, 'gen': None})
        # # baseline with multiplexed lens
        # _, _, _, p_norm, p = get_lens_profile(True, 'multiplexed', h, w, settings.in_dx_m, self.wavelength_set_m, settings.out_distance_m,
        #         aperture_radius_m=self.L, model_name=model_name, depth_set_array_m=[1e-2, 1e-1, 1e0, 1e1], grid_size=[2,2]
        # )
        # self.baseline.append({'name':'Multiplexed pixel',
        #                         'amp': None, 'phase': None, 'aperture': aperture, 
        #                         'p_norm': p_norm, 'p': p, 'gen': None})
        # # baseline with multiplexed cell lens
        # _, _, _, p_norm, p = get_lens_profile(True, 'multiplexed_cell', h, w, settings.in_dx_m, self.wavelength_set_m, settings.out_distance_m,
        #         aperture_radius_m=self.L, model_name=model_name, depth_set_array_m=[1e-2, 1e-1, 1e1], cell_size=16, grid_size=[1,3]
        # )
        # self.baseline.append({'name':'Multiplexed cell',
        #                         'amp': None, 'phase': None, 'aperture': aperture, 
        #                         'p_norm': p_norm, 'p': p, 'gen': None})
        
        # baselines on Arka's paper
        
        # Metalens
        amp, phase, aperture, p_norm, p = get_lens_profile(True, 'metalens', h, w, settings.in_dx_m, self.wavelength_set_m, settings.out_distance_m, aperture_radius_m=self.L, model_name=model_name, f=out_distance_m)
        self.baseline.append({'name':f'Metalens',
                                'amp': amp, 'phase': phase, 'aperture': aperture, 
                                'p_norm': p_norm, 'p': p, 
                                'gen': PointSpreadFunction(
                                            in_size=[h+1, w+1],
                                            in_dx_m=settings.in_dx_m,
                                            out_distance_m=out_distance_m,
                                            out_size=self.out_size,
                                            out_dx_m=settings.out_dx_m,
                                            out_resample_dx_m=None,
                                            radial_symmetry=False,
                                            diffraction_engine=settings.diffraction_engine).cuda()})
        # Cubic
        amp, phase, aperture, p_norm, p = get_lens_profile(True, 'cubic', h, w, settings.in_dx_m, self.wavelength_set_m, settings.out_distance_m, aperture_radius_m=self.L, model_name=model_name, f=out_distance_m)
        self.baseline.append({'name':f'Cubic',
                                'amp': amp, 'phase': phase, 'aperture': aperture, 
                                'p_norm': p_norm, 'p': p, 
                                'gen': PointSpreadFunction(
                                            in_size=[h+1, w+1],
                                            in_dx_m=settings.in_dx_m,
                                            out_distance_m=out_distance_m,
                                            out_size=self.out_size,
                                            out_dx_m=settings.out_dx_m,
                                            out_resample_dx_m=None,
                                            radial_symmetry=False,
                                            diffraction_engine=settings.diffraction_engine).cuda()})
        # Shifted Axicon
        amp, phase, aperture, p_norm, p = get_lens_profile(True, 'shifted_axicon', h, w, settings.in_dx_m, self.wavelength_set_m, settings.out_distance_m, aperture_radius_m=self.L, model_name=model_name, s_1=out_distance_m*50/55, s_2=out_distance_m)
        self.baseline.append({'name':f'Shifted-axicon',
                                'amp': amp, 'phase': phase, 'aperture': aperture, 
                                'p_norm': p_norm, 'p': p, 
                                'gen': PointSpreadFunction(
                                            in_size=[h+1, w+1],
                                            in_dx_m=settings.in_dx_m,
                                            out_distance_m=out_distance_m,
                                            out_size=self.out_size,
                                            out_dx_m=settings.out_dx_m,
                                            out_resample_dx_m=None,
                                            radial_symmetry=False,
                                            diffraction_engine=settings.diffraction_engine).cuda()})
        # Log asphere
        amp, phase, aperture, p_norm, p = get_lens_profile(True, 'log_asphere', h, w, settings.in_dx_m, self.wavelength_set_m, settings.out_distance_m, aperture_radius_m=self.L, model_name=model_name, s_1=out_distance_m*50/55, s_2=out_distance_m)
        self.baseline.append({'name':f'Log asphere',
                                'amp': amp, 'phase': phase, 'aperture': aperture, 
                                'p_norm': p_norm, 'p': p, 
                                'gen': PointSpreadFunction(
                                            in_size=[h+1, w+1],
                                            in_dx_m=settings.in_dx_m,
                                            out_distance_m=out_distance_m,
                                            out_size=self.out_size,
                                            out_dx_m=settings.out_dx_m,
                                            out_resample_dx_m=None,
                                            radial_symmetry=False,
                                            diffraction_engine=settings.diffraction_engine).cuda()})
        
        # #SQUBIC
        # amp, phase, aperture, p_norm, p = get_lens_profile(True, 'squbic', h, w, settings.in_dx_m, self.wavelength_set_m, settings.out_distance_m, aperture_radius_m=self.L, model_name=model_name)
        # self.baseline.append({'name':f'SQUBIC',
        #                         'amp': amp, 'phase': phase, 'aperture': aperture, 
        #                         'p_norm': p_norm, 'p': p, 
        #                         'gen': PointSpreadFunction(
        #                                     in_size=[h+1, w+1],
        #                                     in_dx_m=settings.in_dx_m,
        #                                     out_distance_m=out_distance_m,
        #                                     out_size=self.out_size,
        #                                     out_dx_m=settings.out_dx_m,
        #                                     out_resample_dx_m=None,
        #                                     radial_symmetry=False,
        #                                     diffraction_engine=settings.diffraction_engine).cuda()})

        # Quadratic
        amp, phase, aperture, p_norm, p = get_lens_profile(True, 'quadratic', h, w, settings.in_dx_m, self.wavelength_set_m, 1e-3, aperture_radius_m=self.L, model_name=model_name)
        self.baseline.append({'name':f'Quadratic',
                                'amp': amp, 'phase': phase, 'aperture': aperture, 
                                'p_norm': p_norm, 'p': p, 
                                'gen': PointSpreadFunction(
                                            in_size=[h+1, w+1],
                                            in_dx_m=settings.in_dx_m,
                                            out_distance_m=1e-3,
                                            out_size=self.out_size,
                                            out_dx_m=settings.out_dx_m,
                                            out_resample_dx_m=None,
                                            radial_symmetry=False,
                                            diffraction_engine=settings.diffraction_engine).cuda()})
        # Quadratic with cubic
        amp, phase, aperture, p_norm, p = get_lens_profile(True, 'quadratic_cubic', h, w, settings.in_dx_m, self.wavelength_set_m, 1e-3, aperture_radius_m=self.L, model_name=model_name)
        self.baseline.append({'name':f'Quadratic Cubic',
                                'amp': amp, 'phase': phase, 'aperture': aperture, 
                                'p_norm': p_norm, 'p': p, 
                                'gen': PointSpreadFunction(
                                            in_size=[h+1, w+1],
                                            in_dx_m=settings.in_dx_m,
                                            out_distance_m=1e-3,
                                            out_size=self.out_size,
                                            out_dx_m=settings.out_dx_m,
                                            out_resample_dx_m=None,
                                            radial_symmetry=False,
                                            diffraction_engine=settings.diffraction_engine).cuda()})

        
        self.prepare_trackers()
        self.plot_baseline_figs(do_plot=True)

        # ========================== define generator network
        self.net_g = define_network(opt.network)
        self.models.append(self.net_g)

        if self.is_train:
            self.init_training_settings()
    
    def plot_baseline_figs(self, do_plot=True):
        logger = self.logger
        settings = self.opt.dflat
        out_distance_m = settings.out_distance_m
        h,w = settings.h, settings.w
        # =============================== plot figs
        # Plot for each baseline
        ps_locs_small = []
        for z1 in range(int(np.log10(self.depth_min)), int(np.log10(self.depth_max))+1):
            ps_locs_small.append([0., 0., 10**z1])
        ps_locs_small = torch.tensor(ps_locs_small).to('cuda')
        ps_locs_large = []
        for z1 in np.logspace(np.log10(self.depth_min), np.log10(self.depth_max), 21):
            ps_locs_large.append([0., 0., z1])
        ps_locs_large = torch.tensor(ps_locs_large).to('cuda')
        self.baseline_cutoff = []
        N = len(self.baseline)
        N1 = len(self.wavelength_set_m)
        N2 = len(self.wavelength_val_set_m)
        fig1, axes1 = plt.subplots(1, N, figsize=(4*N, 4)) # aperture
        fig2, axes2 = plt.subplots(N1, N, figsize=(4*N, 4*N1)) # phase
        fig3, axes3 = plt.subplots(N2, N, figsize=(4*N, 4*N2)) # est phase
        fig4, axes4 = plt.subplots(N1, N, figsize=(4*N, 4*N1)) # shape params
        fig5, axes5 = plt.subplots(N2, N, figsize=(4*N, 4*N2)) # psf cross
        fig6, axes6 = plt.subplots(N2, N, figsize=(4*N, 4*N2)) # psf 
        fig7, axes7 = plt.subplots(N2, N, figsize=(4*N, 4*N2)) # mtf
        fig8, axes8 = plt.subplots(1, 1, figsize=(4, 4)) # mtf 1d
        extent = np.array([-self.L, self.L, -self.L, self.L])*1e6
        for i in range(N):
            psf_gen = self.baseline[i]['gen']
            psf_gen = self.PSF if psf_gen is None else psf_gen
            psf_gen = psf_gen.cuda()
            amp = self.baseline[i]['amp']
            phase = self.baseline[i]['phase']
            aperture = self.baseline[i]['aperture']
            p_norm = self.baseline[i]['p_norm']
            p = self.baseline[i]['p']
            name = self.baseline[i]['name']
            logger.info(f"Baseline {name}")

            with torch.no_grad():
                logger.info("Estimating phase from shape parameters...")
                est_amp, est_phase = self.optical_model(p, self.wavelength_val_set_m, pre_normalized=False)
            
            logger.info("Calculating PSF...")
            psf_intensity1, _ = psf_gen(est_amp,est_phase,self.wavelength_val_set_m,ps_locs_small,aperture=aperture,normalize_to_aperture=True)
            batch = len(ps_locs_small)
            psf_intensity2, _ = psf_gen(est_amp,est_phase,self.wavelength_val_set_m,ps_locs_large[:batch],aperture=aperture,normalize_to_aperture=True)
            for bs in range(batch, len(ps_locs_large), batch):
                psf_intensity2_batch, _ = psf_gen(est_amp,est_phase,self.wavelength_val_set_m,ps_locs_large[bs:bs+batch],aperture=aperture,normalize_to_aperture=True)
                psf_intensity2 = torch.cat([psf_intensity2, psf_intensity2_batch], dim=2)
            
            logger.info("Getting cutoff...")
            cf, ff, mtf1d, mtf = get_cutoff(psf_intensity1, self.cutoff_mtf, pixel_pitch=psf_gen.propagator.out_dx_m[0])
            cf2, ff2, mtf1d2, mtf2 = get_cutoff(psf_intensity2, self.cutoff_mtf, pixel_pitch=psf_gen.propagator.out_dx_m[0])
            psf = psf_intensity1[0, 0].cpu().numpy()

            self.baseline_cutoff.append(cf2)
            if not do_plot: continue

            logger.info("Plotting...")
            img = plot_1d_mtf(mtf1d, ff, ps_locs_small, self.cutoff_mtf, mask=0.025, fc=self.fc)
            log_image(self.opt, self.accelerator, img[None], f"1d_mtf_base_{name}", self.global_step)
            img = plot_psf_mtf(psf,mtf,ps_locs_small, settings['out_dx_m'][0]*h/2)
            log_image(self.opt, self.accelerator, img[None], f"2d_psf_mtf_base_{name}", self.global_step)

            # plot mtf 1d
            ax = axes8
            ax.plot(ff[:len(ff)//2], mtf1d[0,0,:len(ff)//2], label=name)
            if i==len(self.baseline)-1: ax.legend()

            # plot the aperture
            ax = axes1[i]
            ax.set_title(f"{name}")
            if i == 0: ax.set_ylabel("y (um)")
            else: ax.yaxis.set_visible(False)
            ax.set_xlabel("x (um)")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(einops.rearrange(aperture[0][0], 'c h w -> h w c'), extent=extent, cmap='gray')
            fig1.colorbar(im, cax=cax, orientation='vertical')

            # plot the phase
            for j in range(len(self.wavelength_set_m)):
                ax = axes2[j, i] if len(self.wavelength_set_m) > 1 else axes2[i]
                if j == 0: ax.set_title(f"{name}")
                if i == 0: ax.set_ylabel("y (um)")
                else: ax.yaxis.set_visible(False)
                if j == len(self.wavelength_set_m)-1:ax.set_xlabel("x (um)")
                else: ax.xaxis.set_visible(False)
                if phase is not None:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    im = ax.imshow(einops.rearrange(phase[0][0][j:j+1]*aperture[0][0], 'c h w -> (c h) w'), extent=extent, cmap='viridis')
                    fig2.colorbar(im, cax=cax, orientation='vertical')

            # plot estimated phase
            for j in range(len(self.wavelength_val_set_m)):
                ax = axes3[j, i] if len(self.wavelength_val_set_m) > 1 else axes3[i]
                if j == 0: ax.set_title(f"{name}")
                if i == 0: ax.set_ylabel("y (um)")
                else: ax.yaxis.set_visible(False)
                if j == len(self.wavelength_val_set_m)-1:ax.set_xlabel("x (um)")
                else: ax.xaxis.set_visible(False)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                im = ax.imshow(einops.rearrange(est_phase.cpu().numpy()[0][0][j:j+1]*aperture[0][0], 'c h w -> (c h) w'), extent=extent, cmap='viridis')
                fig3.colorbar(im, cax=cax, orientation='vertical')

            # plot shape params
            for j in range(len(self.wavelength_set_m)):
                ax = axes4[j, i] if len(self.wavelength_set_m) > 1 else axes4[i]
                if j == 0: ax.set_title(f"{name}")
                if i == 0: ax.set_ylabel("y (um)")
                else: ax.yaxis.set_visible(False)
                if j == len(self.wavelength_set_m)-1:ax.set_xlabel("x (um)")
                else: ax.xaxis.set_visible(False)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                im = ax.imshow(p_norm[j]*einops.rearrange(aperture[0][0], 'c h w -> h w c'), extent=extent, cmap='viridis')
                fig4.colorbar(im, cax=cax, orientation='vertical')

            inf_ps_loc = np.array([[0,0,1e3]])
            psf_intensity, _ = psf_gen(est_amp.cuda(), est_phase.cuda(), self.wavelength_val_set_m, inf_ps_loc, aperture=aperture,normalize_to_aperture=True)
            cf, ff, mtf1d, mtf = get_cutoff(psf_intensity, self.cutoff_mtf, pixel_pitch=psf_gen.propagator.out_dx_m[0])
            psf = einops.rearrange(psf_intensity[0,0].cpu().numpy(), 'b c h w -> b h w c')[0]
            mtf = einops.rearrange(mtf, 'b c h w -> b h w c')[0]

            sensor_distances = np.linspace(out_distance_m/2, out_distance_m*3/2, 25)
            psf_cross = get_psf_cross_section(est_amp, est_phase, aperture, inf_ps_loc, sensor_distances, 
                                              h, w, settings.in_dx_m, settings.out_dx_m, self.wavelength_val_set_m, settings.diffraction_engine)
            
            # plot psf cross
            sensor_hf = settings.out_dx_m[0]*self.out_size[0]/2
            for j in range(len(self.wavelength_val_set_m)):
                ax = axes5[j, i] if len(self.wavelength_val_set_m) > 1 else axes5[i]
                if j == 0: ax.set_title(f"{name}")
                if i == 0: ax.set_ylabel("y (mm)")
                else: ax.yaxis.set_visible(False)
                if j == len(self.wavelength_val_set_m)-1:ax.set_xlabel("Sensor distance (mm)")
                else: ax.xaxis.set_visible(False)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                im = ax.imshow(psf_cross[:, :, j], extent=[sensor_distances[0]*1e3, sensor_distances[-1]*1e3, -sensor_hf*1e3, sensor_hf*1e3], cmap=custom_cmap, aspect='auto')
                fig5.colorbar(im, cax=cax, orientation='vertical')
            
            # plot psf
            for j in range(len(self.wavelength_val_set_m)):
                ax = axes6[j, i] if len(self.wavelength_val_set_m) > 1 else axes6[i]
                if j == 0: ax.set_title(f"{name}")
                if i == 0: ax.set_ylabel("y (mm)")
                else: ax.yaxis.set_visible(False)
                if j == len(self.wavelength_val_set_m)-1:ax.set_xlabel("x (mm)")
                else: ax.xaxis.set_visible(False)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                im = ax.imshow(psf[:,:,j], extent=np.array([-1, 1, -1, 1])*sensor_hf*1e3, cmap=custom_cmap)
                fig6.colorbar(im, cax=cax, orientation='vertical')

            # plot mtf
            for j in range(len(self.wavelength_val_set_m)):
                ax = axes7[j, i] if len(self.wavelength_val_set_m) > 1 else axes7[i]
                if j == 0: ax.set_title(f"{name}")
                if i == 0: ax.set_ylabel("fy")
                else: ax.yaxis.set_visible(False)
                if j == len(self.wavelength_val_set_m)-1:ax.set_xlabel("fx")
                else: ax.xaxis.set_visible(False)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                im = ax.imshow(mtf[:,:,j], extent=np.array([-1,1,-1,1])*(self.out_size[0]/2), cmap='inferno')
                fig7.colorbar(im, cax=cax, orientation='vertical')

        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        fig4.tight_layout()
        fig5.tight_layout()
        fig6.tight_layout()
        fig7.tight_layout()
        fig8.tight_layout()
        fig_1 = fig_to_array(fig1)
        fig_2 = fig_to_array(fig2)
        fig_3 = fig_to_array(fig3)
        fig_4 = fig_to_array(fig4)
        fig_5 = fig_to_array(fig5)
        fig_6 = fig_to_array(fig6)
        fig_7 = fig_to_array(fig7)
        fig_8 = fig_to_array(fig8)
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)
        plt.close(fig5)
        plt.close(fig6)
        plt.close(fig7)
        plt.close(fig8)
        log_image(self.opt, self.accelerator, fig_1[None], f'baselines_aperture', self.global_step)
        log_image(self.opt, self.accelerator, fig_2[None], f'baselines_phase', self.global_step)
        log_image(self.opt, self.accelerator, fig_3[None], f'baselines_est_phase', self.global_step)
        log_image(self.opt, self.accelerator, fig_4[None], f'baselines_shape', self.global_step)
        log_image(self.opt, self.accelerator, fig_5[None], f'baselines_psf_cross', self.global_step)
        log_image(self.opt, self.accelerator, fig_6[None], f'baselines_psf', self.global_step)
        log_image(self.opt, self.accelerator, fig_7[None], f'baselines_mtf', self.global_step)
        log_image(self.opt, self.accelerator, fig_8[None], f'baselines_mtf1d', self.global_step)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)

    def setup_dataloaders(self):
        # create train and validation dataloaders
        train_set1 = create_dataset(self.opt.datasets.train1)
        train_set2 = create_dataset(self.opt.datasets.train2)
        self.dataloader = DataLoader(
            ZipDatasetWrapper({'1': train_set1, '2': train_set2}, transforms=transforms.Compose([]), random=False),
            shuffle=self.opt.datasets.train1.use_shuffle,
            batch_size=self.opt.train.batch_size,
            num_workers=self.opt.datasets.train1.get('num_worker_per_gpu', 1),
        )
        self.dataloader.dataset.gt_key = train_set1.gt_key
        self.dataloader.dataset.lq_key = train_set1.lq_key
                
        val_set1 = create_dataset(self.opt.datasets.val1)
        val_set2 = create_dataset(self.opt.datasets.val2)
        self.test_dataloader = DataLoader(
            ZipDatasetWrapper({'1': val_set1, '2': val_set2}, transforms=transforms.Compose([]), random=False),
            shuffle=self.opt.datasets.val1.use_shuffle,
            batch_size=self.opt.val.batch_size,
            num_workers=self.opt.datasets.val1.get('num_worker_per_gpu', 1),
        )
        self.test_dataloader.dataset.gt_key = val_set1.gt_key
        self.test_dataloader.dataset.lq_key = val_set1.lq_key

    def setup_optimizers(self):
        opt = self.opt.train.optim

        # Optimizer creation
        optimizer_class = torch.optim.AdamW
        optim_params = []
        optim_params_ms = []
        for model in self.ms_models:
            for k,v in model.named_parameters():
                if v.requires_grad:
                    optim_params_ms.append(v)

        for model in self.models:
            if type(model) == MS_arch:
                continue
            for k, v in model.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
        self.logger.info(f"Total parameters MS {sum([param.numel() for param in optim_params_ms])}")
        self.logger.info(f"Total parameters Network {sum([param.numel() for param in optim_params])}")
        # add shape parameters as a parameter to optimize
        params_to_optimize = [{'params': optim_params, 'lr': opt.learning_rate}, 
                              {'params': optim_params_ms, 'lr': opt.learning_rate_ms}]
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(opt.adam_beta1, opt.adam_beta2),
            weight_decay=opt.adam_weight_decay,
            eps=opt.adam_epsilon,
            )
        self.optimizers.append(optimizer)

    def feed_data(self, data, is_train=True):
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        fg = data['foreground_1']
        bg = data['background_2']
        mask = data['mask_1']
        mask = torch.clamp(mask, 1e-6, 1 - 1e-6)  # if we don't clamp, we get NaNs.   
        device = mask.device
        batch_size = mask.shape[0]

        data[gt_key] = []
        data[lq_key] = []
        data['depth'] = []
        data['psf_intensity'] = [] # x2 
        data['ps_locs'] = [] # x2
        for h in range(len(self.MS_pos)):
            p = self.optical_model.denormalize(self.ms_models[h].p_norm)
            est_amp, est_phase = self.optical_model(p, self.wavelength_set_m, pre_normalized=False)
            
            est_amp = est_amp.to(dtype=torch.float32, device=device)
            est_phase = est_phase.to(dtype=torch.float32, device=device)

            # generate the ground truth images [all-in-focus image, depth map]
            bg_depth = (torch.rand(batch_size) if is_train else torch.ones(batch_size))*(self.depth_max - self.depth_min) + self.depth_min
            fg_depth = (torch.rand(batch_size) if is_train else torch.zeros(batch_size))*(bg_depth - self.depth_min) + self.depth_min

            # simulate PSF for current p
            n_ps = len(self.ps_locs)
            ps_locs = self.ps_locs - self.MS_pos[h:h+1, :2]  # shift the PSF locations to the MS center
            
            ps_locs_batch = einops.repeat(ps_locs, 'n d -> (b n) d', b=batch_size)
            fg_ps_locs = torch.cat([ps_locs_batch, einops.repeat(fg_depth, 'b -> (b n) 1', n=n_ps)], dim=-1)
            bg_ps_locs = torch.cat([ps_locs_batch, einops.repeat(bg_depth, 'b -> (b n) 1', n=n_ps)], dim=-1)

            fg_psf_intensity, _ = self.PSF(
                est_amp,
                est_phase,
                self.wavelength_set_m,
                fg_ps_locs,
                aperture=self.aperture,
                normalize_to_aperture=True)
            bg_psf_intensity, _ = self.PSF(
                est_amp,
                est_phase,
                self.wavelength_set_m,
                bg_ps_locs,
                aperture=self.aperture,
                normalize_to_aperture=True)
            fg_psf_intensity = einops.rearrange(fg_psf_intensity, '1 p (B N) c h w -> B p N c h w', B=batch_size, N=n_ps)
            bg_psf_intensity = einops.rearrange(bg_psf_intensity, '1 p (B N) c h w -> B p N c h w', B=batch_size, N=n_ps)
            
            # Need to pass inputs like
            # psf has shape  [B P Z L H W]
            # scene radiance [B P Z L H W]
            # out shape      [B P Z L H W]
            # if not is_train:
            mask_meas = self.renderer(fg_psf_intensity, einops.rearrange(mask, 'b c h w -> b 1 1 c h w'), rfft=True, crop_to_psf_dim=False)
            mask_meas = self.renderer.rgb_measurement(mask_meas, self.wavelength_set_m, gamma=True, process='demosaic')
            fg_meas = self.renderer(fg_psf_intensity, einops.rearrange(fg, 'b c h w -> b 1 1 c h w'), rfft=True, crop_to_psf_dim=False)
            fg_meas = self.renderer.rgb_measurement(fg_meas, self.wavelength_set_m, gamma=True, process='demosaic')

            bg_meas = self.renderer(bg_psf_intensity, einops.rearrange(bg, 'b c h w -> b 1 1 c h w'), rfft=True, crop_to_psf_dim=False)
            bg_meas = self.renderer.rgb_measurement(bg_meas, self.wavelength_set_m, gamma=True, process='demosaic')
            
            fg_meas = fg_meas[:, 0, 0]  # no polarization
            bg_meas = bg_meas[:, 0, 0]
            mask_meas = mask_meas[:, 0, 0]
            if bg.shape[-3] == 1:
                gray = grayscale()
                bg_meas = gray(bg_meas)
                fg_meas = gray(fg_meas)
                mask_meas = gray(mask_meas)
            
            data[gt_key].append(bg*(1 - mask) + fg*mask)  # alpha clipping and merging fg and bg  
            data[lq_key].append(bg_meas*(1 - mask_meas) + fg_meas*mask_meas)  # alpha clipping and merging fg and bg
            depth = bg_depth[:, None, None, None].to(device)*(1-mask) + fg_depth[:, None, None, None].to(device) *mask  # depth map
            data['depth'].append((depth - self.depth_min)/(self.depth_max - self.depth_min))
            data['ps_locs'].append(torch.cat([fg_ps_locs, bg_ps_locs], dim=0))
            data['psf_intensity'].append(torch.cat([fg_psf_intensity, bg_psf_intensity], dim=0))
        # if not is_train:
        data[gt_key] = torch.cat(data[gt_key], dim=0)
        data[lq_key] = torch.cat(data[lq_key], dim=0)
        data['depth'] = torch.cat(data['depth'], dim=0)
        data['psf_intensity'] = torch.cat(data['psf_intensity'], dim=0)
        data['ps_locs'] = torch.cat(data['ps_locs'], dim=0)
        self.sample = data
        if self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key, 'depth'], opt=self.opt.train if is_train else self.opt.val)
    

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        freqs, mtf_1d, mtf = get_mtf(self.sample['psf_intensity'], pixel_pitch=self.pixel_pitch[0]) 
        # compute loss to maximize MTF
        loss = torch.mean(torch.nn.functional.sigmoid(-(mtf_1d-self.cutoff_mtf)) 
                           * (1/torch.sum(mtf_1d, dim=-1, keepdim=True))
                           * (1/freqs))
        losses = {'all': loss, 'mtf_loss': loss}

        preds = self.net_g(self.sample[self.dataloader.dataset.lq_key])
        loss = self.criterion(preds, self.sample[self.dataloader.dataset.gt_key])
        losses['image_loss'] = loss['all']
        losses['all'] += loss['all']

        self.accelerator.backward(losses['all'])
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        return losses
    
    
    def plot_system_figs(self, device):
        phases = []
        shape_p = []
        ps_locs_small = []
        for z1 in range(int(np.log10(self.depth_min)), int(np.log10(self.depth_max))+1):
            ps_locs_small.append([0., 0., 10**z1])
        ps_locs_small = torch.tensor(ps_locs_small).to('cuda')
        ps_locs_large = []
        for z1 in np.logspace(np.log10(self.depth_min), np.log10(self.depth_max), 21):
            ps_locs_large.append([0., 0., z1])
        ps_locs_large = torch.tensor(ps_locs_large).to('cuda')
        for h in range(len(self.MS_pos)):
            p = self.optical_model.denormalize(self.ms_models[h].p_norm)
            est_amp, est_phase = self.optical_model(p, self.wavelength_val_set_m, pre_normalized=False)
            phases.append(einops.rearrange(est_phase[0][0].cpu().numpy(), 'c h w -> h w c'))
            shape_p.append(self.ms_models[h].p_norm[0].cpu().numpy())  # b h w d
            est_amp = est_amp.to(dtype=torch.float32, device=device)
            est_phase = est_phase.to(dtype=torch.float32, device=device)

            psf_intensity1, _ = self.PSF(est_amp,est_phase, self.wavelength_val_set_m, ps_locs_small,aperture=self.aperture,normalize_to_aperture=True)
            batch = len(ps_locs_small)
            psf_intensity2, _ = self.PSF(est_amp,est_phase,self.wavelength_val_set_m,ps_locs_large[:batch],aperture=self.aperture,normalize_to_aperture=True)
            for bs in range(batch, len(ps_locs_large), batch):
                psf_intensity2_batch, _ = self.PSF(est_amp,est_phase,self.wavelength_val_set_m,ps_locs_large[bs:bs+batch],aperture=self.aperture,normalize_to_aperture=True)
                psf_intensity2 = torch.cat([psf_intensity2, psf_intensity2_batch], dim=2)
            cutoff_at_z1, freqs, mtf1d, mtf = get_cutoff(psf_intensity1, self.cutoff_mtf, pixel_pitch=self.pixel_pitch[0])
            cutoff_at_z12, freqs2, mtf1d2, mtf2 = get_cutoff(psf_intensity2, self.cutoff_mtf, pixel_pitch=self.pixel_pitch[0])
            psf = psf_intensity1[0, 0].cpu().numpy()
            
            img = plot_1d_mtf(mtf1d, freqs, ps_locs_small, self.cutoff_mtf, mask=0.025, fc=self.fc)
            log_image(self.opt, self.accelerator, img[None], f'mtf_head{h}', self.global_step)
            img = plot_psf_mtf(psf, mtf, ps_locs_small, self.L)
            log_image(self.opt, self.accelerator, img[None], f'psf_mtf_head{h}', self.global_step)
            
            # Plot the cutoff spatial frequency for each z1
            cutoff = np.concatenate([np.array(cutoff_at_z12)[None], self.baseline_cutoff], axis=0) # include current cutoff
            n, c, l = cutoff.shape
            data = pd.DataFrame({
                'cutoff': cutoff.flatten(),
                'distance': einops.repeat(ps_locs_large[:, 2].cpu().numpy(), 'l -> n c l', n=n, c=c).flatten(),
                'wavelength': einops.repeat(np.arange(c), 'c -> n c l', n=n, l=l).flatten(),
                'lens': einops.repeat(np.array(['current']+[self.baseline[i]['name'] for i in range(n-1)]), 'n -> n c l', c=c, l=l).flatten(),
            })
            fig, axes = plt.subplots(c, 1, figsize=(8, 4*c))
            for i in range(c):
                ax = sns.lineplot(data=data[data['wavelength']==i], x='distance', y='cutoff', hue='lens', style='wavelength', palette='tab20', ax=axes[i], legend=True if i==0 else False)
                if i==0: sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                ax.set(xscale='log', yscale='log')
            plt.tight_layout(rect=[0,0,0.8,1])
            fig_np = fig_to_array(fig)
            plt.close(fig)
            log_image(self.opt, self.accelerator, fig_np[None], f'cutoff_head{h}', self.global_step)
            

        fig = display_images({"phase": np.stack(phases), 'shape_p': np.stack(shape_p)}, size=4)
        fig_np = fig_to_array(fig)
        plt.close(fig)
        log_image(self.opt, self.accelerator, fig_np[None], f'phase', self.global_step)
            
    def validate_step(self, batch, idx,lq_key,gt_key):
        if idx == 0: # generate same image at all validate steps
            self.plot_system_figs('cuda')

            # # Write this lens to a gds file for fabrication
            # for h in range(len(self.MS_pos)):
            #     pfab= self.ms_models[h].p_norm.detach().cpu().numpy().squeeze(0)

            #     mask = np.ones(pfab.shape[0:2])
            #     cell_size = self.opt.dflat.in_dx_m
            #     block_size = self.opt.dflat.in_dx_m # This can be used to repeat the cell as a larger block

            #     from dflat.GDSII.assemble import assemble_cylinder_gds
            #     savepath = os.path.join(self.opt.path.experiments_root, f'ms_head_{h}_{self.global_step}.gds')
            #     assemble_cylinder_gds(pfab, mask, cell_size, block_size, savepath=savepath)

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
            log_image(self.opt, self.accelerator, image1, f'{idx}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx}', self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
            log_metrics(gt[i], lq[i], self.opt.val.metrics, self.accelerator, self.global_step, comment='lq')
        return idx
    