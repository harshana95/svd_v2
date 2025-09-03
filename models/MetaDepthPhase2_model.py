"""
Training a reconstruction neural network, with 1st metasurface constrained. 
Use an oracle equation to get the wavefront before 2nd MS
Simulate 2nd MS using Dflat.
Fixed depth
simultaneous depth estimation and image reconstruction.
"""
import gc
import einops
from matplotlib import pyplot as plt
import numpy as np
import torch
from os import path as osp
from tqdm import tqdm
from torchvision import transforms
from dflat.propagation.propagators_legacy import PointSpreadFunction 
from dflat.initialize import focusing_lens
from dflat.metasurface import load_optical_model, reverse_lookup_optimize


from utils.PSFfromWavefront import PointSpreadFunctionNew

from models.archs import define_network
from models.base_model import BaseModel
from utils.dflat_utils import reverse_lookup_search_Nanocylinders_TiO2_U300nm_H600nm
from utils.image_utils import display_images
from utils.mtf import get_mtf
from utils.zernike import phase2psf, zernike_poly
from utils.loss import Loss

from utils.dataset_utils import crop_arr, merge_patches
from utils.misc import fig_to_array, log_image, log_metrics

class MetaDepthPhase2_model(BaseModel):
    def __init__(self, opt, logger):
        super(MetaDepthPhase2_model, self).__init__(opt, logger)
        torch.autograd.set_detect_anomaly(True)
        self.image_size = opt.image_size
        self.depth = 1e3
        self.fov = 20/180*np.pi  # can we calculate this?
        self.in_size = [2049,2049]
        self.in_dx_m=[2.5e-6, 2.5e-6]
        self.out_size=[2048, 2048]
        self.out_dx_m=[2.74e-6, 2.74e-6] # Basler dart M dmA4096-9gm (Mono) pixel size multiply 3
        radial_symmetry=False
        diffraction_engine="Fresnel"

        self.cutoff_mtf = 0.1

        self.out_distance_m=10e-3
        self.second_metasurface_depth_m=1e-3
        second_metasurface_psf_aperture_r = 0.1e-3
        aperture_radius_m=2e-3
        self.wavelength_set_m=[532e-9]
        self.resize = transforms.Compose([transforms.Resize(self.image_size)])
        
        print(f"A good FOV maybe {2*np.arctan(aperture_radius_m/(self.out_distance_m-self.second_metasurface_depth_m))/np.pi*180}")
        # building wavefront centers at the second MS for all pixels
        px1 = int(second_metasurface_psf_aperture_r / self.in_dx_m[0])
        px2 = px1 + 1  # should be odd
        self.centers = []
        xx, yy = torch.meshgrid(torch.arange(self.image_size[1])-self.image_size[1]//2, 
                                torch.arange(self.image_size[0])-self.image_size[0]//2, indexing='xy')
        dr = 2*np.tan(self.fov/2)*self.depth / ((self.image_size[0]**2 + self.image_size[1]**2)**0.5)
        # dx**2 + dy**2 = dr**2 => 1 + (dy/dx)**2 = (dr/dx)**2 => 1+(h/w)**2 = (dr/dx)**2 => dx = sqrt(dr**2/(1+(h/w)**2))
        self.dx  = np.sqrt((dr**2)/(1+(self.image_size[0]/self.image_size[1])**2))
        self.dy  = np.sqrt((dr**2)/(1+(self.image_size[1]/self.image_size[0])**2))
        xx = xx*self.dx
        yy = yy*self.dy
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                x, y, z = xx[i,j],yy[i,j], self.depth
                z0 = self.out_distance_m - self.second_metasurface_depth_m
                x0 = -x * z0 / z 
                y0 = -y * z0 / z
                x0_px = int(x0 / self.in_dx_m[1]) + self.in_size[1]//2
                y0_px = int(y0 / self.in_dx_m[0]) + self.in_size[0]//2
                self.centers.append((i,j,x0,y0, x0_px, y0_px))
                if x0_px < 0 or y0_px < 0:
                    raise Exception(f"x0_px {x0_px} or yo_px {y0_px} is negative at {i},{j}")
                if x0_px >= self.in_size[1] or y0_px >= self.in_size[0]:
                    raise Exception(f"x0_px {x0_px} or yo_px {y0_px} is too large (>{self.in_size}) at {i},{j}")
        
        # build the wavefront. This is common for all positions at this time
        xx2, yy2 = torch.meshgrid(torch.arange(self.in_size[1])-self.in_size[1]//2, torch.arange(self.in_size[0])-self.in_size[0]//2)
        xx2 = xx2[None]*self.in_dx_m[1]
        yy2 = yy2[None]*self.in_dx_m[0]
        phi = -2*np.pi/torch.tensor(self.wavelength_set_m)* torch.sqrt((xx2)**2 + (yy2)**2 + self.second_metasurface_depth_m**2)
        A = ((xx2)**2 + (yy2)**2 < second_metasurface_psf_aperture_r**2).float()
        
        print(A.shape, phi.shape, px1, px2)
        self.phi = crop_arr(phi, px1+px2, px1+px2)#[:, self.in_size[0]//2 - px1:self.in_size[0]//2 + px2, self.in_size[1]//2 - px1:self.in_size[1]//2 + px2]
        self.A = crop_arr(A, px1+px2, px1+px2)#[:, self.in_size[0]//2 - px1:self.in_size[0]//2 + px2, self.in_size[1]//2 - px1:self.in_size[1]//2 + px2]
        torch_zero = torch.tensor([0.0], dtype=torch.float32, device=x.device)
        self.incident_wavefront = torch.complex(self.A[None,None], torch_zero) * torch.exp(torch.complex(torch_zero, self.phi[None,None]))

        # define 2nd metasurface propagator
        self.PSF = PointSpreadFunctionNew(
                    in_size=[px1+px2,px1+px2],
                    in_dx_m=self.in_dx_m,
                    out_distance_m=self.second_metasurface_depth_m,
                    out_size=[px1+px2,px1+px2],
                    out_dx_m=self.out_dx_m,
                    out_resample_dx_m=None,
                    manual_upsample_factor=1,
                    radial_symmetry=radial_symmetry,
                    diffraction_engine=diffraction_engine
        ).cuda()

        # define 2nd metasurface phase profile
        # define the lens
        amp_exp, phase_exp, aperture = focusing_lens(
            in_size=self.in_size,
            in_dx_m=self.in_dx_m,
            wavelength_set_m=self.wavelength_set_m,
            depth_set_m=[1e6],
            fshift_set_m=[[0,0]],
            out_distance_m=self.second_metasurface_depth_m,
            aperture_radius_m=aperture_radius_m,
            radial_symmetry=radial_symmetry
        )
        model_name = 'Nanocylinders_TiO2_U300H600'
        p_norm, p, err = reverse_lookup_search_Nanocylinders_TiO2_U300nm_H600nm(amp_exp[None,None], phase_exp[None,None], self.wavelength_set_m[0:1])
        p = torch.from_numpy(p).cuda()
        self.p = torch.nn.Parameter(p, requires_grad=True)

        self.optical_model = load_optical_model(model_name).cuda()

        # define network
        self.net = define_network(opt.network)
        self.models.append(self.net)
        self.criterion = Loss(opt['train'].loss).to(self.accelerator.device)   


    def plot_psfs(self):
        fig, axes = plt.subplots(1, 2, figsize=(10,5))
        iw = self.incident_wavefront.cpu().numpy()[0,0]
        axes[0].imshow(np.abs(iw).transpose([1,2,0]))
        axes[1].imshow(np.angle(iw).transpose([1,2,0]))
        img = fig_to_array(fig)
        plt.close(fig)
        log_image(self.opt, self.accelerator, img[None], 'Incident Wavefront', self.global_step)

        psfs = self.sample['psfs'].detach().cpu().numpy()
        psfs = psfs[np.linspace(0, len(psfs)-1, 50, dtype=int)]
        fig = display_images({'PSFs': einops.rearrange(psfs, 'b c h w -> b h w c')}, 
                                independent_colorbar=False, cols_per_plot=10, size=2)
        img = fig_to_array(fig)
        plt.close(fig)
        log_image(self.opt, self.accelerator, img[None], 'PSFs', self.global_step)

        with torch.no_grad():
            est_amp, est_phase = self.optical_model(self.p, self.wavelength_set_m, pre_normalized=False)
        fig, axes = plt.subplots(1, 2, figsize=(10,5))
        axes[0].imshow(est_amp.cpu().numpy()[0,0].transpose([1,2,0]))
        axes[1].imshow(est_phase.cpu().numpy()[0,0].transpose([1,2,0]))
        img = fig_to_array(fig)
        plt.close(fig)
        log_image(self.opt, self.accelerator, img[None], '2nd MS phase profile', self.global_step)


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
        params_to_optimize = [
            {'params': optim_params},
            {'params': self.p, 'lr': opt.learning_rate * 0.00001},
        ]
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
        image = data['image']
        depth = data['depth']
        b, c, h, w = image.shape
        assert h == self.image_size[0] and w == self.image_size[1], f"Image size mismatch: {h}x{w} vs {self.image_size[0]}x{self.image_size[1]}"
        if self.incident_wavefront.device != image.device:
            print("Moving incident wavefront to Cuda")
            self.incident_wavefront = self.incident_wavefront.to(image.device)
            self.A = self.A.to(image.device)
        meas = torch.zeros((b, len(self.wavelength_set_m), self.out_size[0], self.out_size[1]), device=image.device)
        est_amp, est_phase = self.optical_model(self.p, self.wavelength_set_m, pre_normalized=False)
        px1 = self.phi.shape[-1]//2
        px2 = px1 + 1
        psfs = []
        pos = []
        for n in range(len(self.centers)):
            # i, j, x0, y0, x0_px, y0_px = self.centers[n]
            _i, _j = (torch.rand(1).item()*2-1)*self.image_size[1]/2, (torch.rand(1).item()*2-1)*self.image_size[0]/2
            x, y, z = _j*self.dx, _i*self.dy, self.depth
            i, j= int(_i)+self.image_size[1]//2, int(_j)+self.image_size[0]//2
            z0 = self.out_distance_m - self.second_metasurface_depth_m
            x0 = -x * z0 / z 
            y0 = -y * z0 / z
            x0_px = int(x0 / self.in_dx_m[1]) + self.in_size[1]//2
            y0_px = int(y0 / self.in_dx_m[0]) + self.in_size[0]//2
                
            is_close = False # if we use overlapping values, we cannot backprop
            for a in range(len(pos)):
                    if abs(pos[a][0] - x0_px) <= px1+px2 or abs(pos[a][1] - y0_px) <= px1+px2:
                        is_close = True
                        break
            if is_close:
                continue
            pos.append([x0_px, y0_px])
            # print(i,j,est_amp[..., y0_px-px1:y0_px+px2, x0_px-px1:x0_px+px2].shape, self.A.shape)
            intensity, phase = self.PSF(
                est_amp[..., y0_px-px1:y0_px+px2, x0_px-px1:x0_px+px2],
                est_phase[..., y0_px-px1:y0_px+px2, x0_px-px1:x0_px+px2],
                self.wavelength_set_m,
                incident_wavefront=self.incident_wavefront,
                aperture=self.A[None,None],
                normalize_to_aperture=True)
            psfs.append(intensity[:, 0, 0, :, :, :])
            # print(y0,x0, y0_px, x0_px, i, j)
            meas[..., y0_px-px1:y0_px+px2, x0_px-px1:x0_px+px2] += intensity[:, 0,0,:, :, :] * image[:,:,i:i+1, j:j+1]
            
        data['psfs'] = torch.cat(psfs, dim=0)
        data['meas'] = self.resize(meas)
        # data['meas'] /= data['meas'].max()
        self.sample = data


    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        net_in = torch.cat([self.sample['meas']], dim=1)
        preds = self.net(net_in)
        losses1 = self.criterion(preds[:,:1,:,:], self.sample['image'])
        losses2 = self.criterion(preds[:,1:,:,:], self.sample['depth'])
        
        # compute loss to maximize MTF
        freqs, mtf_1d, mtf = get_mtf(self.sample['psfs'][:, None, None], pixel_pitch=self.out_dx_m[0]) 
        loss_mtf = torch.mean(torch.nn.functional.sigmoid(-(mtf_1d-self.cutoff_mtf)) 
                           * (1/torch.sum(mtf_1d, dim=-1, keepdim=True))
                           * (1/freqs))
        
        loss = loss_mtf+losses1['all']+losses2['all']
        self.accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.01)
        
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        return {'loss_image': losses1['all'], 
                'loss_depth': losses2['all'], 
                'loss_mtf':loss_mtf, 
                'loss': loss}


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
    