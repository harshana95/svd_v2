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
from utils.dataset_utils import crop_arr, grayscale, merge_patches, sv_convolution
from utils.image_utils import save_images_as_zip
from utils.loss import Loss
from utils import log_image, log_metrics


from dflat.initialize import focusing_lens
from dflat.metasurface import reverse_lookup_optimize, load_optical_model
from dflat.propagation.propagators_legacy import PointSpreadFunction # Using the legacy version as it takes wavelength as a forward input instead of initialization input
from dflat.render import hsi_to_rgb
from dflat.plot_utilities import format_plot
from dflat.render import Fronto_Planar_Renderer_Incoherent

def generate_coeffs(ps_idx, sim_wl, img_size, r=0.):
    coeff = torch.zeros((1, 1, len(ps_idx), len(sim_wl), *img_size))
    xx, yy = torch.meshgrid(
            torch.arange(0, img_size[0], dtype=torch.int16),
            torch.arange(0, img_size[1], dtype=torch.int16),
            indexing="xy",
        )
    xx = xx - (xx.shape[-1] - 1) / 2
    yy = yy - (yy.shape[-2] - 1) / 2
    for i in range(len(ps_idx)):
        x, y = ps_idx[i]
        d = 1/(1 + torch.sqrt((xx - x)**2 + (yy -y)**2))
        for w in range(len(sim_wl)):
            coeff[:, :, i, w] = d
    return coeff

class DFlatArrayDepth_model(BaseModel):
    def __init__(self, opt, logger):
        super(DFlatArrayDepth_model, self).__init__(opt, logger)
        INIT_NOISE_FACTOR = 0.01

        # ============ define DFlat PSF simulator and image generator
        settings = opt.dflat
        # self.pixel_dx_m = settings.pixel_dx_m
        self.radially_symmetric = settings.get("radially_symmetric", False)
        self.single_wavelength = settings.get("single_wavelength", False)

        # 1. initialize the target phase profile of the metasurface
        self.wavelength_set_m = np.array(settings.wavelength_set_m)  # if we are using hsi/rgb images, this should change
        self.ps_locs = torch.tensor([list(map(float, v)) for v in settings.ps_locs])  # point spread locations (we use a sparse grid)
        assert self.ps_locs.shape[0] > 0, f"Need at least one Point Spread location. Given {settings.ps_locs}"
        logger.info(f"Point spread locations (in meters): {self.ps_locs}")
        self.depth_min = settings.depth_min
        self.depth_max = settings.depth_max
        
        self.depth_levels = settings.depth_levels

        # find the MS center positions of the array
        self.array_size = settings.array_size  # col, row
        self.array_spacing = settings.array_spacing # col, row
        self.MS_size = [(settings.w+1)*settings.in_dx_m[0], (settings.h+1)*settings.in_dx_m[1]]  # width, height
        self.MSArray_size = [(self.array_size[0]-1)*self.array_spacing[0], (self.array_size[1]-1)*self.array_spacing[1]] # width, height
        logger.info(f"MS size (width, height): {self.MS_size} MS Array size (width, height): {self.MSArray_size}")
        self.MS_pos = []
        for i in range(self.array_size[0]): # cols
            for j in range(self.array_size[1]): # rows
                self.MS_pos.append([i*self.array_spacing[0] - self.MSArray_size[0]/2,   # x
                                    j*self.array_spacing[1] - self.MSArray_size[1]/2,   # y
                                    0])                                                 # z                                                               # z
                logger.info(f"MS {i},{j} center: {self.MS_pos[-1]}")
        self.MS_pos = torch.tensor(self.MS_pos)

        # find the out_size of the MS. Downscale to preserve memory
        downscale_factor = settings.downscale_factor
        self.out_size = [settings.h//downscale_factor[0], settings.w//downscale_factor[1]]

        if settings.initialization.type == 'focusing_lens':
            lenssettings = {
                "in_size": [settings.h+1, settings.w+1],
                "in_dx_m": settings.in_dx_m,
                "wavelength_set_m": self.wavelength_set_m,
                "depth_set_m": settings.initialization.depth_set_m,
                "fshift_set_m": settings.initialization.fshift_set_m,
                "out_distance_m": settings.out_distance_m,
                "aperture_radius_m": None,
                "radial_symmetry": self.radially_symmetric  # if True slice values along one radius
                }
            self.amp, self.phase, self.aperture = focusing_lens(**lenssettings) # [Lam, H, W]
            print("amp phase aperture", self.amp.shape, self.phase.shape, self.aperture.shape)
        else:
            raise NotImplementedError()

        # 2. Reverse look-up to find the metasurface that implements the target profile
        model_name = settings.model_name
        self.p_norm, self.p, err = reverse_lookup_optimize(
            self.amp[None, None],
            self.phase[None, None],
            self.wavelength_set_m,
            model_name,
            lr=1e-1,
            err_thresh=1e-6,
            max_iter=100,
            opt_phase_only=False)
        print("p shape", self.p.shape)
        # need to move to GPU before parameter creation, or else move after optimizer creation
        def get_noise():
            return np.random.normal(self.p_norm.mean(), self.p_norm.std(), np.prod(self.p_norm.shape)).reshape(self.p_norm.shape)
        
        # add noise to the p values 
        self.p_norm = [torch.nn.Parameter(torch.from_numpy(self.p_norm*(1-INIT_NOISE_FACTOR) + get_noise()*INIT_NOISE_FACTOR).cuda(), requires_grad=opt.train.optimize_shape_param) for _ in range(len(self.MS_pos))]
        # [B, H, W, D] where D = model.dim_in - 1 is the number of shape parameters

        # 3. load optical model
        self.optical_model = load_optical_model(model_name).cuda()

        # 4. setup PSF generators from phase, amp
        # Compute the point spread function given this broadband stack of field amplitude and phases
        self.PSF = PointSpreadFunction(
            in_size=[settings.h+1, settings.w+1],
            in_dx_m=settings.in_dx_m,
            out_distance_m=settings.out_distance_m,
            out_size=self.out_size,
            out_dx_m=settings.out_dx_m,
            out_resample_dx_m=None,
            radial_symmetry=self.radially_symmetric,
            diffraction_engine="ASM"
        ).cuda()
        
        # TODO: Using ps_idx for rolling PSFs might be inaccurate in some cases
        # ps idx for ps shift in pixels. if the ps are off axis, the psf might not have the same shift. TODO find a way to solve this.
        self.ps_idx = self.ps_locs.clone()
        out_dx_m = self.PSF.propagator.out_dx_m
        self.ps_idx[:, 0] /= out_dx_m[0]
        self.ps_idx[:, 1] /= out_dx_m[1]
        self.ps_idx = self.ps_idx.to(int)[:, :2]
        logger.info(f"PSF indices {self.ps_idx}")

        self.coeffs = generate_coeffs(self.ps_idx, self.wavelength_set_m, [settings.h, settings.w]).cuda()
        self.coeffs_sum = torch.sum(self.coeffs, dim=-4,keepdim=True)

        # 5. renderer for image blurring
        self.renderer = Fronto_Planar_Renderer_Incoherent()
                
        # ========================== define generator network
        network_opt = opt.network
        if network_opt is not None:
            network_opt['img_channel'] = network_opt['img_channel'] if not self.single_wavelength else 1
            network_opt['array_size'] = self.array_size
            network_opt['downscale_factor'] = downscale_factor
            self.net_g = define_network(network_opt)
            self.models.append(self.net_g)

            if self.is_train:
                self.net_g.train()

        # setup loss function
        train_opt = self.opt['train']
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)

    def save_other_parameters(self, path):
        p_cpu = [pp.detach().cpu() for pp in self.p_norm]
        torch.save(p_cpu, os.path.join(path, 'shape_params.pt'))


        # # Write this lens to a gds file for fabrication
        # from dflat.GDSII.assemble import assemble_cylinder_gds
        # cell_size = self.opt.dflat.in_dx_m
        # block_size = self.opt.dflat.in_dx_m # This can be used to repeat the cell as a larger block

        # self.array_size
        # for i, pp in enumerate(self.p_norm):
        #     col, row = i%self.array_size[0], i//self.array_size[0]
        #     p_norm = pp.detach().cpu().numpy()
        #     p_denorm = self.optical_model.denormalize(p_norm)
        #     pfab = p_denorm.squeeze(0)
        #     mask = np.ones(pfab.shape[0:2])
        #     assemble_cylinder_gds(pfab, mask, cell_size, block_size, savepath=os.path.join(path, f"MO_r{row}_c{col}.gds"))
    
    def load_other_parameters(self, path):
        p_cpu = torch.load(os.path.join(path, 'shape_params.pt'))
        for i in range(len(self.p_norm)):
            self.p_norm[i].data = p_cpu[i].data.to(self.p_norm[i].device)

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
                              {'params': self.p_norm, 'lr': opt.learning_rate}]
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
        fg = data['foreground_1'] # batch size * (depth_levels -1)
        bg = data['background_2'] # batch size * (depth_levels -1)
        mask = data['mask_1']  # batch size * (depth_levels -1)
        # print(bg.shape, fg.shape, mask.shape)

        bg = bg[:len(bg)//(self.depth_levels - 1)]  # pick first batch_size images. it will load batch_size*(depth_levels -1) images
        bg = einops.rearrange(bg, 'b c h w -> b 1 1 c h w')
        fg = einops.rearrange(fg, '(b n) c h w -> b n 1 1 c h w', n=self.depth_levels - 1)
        mask = einops.rearrange(mask, '(b n) c h w -> b n 1 1 c h w', n=self.depth_levels - 1)
        
        
        if is_train:
            mask = torch.clamp(mask, 1e-6, 1 - 1e-6)  # if we don't clamp, we get NaNs.   
        device = mask.device
        n_channels = fg.shape[-3]
        n_pos = len(self.ps_idx)
        
        # generate the ground truth images [all-in-focus image, depth map]
        bg_depth = torch.rand(1)*(self.depth_max - self.depth_min) + self.depth_min
        fg_depth = torch.rand(self.depth_levels - 1)*(bg_depth - self.depth_min) + self.depth_min
        fg_depth = torch.sort(fg_depth, descending=True)[0] # from far to near for easy alpha blending
        bg_ps_locs = torch.cat([self.ps_locs, torch.ones(len(self.ps_locs), 1)*bg_depth], dim=-1)
        fg_ps_locs = [torch.cat([self.ps_locs, torch.ones(len(self.ps_locs), 1)*fg_depth[_i]], dim=-1) for _i in range(self.depth_levels - 1)]
        _ps_locs = torch.cat(fg_ps_locs + [bg_ps_locs], dim=0) # [P, 3] P = n_pos * depth_levels (fg1, fg2, ..., bg)
        
        self.all_psf_intensity = []
        all_meas = []
        # simulate PSF for current p
        for i in range(len(self.p_norm)):  # cols x rows
            ps_locs = _ps_locs - self.MS_pos[i] # MS at center, translate obj
            # find chief ray
            #simulate the images in single sensor?

            assert not torch.isnan(self.p_norm[i]).sum(), f'{torch.isnan(self.p_norm[i]).sum()} Nan in p_norm'
            if self.single_wavelength:
                wv_idx = (i//self.array_size[0])%len(self.wavelength_set_m)
                wv = [self.wavelength_set_m[wv_idx]]
                _bg = bg[..., wv_idx:wv_idx+1,:,:]
                _fg = fg[..., wv_idx:wv_idx+1,:,:]
                _mask = mask[..., wv_idx:wv_idx+1,:,:]
            else:
                wv = self.wavelength_set_m
                _bg = bg
                _fg = fg
                _mask = mask
            est_amp, est_phase = self.optical_model(self.p_norm[i], wv, pre_normalized=True)
            psf_intensity, _ = self.PSF(
                est_amp.to(dtype=torch.float32, device=mask.device),
                est_phase.to(dtype=torch.float32, device=mask.device),
                wv,
                ps_locs,
                aperture=None,
                normalize_to_aperture=True)

            # # shift psfs # TODO: This shifting is incorrect!!!
            # TODO: Using ps_idx for rolling PSFs might be inaccurate in some cases
            # ps idx for ps shift in pixels. if the ps are off axis, the psf might not have the same shift. TODO find a way to solve this.
            # self.ps_idx = ps_locs.clone()
            # out_dx_m = self.PSF.propagator.out_dx_m
            # self.ps_idx[:, 0] /= out_dx_m[0]
            # self.ps_idx[:, 1] /= out_dx_m[1]
            # self.ps_idx = self.ps_idx.to(int)[:, :2]
            # for i in range(n_pos):
            #     self.psf_intensity[:,:,i] = torch.roll(
            #         self.psf_intensity[:,:,i], 
            #         shifts=[self.ps_idx[i, 1], self.ps_idx[i, 0]], # 0th axis along horizontal (W), 1st axis along vertical (H)
            #         dims=(-2,-1)
            #     )
            #     self.psf_intensity[:,:,n_pos + i] = torch.roll(
            #         self.psf_intensity[:,:,n_pos + i], 
            #         shifts=[self.ps_idx[i, 1], self.ps_idx[i, 0]], # 0th axis along horizontal (W), 1st axis along vertical (H)
            #         dims=(-2,-1)
            #     )
            self.all_psf_intensity.append(psf_intensity)

            # Need to pass inputs like
            # psf has shape  [B P Z L H W]
            # scene radiance [B P Z L H W]
            # out shape      [B P Z L H W]
            crop_size = 256
            h1 = psf_intensity.shape[-2]//2 - crop_size//2
            h2 = h1 + crop_size
            

            # simulating bg
            bg_meas = self.renderer(psf_intensity[:,:,-n_pos:, :, h1:h2, h1:h2], _bg, rfft=True, crop_to_psf_dim=False)
            # bg_meas = self.renderer.rgb_measurement(bg_meas, self.wavelength_set_m, gamma=True, process='demosaic')
            
            # from torch.fft import fft2, ifft2, fftshift, ifftshift
            # X = fft2(bg, dim=(-2, -1))
            # H = fft2(psf_intensity[:, :, -n_pos:], dim=(-2,-1))
            # out = ifftshift(ifft2(X * H, dim=(-2, -1)), dim=(-2, -1))
            # X_log = np.log(1 + np.abs(X.real[0,0,0].cpu().numpy().transpose([1,2,0])))
            # plt.imsave('tmp.png', (X_log/X_log.max()).clip(0,1))
            
            gray = grayscale()
            if n_channels == 1:    
                bg_meas = gray(bg_meas)

            meas = bg_meas
            # simulating fg 
            for d in range(self.depth_levels -1):  # loop through fg images
                # TODO: add random translation to the fg and mask
                i_start = n_pos*d
                i_end = n_pos*(d+1)
                mask_meas = self.renderer(psf_intensity[:,:,i_start:i_end, :, h1:h2, h1:h2], _mask[:, d], rfft=True, crop_to_psf_dim=False)
                # mask_meas = self.renderer.rgb_measurement(mask_meas, self.wavelength_set_m, gamma=True, process='demosaic')
                fg_meas = self.renderer(psf_intensity[:,:,i_start:i_end, :, h1:h2, h1:h2], _fg[:, d], rfft=True, crop_to_psf_dim=False)
                # fg_meas = self.renderer.rgb_measurement(fg_meas, self.wavelength_set_m, gamma=True, process='demosaic')
                mask_meas = gray(mask_meas)
                if n_channels == 1:
                    fg_meas = gray(fg_meas)

                # alpha clipping and merging with previous measurement
                mask_meas = mask_meas/mask_meas.max()
                meas = meas*(1 - mask_meas) + fg_meas*mask_meas

            # assert torch.isnan(fg_meas).sum() == 0, f'{torch.isnan(fg_meas).sum()} Nan in fg_meas'
            # assert torch.isnan(bg_meas).sum() == 0, f'{torch.isnan(bg_meas).sum()} Nan in bg_meas'
            # assert torch.isnan(mask_meas).sum() == 0, f'{torch.isnan(mask_meas).sum()} Nan in mask_meas'

            # fg_meas = (fg_meas * self.coeffs).sum(2, keepdim=True)/self.coeffs_sum
            # bg_meas = (bg_meas * self.coeffs).sum(2, keepdim=True)/self.coeffs_sum
            # mask_meas = (mask_meas * self.coeffs).sum(2, keepdim=True)/self.coeffs_sum

            all_meas.append(meas[:, 0, 0])  # no polarization

        # this is not correct for all MS in the array. there can be parallax effect for GT
        gt = bg
        for d in range(self.depth_levels -1):
            gt = gt*(1 - mask[:, d]) + fg[:, d]*mask[:, d]
        gt = gt[:, 0, 0]
        # calculate depth
        depth = bg_depth.to(device)
        for d in range(self.depth_levels - 1):
            depth = depth*(1-mask[:, d]) + fg_depth[d].to(device)*mask[:, d]
        depth = depth[:, 0, 0]
        if self.single_wavelength:
            depth = gray(depth)
        all_meas = torch.stack(all_meas)
        all_meas = einops.rearrange(all_meas, "n b c h w -> b n c h w")
        data[gt_key] = gt
        data[lq_key] = all_meas
        data['depth'] = depth
        data['depth'] = (data['depth'] - self.depth_min)/(self.depth_max - self.depth_min) # normalize depth to 0-1
        data['mask'] = mask
        
        self.sample = data
        if self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key, 'depth'], opt=self.opt.train if is_train else self.opt.val)
            
    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        preds_depth = self.net_g(self.sample[lq_key])

        losses = self.criterion(preds_depth, self.sample['depth'])
        self.accelerator.backward(losses['all'])
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        return losses

    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        psfs = {}
        for i in range(len(self.all_psf_intensity)):
            psf_intensity = self.all_psf_intensity[i]
            for j in range(psf_intensity.shape[2]):
                psf = psf_intensity[0,0,j].detach().cpu().numpy()
                # psf = crop_arr(psf, 64,64)
                if j in psfs.keys():
                    psfs[j].append(psf)
                else:
                    psfs[j] = [psf]
        for k in psfs.keys():
            depth_obj = k//len(self.ps_locs)
            image1 = einops.rearrange(np.stack(psfs[k]), '(n2 n1) c h w -> c (n2 h) (n1 w)', n1=self.array_size[0], n2=self.array_size[1])
            image1 = np.stack([image1])
            image1 = image1/image1.max()
            image1 = np.clip(image1, 0, 1)
            ps_loc = self.ps_locs[k%len(self.ps_locs)]
            log_image(self.opt, self.accelerator, image1, f"img{idx}_psfs_{k}_depth_{depth_obj}_{ps_loc}", self.global_step)
        
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
                if image1[j].shape[0] == 1:
                    image1[j] = einops.repeat(image1[j], '1 h w -> 3 h w')
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx:04d}', self.global_step)
            log_metrics(gt_depth[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx

