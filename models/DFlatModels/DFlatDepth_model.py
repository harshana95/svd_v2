import einops
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms

from dataset import create_dataset
from dataset.ZipDatasetWrapper import ZipDatasetWrapper
from models.archs import define_network
from models.base_model import BaseModel
from utils.dataset_utils import grayscale, merge_patches
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

class DFlatDepth_model(BaseModel):

    def __init__(self, opt, logger):
        super(DFlatDepth_model, self).__init__(opt, logger)

        # ============ define DFlat PSF simulator and image generator
        # 1. initialize the target phase profile of the metasurface
        init_settings = opt.dflat.initialization
        self.wavelength_set_m = np.array(opt.dflat.wavelength_set_m)  # if we are using hsi/rgb images, this should change
        self.ps_locs = torch.tensor([list(map(float, v)) for v in opt.dflat.ps_locs.values()])  # point spread locations (we use a sparse grid)
        self.depth_min = opt.dflat.depth_min
        self.depth_max = opt.dflat.depth_max

        if init_settings.type == 'focusing_lens':
            settings = {
                "in_size": [init_settings.h+1, init_settings.w+1],
                "in_dx_m": [10*300e-9, 10*300e-9],
                "wavelength_set_m": self.wavelength_set_m,
                "depth_set_m": opt.dflat.depth_set_m,
                "fshift_set_m": opt.dflat.fshift_set_m,
                "out_distance_m": opt.dflat.out_distance_m,
                "aperture_radius_m": None,
                "radial_symmetry": False  # if True slice values along one radius
                }
            self.amp, self.phase, self.aperture = focusing_lens(**settings) # [Lam, H, W]
            print("amp phase aperture", self.amp.shape, self.phase.shape, self.aperture.shape)
        else:
            raise NotImplementedError()
        # print(np.isnan(self.amp).sum(), np.isnan(self.phase).sum(), np.isnan(self.aperture).sum())

        # 2. Reverse look-up to find the metasurface that implements the target profile
        model_name = opt.dflat.model_name
        self.p_norm, self.p, err = reverse_lookup_optimize(
            self.amp[None, None],
            self.phase[None, None],
            self.wavelength_set_m,
            model_name,
            lr=1e-1,
            err_thresh=1e-6,
            max_iter=100,
            opt_phase_only=False)
        self.p = torch.nn.Parameter(torch.from_numpy(self.p).cuda(), requires_grad=True) # need to move to GPU before parameter creation, or else move after optimizer creation
        print(self.p.shape, self.p.device, self.p.is_leaf)
        # [B, H, W, D] where D = model.dim_in - 1 is the number of shape parameters

        # 3. load optical model
        self.optical_model = load_optical_model(model_name).cuda()
        # self.models.append(self.optical_model)

        # 4. setup PSF generators from phase, amp
        # Compute the point spread function given this broadband stack of field amplitude and phases
        self.PSF = PointSpreadFunction(
            in_size=[init_settings.h+1, init_settings.w+1],
            in_dx_m=[10*300e-9, 10*300e-9],
            out_distance_m=opt.dflat.out_distance_m,
            out_size=[init_settings.h, init_settings.w],
            out_dx_m=[5e-6,5e-6],
            out_resample_dx_m=None,
            radial_symmetry=False,
            diffraction_engine="ASM").cuda()
        
        self.ps_idx = self.ps_locs.clone()
        out_dx_m = self.PSF.propagator.out_dx_m
        self.ps_idx[:, 0] /= out_dx_m[0]
        self.ps_idx[:, 1] /= out_dx_m[1]
        self.ps_idx = self.ps_idx.to(int)[:, :2]
        print("PSF indices", self.ps_idx)

        self.coeffs = generate_coeffs(self.ps_idx, self.wavelength_set_m, [init_settings.h, init_settings.w]).cuda()
        self.coeffs_sum = torch.sum(self.coeffs, dim=-4,keepdim=True)

        # 5. renderer for image blurring
        self.renderer = Fronto_Planar_Renderer_Incoherent()
                
        # ========================== define generator network
        self.net_g = define_network(opt.network_g)
        self.net_d = define_network(opt.network_d)
        self.models.append(self.net_g)
        self.models.append(self.net_d)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
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
        
        # add shape parameters as a parameter to optimize
        params_to_optimize = [{'params': optim_params}, 
                              {'params': self.p, 'lr': opt.learning_rate * 0.00001}]
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


    def feed_data(self, data, is_train=True):
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        fg = data['foreground_1']
        bg = data['background_2']
        mask = data['mask_1']
        mask = torch.clamp(mask, 1e-6, 1 - 1e-6)  # if we don't clamp, we get NaNs.   
        device = mask.device
        
        # assert mask.min() ==0 and mask.max() == 1, f"mask min max {mask.min(dim=(1,2,3))} {mask.max(dim=(1,2,3))}"
        # generate the ground truth images [all-in-focus image, depth map]
        bg_depth = torch.rand(1)*(self.depth_max - self.depth_min) + self.depth_min
        fg_depth = torch.rand(1)*(bg_depth - self.depth_min) + self.depth_min

        # simulate PSF for current p
        ps_locs = torch.cat([torch.cat([self.ps_locs, torch.ones(len(self.ps_locs), 1)*fg_depth], dim=-1),
                             torch.cat([self.ps_locs, torch.ones(len(self.ps_locs), 1)*bg_depth], dim=-1)], dim=0)
        
        assert not torch.isnan(self.p).sum(), f'{torch.isnan(self.p).sum()} Nan in p'
        est_amp, est_phase = self.optical_model(self.p, self.wavelength_set_m, pre_normalized=False)
        self.psf_intensity, _ = self.PSF(
            est_amp.to(dtype=torch.float32, device=mask.device),
            est_phase.to(dtype=torch.float32, device=mask.device),
            self.wavelength_set_m,
            ps_locs,
            aperture=None,
            normalize_to_aperture=True)
        # print(torch.isnan(self.psf_intensity).sum(), torch.isnan(est_amp).sum(), torch.isnan(est_phase).sum())

        # shift psfs 
        n_pos = len(self.ps_idx)
        for i in range(n_pos):
            self.psf_intensity[:,:,i] = torch.roll(
                self.psf_intensity[:,:,i], 
                shifts=[self.ps_idx[i, 1], self.ps_idx[i, 0]], # 0th axis along horizontal (W), 1st axis along vertical (H)
                dims=(-2,-1)
            )
            self.psf_intensity[:,:,n_pos + i] = torch.roll(
                self.psf_intensity[:,:,n_pos + i], 
                shifts=[self.ps_idx[i, 1], self.ps_idx[i, 0]], # 0th axis along horizontal (W), 1st axis along vertical (H)
                dims=(-2,-1)
            )
        
        # Need to pass inputs like
        # psf has shape  [B P Z L H W]
        # scene radiance [B P Z L H W]
        # out shape      [B P Z L H W]
        
        # fg = fg*mask + bg*(1-mask)*0.0
        # mask = torch.ones_like(fg)*mask

        # fg_meas and mask_meas gives NaNs after first backward pass. If they are not in the equation it works fine.
        # bg_meas is also fine. mask is also fine.
        # fg = fg*mask + bg*(1-mask) before blurring removes the error when using fg_meas
        # clamping mask avoid some Nans, but we get nans in the middle of training for some reason
        # when I added noise to both foreground and mask, the Nan error did not occur.
        mask_meas = self.renderer(self.psf_intensity[:,:,:n_pos], einops.rearrange(mask, 'b c h w -> b 1 1 c h w'), rfft=True, crop_to_psf_dim=False)
        mask_meas = self.renderer.rgb_measurement(mask_meas, self.wavelength_set_m, gamma=True, process='demosaic')
        fg_meas = self.renderer(self.psf_intensity[:,:,:n_pos], einops.rearrange(fg, 'b c h w -> b 1 1 c h w'), rfft=True, crop_to_psf_dim=False)
        fg_meas = self.renderer.rgb_measurement(fg_meas, self.wavelength_set_m, gamma=True, process='demosaic')

        bg_meas = self.renderer(self.psf_intensity[:,:,n_pos:], einops.rearrange(bg, 'b c h w -> b 1 1 c h w'), rfft=True, crop_to_psf_dim=False)
        bg_meas = self.renderer.rgb_measurement(bg_meas, self.wavelength_set_m, gamma=True, process='demosaic')
        assert torch.isnan(fg_meas).sum() == 0, f'{torch.isnan(fg_meas).sum()} Nan in fg_meas'
        assert torch.isnan(bg_meas).sum() == 0, f'{torch.isnan(bg_meas).sum()} Nan in bg_meas'
        assert torch.isnan(mask_meas).sum() == 0, f'{torch.isnan(mask_meas).sum()} Nan in mask_meas'

        fg_meas = (fg_meas * self.coeffs).sum(2, keepdim=True)/self.coeffs_sum
        bg_meas = (bg_meas * self.coeffs).sum(2, keepdim=True)/self.coeffs_sum
        mask_meas = (mask_meas * self.coeffs).sum(2, keepdim=True)/self.coeffs_sum
        fg_meas = fg_meas[:, 0, 0]  # no polarization
        bg_meas = bg_meas[:, 0, 0]
        mask_meas = mask_meas[:, 0, 0]

        data[gt_key] = bg*(1 - mask) + fg*mask  # alpha clipping and merging fg and bg  
        data[lq_key] = bg_meas*(1 - mask_meas) + fg_meas*mask_meas  #bg_meas*(1 - mask_meas) + fg_meas*mask_meas # alpha clipping and merging fg and bg
        data['depth'] = bg_depth.to(device)*(1-mask) + fg_depth.to(device) *mask  # depth map
        data['depth'] = (data['depth'] - self.depth_min)/(self.depth_max - self.depth_min)
        data['mask'] = mask
        data['mask_meas'] = mask_meas
        
        self.sample = data
        if self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key, 'depth'], opt=self.opt.train if is_train else self.opt.val)
            
    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        preds_allfocus = self.net_g(self.sample[lq_key])
        preds_depth = self.net_d(self.sample[lq_key])

        losses_g = self.criterion(preds_allfocus, self.sample[gt_key])
        losses_d = self.criterion(preds_depth, self.sample['depth'])
        losses = {'all': losses_g['all'] + losses_d['all'], 'g': losses_g, 'd': losses_d}
        # print(losses)
        self.accelerator.backward(losses['all'])
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), 0.01)
        
        assert not torch.isnan(self.p).sum(), f'{torch.isnan(self.p).sum()} Nan in p'
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        return losses

    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        if idx==0:
            for i in range(self.psf_intensity.shape[2]):
                image1 = [self.psf_intensity[0,0,i].detach().cpu().numpy()]
                image1 = np.stack(image1)
                image1 = image1/image1.max()
                image1 = np.clip(image1, 0, 1)
                ps_loc = self.ps_locs[i%len(self.ps_locs)]
                is_fg = i < len(self.ps_locs)
                log_image(self.opt, self.accelerator, image1, f"psf_{i}_{'fg' if is_fg else 'bg'}_{ps_loc}", self.global_step)

        if self.opt.val.patched:
            b,c,h,w = self.original_size[lq_key]
            pred = []
            depth = []
            for _ in self.setup_patches():
                pred.append(self.net_g(self.sample[lq_key]))
                depth.append(self.net_d(self.sample[lq_key]))
            pred = torch.cat(pred, dim=0)
            depth = torch.cat(depth, dim=0)
            pred = einops.rearrange(pred, '(b n) c h w -> b n c h w', b=b)
            depth = einops.rearrange(depth, '(b n) c h w -> b n c h w', b=b)
            out_allfocus = []
            out_depths = []
            for i in range(len(pred)):
                merged_allfocus = merge_patches(pred[i], self.sample[lq_key+'_patched_pos'])
                merged_depth = merge_patches(depth[i], self.sample[lq_key+'_patched_pos'])
                out_allfocus.append(merged_allfocus[..., :h, :w])
                out_depths.append(merged_depth[..., :h, :w])
            lq = self.sample[lq_key+'_original']
            gt = self.sample[gt_key+'_original']
            gt_depth = self.sample['depth_original']
            out_allfocus = torch.stack(out_allfocus)
            out_depths = torch.stack(out_depths)
        else: 
            lq = self.sample[lq_key]
            gt = self.sample[gt_key]
            gt_depth = self.sample['depth']
            out_allfocus = self.net_g(lq)
            out_depths = self.net_d(lq)

        lq = lq.cpu().numpy()
        gt = gt.cpu().numpy()
        gt_depth = gt_depth.cpu().numpy()
        out_allfocus = out_allfocus.cpu().numpy()
        out_depths = out_depths.cpu().numpy()
        mask = self.sample['mask'].cpu().numpy()
        mask_meas = self.sample['mask_meas'].cpu().numpy()
        for i in range(len(gt)):
            idx += 1
            image1 = [lq[i], gt[i], out_allfocus[i], out_depths[i], gt_depth[i]]
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, np.clip(np.stack([mask[i]]), 0,1), f'mask_{idx:04d}', self.global_step)
            log_image(self.opt, self.accelerator, np.clip(np.stack([mask_meas[i]]), 0,1), f'maskmeas_{idx:04d}', self.global_step)
            
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out_allfocus[i]]), 0,1), f'out_{idx:04d}', self.global_step)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out_depths[i]]), 0,1), f'depth_{idx:04d}', self.global_step)
            log_metrics(gt[i], out_allfocus[i], self.opt.val.metrics, self.accelerator, self.global_step)
            log_metrics(gt_depth[i], out_depths[i], self.opt.val.metrics, self.accelerator, self.global_step, comment='depth_')
        return idx
    
    
        