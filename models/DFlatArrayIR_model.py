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

class DFlatArrayIR_model(BaseModel):

    def __init__(self, opt, logger):
        super(DFlatArrayIR_model, self).__init__(opt, logger)

        # ============ define DFlat PSF simulator and image generator
        # 1. initialize the target phase profile of the metasurface
        settings = opt.dflat
        self.wavelength_set_m = np.array(settings.wavelength_set_m)  # if we are using hsi/rgb images, this should change
        self.ps_locs = torch.tensor([list(map(float, v)) for v in settings.ps_locs.values()])  # point spread locations (we use a sparse grid)
        self.ps_loc_z = settings.ps_loc_z
        
        # find the MS center positions of the array
        self.array_size = settings.array_size
        self.array_spacing = settings.array_spacing
        self.MS_size = [(settings.h+1)*settings.in_dx_m[0], (settings.w+1)*settings.in_dx_m[1]]
        self.MSArray_size = [self.array_size[0]*self.array_spacing[0], self.array_size[1]*self.array_spacing[1]]
        self.MS_pos = []
        for i in range(self.array_size[0]):
            for j in range(self.array_size[1]):
                self.MS_pos.append([i*self.array_spacing[0] - self.MSArray_size[0]/2,   # x
                                    j*self.array_spacing[1] - self.MSArray_size[1]/2,   # y
                                    ])                                                 
        self.MS_pos = torch.tensor(self.MS_pos)

        # find the out_size of the MS. Downscale to preserve memory
        downscale_factor = [self.array_size[0], self.array_size[1]]
        self.out_size = [settings.h//downscale_factor[0], settings.w//downscale_factor[1]]

        if settings.initialization.type == 'focusing_lens':
            lens_settings = {
                "in_size": [settings.h+1, settings.w+1],
                "in_dx_m": settings.in_dx_m,
                "wavelength_set_m": self.wavelength_set_m,
                "depth_set_m": settings.initialization.depth_set_m,
                "fshift_set_m": settings.initialization.fshift_set_m,
                "out_distance_m": settings.out_distance_m,
                "aperture_radius_m": None,
                "radial_symmetry": False  # if True slice values along one radius
                }
            self.amp, self.phase, self.aperture = focusing_lens(**lens_settings) # [Lam, H, W]
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

        # need to move to GPU before parameter creation, or else move after optimizer creation
        def get_noise():
            return np.random.normal(self.p.mean(), self.p.std(), np.prod(self.p.shape)).reshape(self.p.shape)
        self.p = [torch.nn.Parameter(torch.from_numpy(self.p + get_noise()).cuda(), requires_grad=True) for _ in range(len(self.MS_pos))]
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
            radial_symmetry=False,
            diffraction_engine="ASM").cuda()
        
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
        network_opt['array_size'] = self.array_size
        network_opt['downscale_factor'] = downscale_factor
        self.net_g = define_network(network_opt)
        self.models.append(self.net_g)

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


    def feed_data(self, data, is_train=True):
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        gt = data[gt_key]

        gt = torch.nn.functional.interpolate(gt, size=self.out_size, mode='bilinear', align_corners=False)
        
        all_meas =[]
        self.all_psf_intensity = []
        for i in range(len(self.p)):
            ps_locs = torch.cat([torch.cat([self.ps_locs - self.MS_pos[i], torch.ones(len(self.ps_locs), 1)*self.ps_loc_z], dim=-1)], dim=0)
            
            assert not torch.isnan(self.p[i]).sum(), f'{torch.isnan(self.p[i]).sum()} Nan in p[{i}]'
            est_amp, est_phase = self.optical_model(self.p[i], self.wavelength_set_m, pre_normalized=False)
            psf_intensity, _ = self.PSF(
                est_amp.to(dtype=torch.float32, device=gt.device),
                est_phase.to(dtype=torch.float32, device=gt.device),
                self.wavelength_set_m,
                ps_locs,
                aperture=None,
                normalize_to_aperture=True)
        
            # shift psfs 
            n_pos = len(self.ps_idx)
            for i in range(n_pos):
                psf_intensity[:,:,i] = torch.roll(
                    psf_intensity[:,:,i], 
                    shifts=[self.ps_idx[i, 1], self.ps_idx[i, 0]], # 0th axis along horizontal (W), 1st axis along vertical (H)
                    dims=(-2,-1)
                )
            self.all_psf_intensity.append(psf_intensity)

            # Need to pass inputs like
            # psf has shape  [B P Z L H W]
            # scene radiance [B P Z L H W]
            # out shape      [B P Z L H W]
            meas = self.renderer(psf_intensity, einops.rearrange(gt, 'b c h w -> b 1 1 c h w'), rfft=True, crop_to_psf_dim=False)
            meas = self.renderer.rgb_measurement(meas, self.wavelength_set_m, gamma=True, process='demosaic')
            all_meas.append(meas[:,0,0])
        all_meas = torch.stack(all_meas)
        all_meas = einops.rearrange(all_meas, "n b c h w -> b n c h w")
        data[lq_key] = all_meas
        self.sample = data
        if self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.train if is_train else self.opt.val)
            
    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        pred = self.net_g(self.sample[lq_key])
        losses = self.criterion(pred, self.sample[gt_key])
        
        self.accelerator.backward(losses['all'])
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        return losses

    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        if idx==0:
            for i in range(len(self.all_psf_intensity)):
                psf_intensity = self.all_psf_intensity[i]
                for j in range(psf_intensity.shape[2]):
                    image1 = [psf_intensity[0,0,j].detach().cpu().numpy()]
                    image1 = np.stack(image1)
                    image1 = image1/image1.max()
                    image1 = np.clip(image1, 0, 1)
                    ps_loc = self.ps_locs[j]
                    log_image(self.opt, self.accelerator, image1, f"MS{i}_psf_{j}_{ps_loc}_{ps_loc + self.MS_pos[i]}", self.global_step)
                

        if self.opt.val.patched:
            b,n,c,h,w = self.original_size[lq_key]
            pred = []
            for _ in self.setup_patches():
                pred.append(self.net_g(self.sample[lq_key]))
            pred = torch.cat(pred, dim=0)
            pred = einops.rearrange(pred, '(b n) c h w -> b n c h w', b=b)
            out_allfocus = []
            for i in range(len(pred)):
                merged_allfocus = merge_patches(pred[i], self.sample[lq_key+'_patched_pos'])
                out_allfocus.append(merged_allfocus[..., :h, :w])
            lq = self.sample[lq_key+'_original']
            gt = self.sample[gt_key+'_original']
            out_allfocus = torch.stack(out_allfocus)
        else: 
            lq = self.sample[lq_key]
            gt = self.sample[gt_key]
            out_allfocus = self.net_g(lq)

        lq = lq.cpu().numpy()
        gt = gt.cpu().numpy()
        out_allfocus = out_allfocus.cpu().numpy()
        for i in range(len(gt)):
            idx += 1
            lq_i = einops.rearrange(lq[i], '(n1 n2) c h w -> c (n1 h) (n2 w)', n1=self.array_size[0], n2=self.array_size[1])
            image1 = [lq_i, gt[i], out_allfocus[i]]
            for j in range(len(image1)):
                if image1[j].shape[0] == 1:
                    image1[j] = einops.repeat(image1[j], '1 h w -> 3 h w')
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out_allfocus[i]]), 0,1), f'out_{idx}', self.global_step)
            log_metrics(gt[i], out_allfocus[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx
    
    
        