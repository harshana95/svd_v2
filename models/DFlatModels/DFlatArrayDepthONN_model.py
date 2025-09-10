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
from models.DFlatModels.DFlatArrayDepth_model import DFlatArrayDepth_model
from models.archs import define_network
from models.archs.related.AdaBinsMonoDepth.unet_adaptive_bins import UnetAdaptiveBins
from utils.dataset_utils import crop_arr, grayscale, merge_patches
from utils.image_utils import save_images_as_zip
from utils.loss import Loss
from utils import log_image, log_metrics


from dflat.initialize import focusing_lens
from dflat.metasurface import reverse_lookup_optimize, load_optical_model
from dflat.propagation.propagators_legacy import PointSpreadFunction # Using the legacy version as it takes wavelength as a forward input instead of initialization input
from dflat.render import hsi_to_rgb
from dflat.plot_utilities import format_plot
from dflat.render import Fronto_Planar_Renderer_Incoherent

class DFlatArrayDepthONN_model(DFlatArrayDepth_model):
    def __init__(self, opt, logger):
        super(DFlatArrayDepthONN_model, self).__init__(opt, logger)  # depth array  initialization logic is here

        depth_network_opt = opt.get('depth_network', None)
        if depth_network_opt is None:
            raise ValueError("Depth network not provided in the configs")

        if depth_network_opt.type == "AdaBins":  # This looks too big
            MIN_DEPTH = 1e-3
            MAX_DEPTH_NYU = 10
            N_BINS = 256 

            # NYU
            model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_NYU)
            pretrained_path = "./models/archs/related/AdaBinsMonoDepth/AdaBins_nyu.pt"
            model_state_dict = torch.load(pretrained_path, map_location='cpu')['model']
            for k in model_state_dict.keys():
                k0 = k[7:]
                try:
                    model.get_parameter(k0).data.copy_(model_state_dict[k])
                except:
                    pass
            model.eval()
            model.requires_grad_(False)
            self.depth_model = model
            
        elif depth_network_opt.type == "LiteMono":
            # https://github.com/noahzn/Lite-Mono?tab=readme-ov-file#kitti
            pass
        elif depth_network_opt.type == "file":
            kernels = torch.load(depth_network_opt.path)
            self.kernels = []
            for key in kernels.keys():
                self.kernels.append(kernels[key].to(self.accelerator.device))
            self.kernels = torch.cat(self.kernels, dim=0) #[:4]
            self.kernels = F.interpolate(self.kernels, scale_factor=depth_network_opt.get('rescale', 1), mode='nearest') #, align_corners=False)

    def setup_dataloaders(self):
        # create train and validation dataloaders
        train_set = create_dataset(self.opt.datasets.train)
        self.dataloader = DataLoader(
            train_set,
            shuffle=self.opt.datasets.train.use_shuffle,
            batch_size=self.opt.train.batch_size,
            num_workers=self.opt.datasets.train.get('num_worker_per_gpu', 1),
        )
        val_set = create_dataset(self.opt.datasets.val)
        self.test_dataloader = DataLoader(
            val_set,
            shuffle=self.opt.datasets.val.use_shuffle,
            batch_size=self.opt.val.batch_size,
            num_workers=self.opt.datasets.val.get('num_worker_per_gpu', 1),
        )
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

    def feed_data(self, data, is_train=True):
        assert not self.opt.train.patched
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        self.sample = data
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

            # Need to pass inputs like
            # psf has shape  [B P Z L H W]
            # scene radiance [B P Z L H W]
            # out shape      [B P Z L H W]
            meas = self.renderer(psf_intensity, einops.rearrange(x, 'b c h w -> b 1 1 c h w'), rfft=True, crop_to_psf_dim=False)
            meas = self.renderer.rgb_measurement(meas, np.array([self.wavelength_set_m[i%len(self.wavelength_set_m)]]), gamma=True, process='demosaic')

            assert torch.isnan(meas).sum() == 0, f'{torch.isnan(meas).sum()} Nan in meas'
            meas = meas[:, 0, 0]  # no polarization
            all_meas.append(meas)
        
        all_meas = torch.stack(all_meas)
        all_meas = einops.rearrange(all_meas, "n b c h w -> b n c h w")
        data[lq_key] = all_meas
        data['depth'] = d  # depth map
        
        self.sample = data
        if self.opt.train.patched:
            self.grids(keys=[lq_key, 'depth'], opt=self.opt.train if is_train else self.opt.val)
        
    
    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key
        
        # we don't have a post pocessing network
        # preds_depth = self.net_g(self.sample[lq_key])
        
        all_psfs = self.all_psf_intensity

        total_loss = 0
        ms_idx = 0
        LOSS_SIZE = (self.kernels[0,0].shape[-2]*2, self.kernels[0,0].shape[-1]*2)  # self.out_size[0], self.out_size[1]
        for k in range(len(self.kernels)):
            for c in range(self.kernels.shape[1]):
                # print(all_psfs[ms_idx].shape, pos_kernels[k, c].shape) # torch.Size([1, 1, 1, 1, 512, 512]) torch.Size([7, 7])
                psf_pred = all_psfs[ms_idx][0,0,0,0]
                psf_gt = crop_arr(self.kernels[k, c], LOSS_SIZE[0], LOSS_SIZE[1])
                psf_pred = crop_arr(psf_pred, LOSS_SIZE[0], LOSS_SIZE[1])
                kernel_loss = self.criterion(psf_pred/(psf_pred.sum() + 1e-6), psf_gt/(psf_gt.sum() + 1e-6))
                total_loss += kernel_loss['all']*1  # loss was around 1e-8
                ms_idx += 1
        
        self.accelerator.backward(total_loss)
        # torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()

        return {"all": total_loss}

    def validate_step(self, batch, idx, lq_key, gt_key):
        if idx == 0:
            idx += 1 # remove if youre processing the images because you should do this when processing images
            self.feed_data(batch, is_train=False)
            psfs = {}
            kh, kw = self.kernels.shape[2], self.kernels.shape[3]
            NORMALIZE_PSFS = True
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

                tmp = einops.rearrange(self.kernels[:, :].detach().cpu().numpy(), 'n c h w -> (n c) h w')
                if NORMALIZE_PSFS:
                    tmp = tmp/tmp.sum(0, keepdims=True)
                image2 = einops.rearrange(tmp, '(n1 n2) h w -> 1 1 (n2 h) (n1 w)', n1=self.array_size[0], n2=self.array_size[1])
                log_image(self.opt, self.accelerator, image2, f"img{idx:04d}_gt_psfs_ch{k}_{ps_loc}", self.global_step)
        
        # if self.opt.val.patched:
        #     b,c,h,w = self.original_size[lq_key]
        #     pred = []
        #     for _ in self.setup_patches():
        #         pred.append(self.net_g(self.sample[lq_key]))
        #     pred = torch.cat(pred, dim=0)
        #     pred = einops.rearrange(pred, '(b n) c h w -> b n c h w', b=b)
        #     out = []
        #     for i in range(len(pred)):
        #         merged = merge_patches(pred[i], self.sample[lq_key+'_patched_pos'])
        #         out.append(merged[..., :h, :w])
        #     lq = self.sample[lq_key+'_original']
        #     gt = self.sample[gt_key+'_original']
        #     gt_depth = self.sample['depth_original']
        #     out = torch.stack(out)
        # else: 
        #     lq = self.sample[lq_key]
        #     gt = self.sample[gt_key]
        #     gt_depth = self.sample['depth']
        #     out = self.net_g(lq)
            
        # lq = lq.cpu().numpy()
        # gt = gt.cpu().numpy()
        # gt_depth = gt_depth.cpu().numpy()
        # out = out.cpu().numpy()
        # for i in range(len(gt)):
        #     idx += 1
        #     # save_images_as_zip(lq[i], self.opt.path.experiments_root, f'lq_{idx:04d}.zip')
        #     lq_i = einops.rearrange(lq[i], '(n1 n2) c h w -> c (n2 h) (n1 w)', n1=self.array_size[0], n2=self.array_size[1])
        #     lq_i = np.stack([lq_i])
        #     lq_i = np.clip(lq_i, 0, 1)
        #     log_image(self.opt, self.accelerator, lq_i, f'lq_{idx:04d}', self.global_step)

        #     image1 = [gt[i], out[i], gt_depth[i]]
        #     image1 = np.stack(image1)
        #     image1 = np.clip(image1, 0, 1)
        #     log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)
        #     log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx:04d}', self.global_step)
        #     log_metrics(gt_depth[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
        return idx

