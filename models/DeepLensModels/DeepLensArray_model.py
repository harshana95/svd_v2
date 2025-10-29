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

from deeplens.hybridlens import HybridLens
from deeplens.diffraclens import DiffractiveLens
from deeplens.optics.psf import conv_psf
from deeplens.optics.loss import PSFLoss
from deeplens.optics.diffractive_surface import Binary2, Pixel2D, Fresnel, Zernike

from utils.misc import log_metric

class DeepLensArray_model(BaseModel):
    def __init__(self, opt, logger):
        super(DeepLensArray_model, self).__init__(opt, logger)
        self.m_scale = 1e3  # TODO: check if we want to give depth in mm. This converts depth to mm

        # ============ define PSF simulator and image generator
        settings = opt.deeplens
        self.single_wavelength = settings.get("single_wavelength", False)
        self.depth_min = settings.depth_min
        self.depth_max = settings.depth_max

        self.fov = settings.fov
        self.fields = settings.fields
        
        # 1. initialize the target phase profile of the metasurface
        self.wavelength_set_m = np.array(settings.wavelength_set_m)  # if we are using hsi/rgb images, this should change
        # self.ps_locs = torch.tensor([list(map(float, v)) for v in settings.ps_locs])  # point spread locations (we use a sparse grid)
        # self.ps_locs *= self.m_scale
        # assert self.ps_locs.shape[0] > 0, f"Need at least one Point Spread location. Given {settings.ps_locs}"
        # logger.info(f"Point spread locations (in meters): {self.ps_locs}")
        
        
        # find the MS center positions of the array
        self.array_size = settings.array_size  # col, row
        self.array_spacing = settings.array_spacing # col, row
        self.MSArray_size = [(self.array_size[0]-1)*self.array_spacing[0], (self.array_size[1]-1)*self.array_spacing[1]] # width, height

        self.MS_pos = []
        self.MO = []
        for i in range(self.array_size[0]): # cols
            for j in range(self.array_size[1]): # rows
                lens = HybridLens(filename=settings.lens_file,
                               device=self.accelerator.device
                               )
                # lens.refocus(foc_dist=-1000)
                self.MO.append(lens)

                self.MS_pos.append([i*self.array_spacing[0] - self.MSArray_size[0]/2,   # x
                                    j*self.array_spacing[1] - self.MSArray_size[1]/2,   # y
                                    0])                                                 # z
                logger.info(f"MS {i},{j} center: {self.MS_pos[-1]}")

        self.MS_pos = torch.tensor(self.MS_pos) * self.m_scale
       
        # setup loss function
        self.psf_loss = PSFLoss().to(self.accelerator.device)

        self.current_fov = 0

    def save_other_parameters(self, path):
        for i, mo in enumerate(self.MO):
            mo.write_lens_json(os.path.join(path, f'lens_{i}.json'))
        

    def load_other_parameters(self, path):
        for i in range(len(self.MO)):
            self.MO[i].read_lens_json(os.path.join(path, f'lens_{i}.json'))  # this will replace the parameters !!!! FIX
        self.optimizers = []
        self.setup_optimizers()

    def setup_optimizers(self):
        opt = self.opt.train.optim

        # Optimizer creation
        optimizer_class = torch.optim.AdamW
        lens_params = {}
        for i in range(len(self.MO)):
            params = []
            # params += self.MO[i].geolens.get_optimizer_params(lrs=[1e-4, 1e-4, 1e-1, 1e-5], decay=0.01)
            params += self.MO[i].doe.get_optimizer_params(lr=0.1)
            # add shape parameters as a parameter to optimize
            optimizer = optimizer_class(
                params,
                betas=(opt.adam_beta1, opt.adam_beta2),
                weight_decay=opt.adam_weight_decay,
                eps=opt.adam_epsilon,
                )
            self.optimizers.append(optimizer)
        #     for param_group in params:
        #         lr = param_group['lr']
        #         if lr in lens_params:
        #             lens_params[lr]['params'] += param_group['params']
        #         else:
        #             lens_params[lr] = param_group
        # params = []
        # for lr in lens_params.keys():
        #     params.append(lens_params[lr])
            
        # # add shape parameters as a parameter to optimize
        # optimizer = optimizer_class(
        #     params,
        #     lr=opt.learning_rate,
        #     betas=(opt.adam_beta1, opt.adam_beta2),
        #     weight_decay=opt.adam_weight_decay,
        #     eps=opt.adam_epsilon,
        #     )
        # self.optimizers.append(optimizer)
                

    def setup_dataloaders(self):
        # create train and validation dataloaders
        train_set = create_dataset(self.opt.datasets.train)
        train_set.gt_key = None
        train_set.lq_key = None
        self.dataloader = DataLoader(
            train_set,
            shuffle=self.opt.datasets.train.use_shuffle,
            batch_size=self.opt.train.batch_size,
            num_workers=self.opt.datasets.train.get('num_worker_per_gpu', 1)
        )
                
        val_set = create_dataset(self.opt.datasets.val)
        val_set.gt_key = None
        val_set.lq_key = None
        self.test_dataloader = DataLoader(
            val_set,
            shuffle=self.opt.datasets.val.use_shuffle,
            batch_size=self.opt.val.batch_size,
            num_workers=self.opt.datasets.val.get('num_worker_per_gpu', 1)
        )

    def feed_data(self, data, is_train=True):
        levels = len(self.MO)
        n_fields = 3
        # depth = torch.rand(self.depth_levels)
        depth = torch.linspace(0, 1, steps=levels)
        depth = depth*(self.depth_max - self.depth_min) + self.depth_min
        depth = torch.sort(depth, descending=True)[0] # from far to near
        depth *= self.m_scale

        # _ps_locs = [torch.cat([self.ps_locs, -torch.ones(len(self.ps_locs), 1)*depth[_i]], dim=-1) for _i in range(levels)]
        # _ps_locs = torch.cat(_ps_locs, dim=0)  # [P, 3] P = n_pos * depth_levels
        
        self.all_psf_intensity = {}
        # simulate PSF for current p
        for i in range(len(self.MO)):
            if self.single_wavelength:
                wv_idx = (i//self.array_size[0])%len(self.wavelength_set_m)
                wv = self.wavelength_set_m[wv_idx].item()*1e6 # um   
            else:
                raise Exception()
            ks = None # 1024
            _depth = depth[i]
            if is_train:
                theta = np.random.rand()*2*np.pi
                _ps_locs = [[_depth*np.tan(field*np.pi/180)*np.cos(theta), 
                             _depth*np.tan(field*np.pi/180)*np.sin(theta), 
                             -_depth] for field in np.linspace(0, self.current_fov/2, n_fields)]
            else:
                _ps_locs = [[0,
                             _depth*np.tan(field*np.pi/180), 
                             -_depth] for field in np.linspace(0, self.fov/2, n_fields)]
            _ps_locs = torch.tensor(_ps_locs)
            ps_locs = _ps_locs - self.MS_pos[i] # MS at center, translate obj
            
            # looks like deeplens needs x,y in range -1 to 1
            ps_locs[:, 0] *= (self.MO[i].geolens.foclen/ps_locs[:, 2])*(2/self.MO[i].geolens.sensor_size[1])
            ps_locs[:, 1] *= (self.MO[i].geolens.foclen/ps_locs[:, 2])*(2/self.MO[i].geolens.sensor_size[0])

            self.all_psf_intensity[i] = []
            for ps_i in range(len(ps_locs)):
                psf = self.MO[i].psf(points=ps_locs[ps_i], ks=ks, wvln=wv, spp=1_000_000).to(torch.float32)
                psf = psf[None] # 1 h w
                self.all_psf_intensity[i].append(psf)
            self.all_psf_intensity[i] = torch.stack(self.all_psf_intensity[i]) # n 1 h w
        
        self.sample = data

            
    def optimize_parameters(self):
        self.current_fov = self.fov*((self.global_step/self.opt.train.max_train_steps)**0.5)
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        losses = {}
        total = 0
        for i in range(len(self.MO)):
            loss = self.psf_loss(self.all_psf_intensity[i])
            self.accelerator.backward(loss)
            self.optimizers[i].step()
            losses[f"{i}"] = loss
            total += loss
        losses['all'] = total
        log_metric(self.accelerator, {"current_fov":self.current_fov}, self.global_step)
        return losses

    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        
        for i in range(len(self.MO)):
            psfs = einops.rearrange(self.all_psf_intensity[i].detach().cpu().numpy(), 'n 1 h w -> n 1 h w')
            image1 = einops.rearrange(psfs, 'n c h w -> 1 c (n h) w')
            image1 = image1/image1.max()
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f"psfs_{i}", self.global_step)
        
        os.makedirs(os.path.join(self.opt.path.experiments_root, 'images'), exist_ok=True)
        for i in range(len(self.MO)):
            save_path = os.path.join(self.opt.path.experiments_root, 'images', f"DOELens_{i}.png")
            self.MO[i].draw_layout(save_name=save_path)
            log_image(self.opt, self.accelerator, plt.imread(save_path).transpose([2,0,1])[None], f"lens_{i}", self.global_step)

            phase = self.MO[i].doe._phase_map0().clone().detach().cpu()
            print(phase.min(), phase.max())
            phase = (phase - phase.min())/(phase.max() - phase.min())
            log_image(self.opt, self.accelerator, phase[None, None], f"doe_phase_{i}", self.global_step)


        return idx+1

