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
from deeplens.geolens import GeoLens
from deeplens.optics.psf import conv_psf
from deeplens.optics.loss import PSFLoss
from deeplens.optics.diffractive_surface import Binary2, Pixel2D, Fresnel, Zernike
from deeplens import PSFNet

from utils.misc import log_metric

class PSF_model(BaseModel):
    def __init__(self, opt, logger):
        super(PSF_model, self).__init__(opt, logger)
        self.m_scale = 1e3  # This converts depth to mm

        self.psf_rescale_factor = 1
        
        settings = opt.deeplens
        self.single_wavelength = settings.get("single_wavelength", False)
        self.spp = settings.get("spp", 100000)
        self.kernel_size = settings.get("kernel_size", 65)

        self.depth_min = -settings.depth_min * self.m_scale
        self.depth_max = -settings.depth_max * self.m_scale
        self.fov = settings.fov
        self.foc_d_arr = np.array(
            [
                -400,
                -425,
                -450,
                -500,
                -550,
                -600,
                -650,
                -700,
                -800,
                -900,
                -1000,
                -1250,
                -1500,
                -1750,
                -2000,
                -2500,
                -3000,
                -4000,
                -5000,
                -6000,
                -8000,
                -10000,
                -12000,
                -15000,
                -20000,
            ]
        )
        # normalize focal distance [0, 1]
        self.foc_z_arr = (self.foc_d_arr - self.depth_min) / (self.depth_max - self.depth_min)

        # initialize optics
        self.wavelength_set_m = np.array(settings.wavelength_set_m)  # if we are using hsi/rgb images, this should change
        self.lens = GeoLens(filename=settings.lens_file, device=self.accelerator.device)
        # self.lens = HybridLens(filename=settings.lens_file, device=self.accelerator.device)
        
        # lens.refocus(foc_dist=-1000)
       
        self.net_g = define_network(opt.network)
        self.net_g.train()
        self.models.append(self.net_g)

        # setup loss function
        self.psf_loss = PSFLoss().to(self.accelerator.device)
        self.criterion = Loss(self.opt['train'].loss).to(self.accelerator.device)    

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
    
    def depth2z(self, depth):
        z = (depth - self.depth_min) / (self.depth_max - self.depth_min)
        z = torch.clamp(z, min=0, max=1)
        return z

    def z2depth(self, z):
        depth = z * (self.depth_max - self.depth_min) + self.depth_min
        return depth
    
    def feed_data(self, data, is_train=True):
        lens = self.lens
        spp = self.spp

        num_points = len(data['x'])
        with torch.no_grad():
            # In each iteration, sample only one f_d
            # Sample (x, y), uniform distribution
            # Sample (z), Gaussian distribution (3-sigma interval)
            if is_train:
                foc_z = float(np.random.choice(self.foc_z_arr))
                x = (torch.rand(num_points) - 0.5) * 2
                y = (torch.rand(num_points) - 0.5) * 2
                z_gauss = torch.clamp(torch.randn(num_points), min=-3, max=3)
            else:
                batch_idx = data['x'][0]//num_points
                total_batches = len(self.test_dataloader)
                foc_z = float(self.foc_z_arr[int(len(self.foc_z_arr)*batch_idx/total_batches)])
                y, x = torch.meshgrid(torch.linspace(-1, 1, 7), torch.linspace(1, -1, 7), indexing='xy')
                x = x.flatten()
                y = y.flatten()
                z_gauss = torch.clamp(torch.linspace(-1, 1, len(x)), min=-3, max=3)
            
            # refocus; changes the sensor position
            foc_dist = self.z2depth(foc_z) # mm
            lens.refocus(foc_dist)

            z = torch.zeros_like(z_gauss)
            # sample [foc_z, 1], then scale to [foc_d, dmax]
            z[z_gauss > 0] = (1 - foc_z) * z_gauss[z_gauss > 0] / 3 + foc_z
            # sample [0, foc_z], then scale to [dmin, foc_d]
            z[z_gauss < 0] = foc_z * z_gauss[z_gauss < 0] / 3 + foc_z

            # Network input, shape of [N, 4] 
            # (x,y norm to -1,1; z norm to 0,1; foc_z norm to 0,1)
            foc_z_tensor = torch.full_like(x, foc_z)
            inp = torch.stack((x, y, z, foc_z_tensor), dim=-1)

            # Ray tracing to compute PSFs, shape of [N, 3, ks, ks]
            depth = self.z2depth(z)
            points = torch.stack((x, y, depth), dim=-1)
            psf = lens.psf_rgb(points=points, ks=self.kernel_size, spp=spp)

        
        self.sample = {
            'inp': inp.to(self.accelerator.device),
            'psf': psf.to(self.accelerator.device),
        }

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        losses = {}
        pred_psf = self.net_g(self.sample['inp'])/self.psf_rescale_factor
        psf = self.sample['psf']

        psf_max = psf.max()
        loss = self.criterion(pred_psf/psf_max, psf/psf_max) 

        # rescale loss because the loss is too small??
        self.accelerator.backward(loss['all']*1e6)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        log_metric(self.accelerator, {'parameter mean': next(self.net_g.parameters()).mean().item()}, self.global_step)
        # breakpoint()
        for optimizer in self.optimizers:
            optimizer.step()
        return losses

    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)

        pred_psf = self.net_g(self.sample['inp']).cpu().numpy()/self.psf_rescale_factor
        real_psf = self.sample['psf'].cpu().numpy()

        N = int(pred_psf.shape[0]**0.5)
        pred_psf = einops.rearrange(pred_psf, '(W H) c h w -> 1 c (H h) (W w)', H=N, W=N)
        real_psf = einops.rearrange(real_psf, '(W H) c h w -> 1 c (H h) (W w)', H=N, W=N)

        # breakpoint()
        # image1 = np.concatenate([pred_psf, pred_psf/pred_psf.max()*psf.max(), psf, abs(pred_psf-psf)], axis=-1)
        norm = real_psf.max()
        log_image(self.opt, self.accelerator, 1-np.clip(pred_psf/norm, 0, 1), f"pred_psfs", self.global_step)
        log_image(self.opt, self.accelerator, 1-np.clip(real_psf/norm, 0, 1), f"real_psfs", self.global_step)

        if idx == 0:
            os.makedirs(os.path.join(self.opt.path.experiments_root, 'images'), exist_ok=True)
            save_path = os.path.join(self.opt.path.experiments_root, 'images', f"Lens.png")
            self.lens.draw_layout(filename=save_path)
            log_image(self.opt, self.accelerator, plt.imread(save_path).transpose([2,0,1])[None], f"lens", self.global_step)

        return idx+1

