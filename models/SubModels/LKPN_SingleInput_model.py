import functools
import gc
import random
import einops
import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms

from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, T2IAdapter, EMAModel
from dataset import create_dataset
from dataset.ZipDatasetWrapper import ZipDatasetWrapper
from models.Simple_model import Simple_model
from models.archs import define_network
from models.archs.LKPN_arch import EfficientAffineConvolution
from pipelines.DiffusionWithLKPNPipeline import DiffusionTwoImageLKPNPipeline
from utils.dataset_utils import merge_patches
from utils import log_image, log_metrics


class LKPN_SingleInput_model(Simple_model):

    def __init__(self, opt, logger):
        super(LKPN_SingleInput_model, self).__init__(opt, logger)
        self.eac = EfficientAffineConvolution(k=opt.network.k)
        self.preprocessing_space = opt.preprocessing_space

        pretrained_model_name_or_path = self.opt.pretrained_model_name_or_path
        revision = self.opt.revision
        variant = self.opt.variant
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            revision=revision,
            variant=variant,
        )
        self.vae.requires_grad_(False)
        self.vae.to(self.accelerator.device)
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        
    def setup_optimizers(self):
        opt = self.opt.train.optim

        # Optimizer creation
        optimizer_class = torch.optim.AdamW
        lkpn_params_to_optimize = list(self.net_g.parameters())
        
        lkpn_optimizer = optimizer_class(
            lkpn_params_to_optimize,
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2),
            weight_decay=opt.adam_weight_decay,
            eps=opt.adam_epsilon,
            )
        self.optimizers.append(lkpn_optimizer)
        
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def _forward(self, image_1, gt_1):
        gt_latents = self.vae.encode(gt_1).latent_dist.sample() * self.vae.config.scaling_factor
        bsz = gt_latents.shape[0]
        # gt_latents = gt_latents.to(self.weight_dtype)


        # Cubic sampling to sample a random timestep for each image.
        # For more details about why cubic sampling is used, refer to section 3.4 of https://arxiv.org/abs/2302.08453
        timesteps = torch.rand((bsz,), device=gt_latents.device)
        timesteps = (1 - timesteps ** 3) * self.noise_scheduler.config.num_train_timesteps
        timesteps = timesteps.long().to(self.noise_scheduler.timesteps.dtype)
        timesteps = timesteps.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)

        #  # Sample noise that we'll add to the latents
        # noise = torch.randn_like(gt_latents)

        # # Add noise to the latents according to the noise magnitude at each timestep
        # noisy_latents = self.noise_scheduler.add_noise(gt_latents, noise, timesteps)

        # # Scale the noisy latents for the UNet
        # sigmas = self.get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
        # inp_noisy_latents = noisy_latents / ((sigmas ** 2 + 1) ** 0.5)  # these are around -4 to +4
        

        # get latents for img1 and img2
        if self.preprocessing_space == 'pixel':
            z_1 = image_1
            # z_t = self.vae.decode(inp_noisy_latents / self.vae.config.scaling_factor, return_dict=False)[0]
            z_0_1 = gt_1
        else:
            z_1 = self.vae.encode(image_1).latent_dist.sample() * self.vae.config.scaling_factor
            # z_t = inp_noisy_latents
            z_0_1 = gt_latents

        # calculate latent kernels
        k_1 = self.net_g(z_1, z_1, timesteps)
        
        # convolve latents with estimated kernels
        batch_size, channels, height, width = z_1.shape
        k_1 = k_1.view(batch_size, channels, self.net_g.k, self.net_g.k, height, width)
        k_1 = k_1.permute(0, 1, 4, 5, 2, 3)
        z_1_ref = self.eac(k_1, z_1)
        
        return z_1_ref, z_0_1

    def optimize_parameters(self):
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        image_1 = self.sample[lq_key]
        gt_1 = self.sample[gt_key]

        z_1_ref, z_0_1 = self._forward(image_1, gt_1)
        
        # LKPN loss
        loss_lkpn = torch.mean(torch.nn.functional.mse_loss(z_1_ref, z_0_1))  # latent space loss 
        # loss_lkpn += torch.mean(torch.nn.functional.mse_loss(self.vae.decode(z_1_ref / self.vae.config.scaling_factor).sample, gt_1))  # pixel space loss
        loss_lkpn *= 1
        self.accelerator.backward(loss_lkpn)
        if self.accelerator.sync_gradients:
            params_to_clip = list(self.net_g.parameters())
            self.accelerator.clip_grad_norm_(params_to_clip, 1.0)
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()
        return {'all': loss_lkpn}
    
    @torch.no_grad()
    def validation(self):
        gc.collect()
        torch.cuda.empty_cache()
        idx = 0
        for model in self.models:
            model.eval()
        dataloader = DataLoader(Subset(self.dataloader.dataset, np.arange(5)), 
                                shuffle=False, 
                                batch_size=1)
        print(f"Tesing using {len(dataloader)} training data...")
        dataloader = self.accelerator.prepare(dataloader)
        for batch in dataloader:
            idx = self.validate_step(batch, idx, self.dataloader.dataset.lq_key, self.dataloader.dataset.gt_key)
        self.accelerator._dataloaders.remove(dataloader)
        for batch in tqdm(self.test_dataloader):
            idx = self.validate_step(batch, idx, self.test_dataloader.dataset.lq_key, self.test_dataloader.dataset.gt_key)
            

        for model in self.models:
            model.train()
            
    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        
        if self.opt.val.patched:
            pass
        else: 
            lq1 = self.sample[lq_key]
            gt = self.sample[gt_key]
            z_1_ref, z_0_1 = self._forward(lq1, gt)
            if self.preprocessing_space == 'pixel':
                out = z_1_ref
            else:
                out = self.vae.decode(z_1_ref / self.vae.config.scaling_factor, return_dict=False)[0]
            
        lq1 = lq1.cpu().numpy()
        gt = gt.cpu().numpy()
        out = out.cpu().numpy()
        for i in range(len(gt)):
            idx += 1
            image1 = [lq1[i], gt[i], out[i]]
            for j in range(len(image1)):
                if image1[j].shape[0] == 1:
                    image1[j] = einops.repeat(image1[j], '1 h w -> 3 h w')
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx:04d}', self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)
            
        return idx
    