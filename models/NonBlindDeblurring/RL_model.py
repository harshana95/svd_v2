import torch
import torch.nn.functional as F
import numpy as np
import einops
from torch.utils.data import DataLoader
from dataset import create_dataset
from models.base_model import BaseModel
from utils import log_image, log_metrics
from utils.dataset_utils import merge_patches

class RL_model(BaseModel):
    """
    Richardson-Lucy deconvolution model.
    This is a non-blind deconvolution algorithm, so it requires a PSF.
    It's an iterative algorithm. The number of iterations can be set in the options.
    This model supports both spatially-invariant and patch-wise spatially-varying deblurring.
    """

    def __init__(self, opt, logger=None):
        super().__init__(opt, logger)
        self.iterations = self.opt.get('iterations', 30)
        self.spatially_varying = self.opt.get('spatially_varying', False)
        self.sensor_size = self.opt.get('sensor_size', 1024)
        self.grid_size = self.opt.get('grid', 1)
        self.ks = self.opt.get('ks', 65)
        assert self.ks % 2 == 1
        if self.spatially_varying:
            pad_1  = (self.ks - 1) // 2
            pad_2 = self.ks // 2
            patch_size = self.sensor_size // self.grid_size + pad_1 + pad_2
            self.total_pad = pad_1 + pad_2
            self.patch_size = patch_size
            self.stride = self.sensor_size // self.grid_size

    def setup_dataloaders(self):
        # create train and validation dataloaders
        train_set = create_dataset(self.opt.datasets.train)
        self.dataloader = DataLoader(
            train_set,
            shuffle=self.opt.datasets.train.use_shuffle,
            batch_size=self.opt.train.batch_size,
            num_workers=0,
        )
        val_set = create_dataset(self.opt.datasets.val)
        self.test_dataloader = DataLoader(
            val_set,
            shuffle=self.opt.datasets.val.use_shuffle,
            batch_size=self.opt.val.batch_size,
            num_workers=0,
        )

    def deblur(self, blurred_image, psf, iterations):
        """
        Performs spatially-invariant Richardson-Lucy deconvolution using a single PSF.

        Args:
            blurred_image (torch.Tensor): The blurred image (B, C, H, W).
            psf (torch.Tensor): The Point Spread Function (B, 1, kH, kW).
            iterations (int): The number of iterations to perform.

        Returns:
            torch.Tensor: The deblurred image (B, C, H, W).
        """
        # breakpoint()
        b, c, h, w = blurred_image.shape
        if len(psf.shape) == 6:
            assert psf.shape[1] == 1 and psf.shape[2] == 1
            psf = psf[:, 0, 0]
        _, _, hk, wk = psf.shape

        latent_est = blurred_image.clone()
        
        # Prepare psf for grouped convolution: (B, 1, kH, kW) -> (B*C, 1, kH, kW)
        if psf.shape[1] == 1:
            psf = psf.repeat(1, c, 1, 1)
        psf_conv = psf.view(b * c, 1, hk, wk)
        psf_flipped_conv = torch.flip(psf_conv, [2, 3])
        
        # TOOD: fix conv size issue when kernel size is even
        padding_h = (hk - 1) // 2 #+ 1
        padding_w = (wk - 1) // 2 #+ 1

        for i in range(iterations):
            # Reshape latent_est for grouped convolution: (B, C, H, W) -> (1, B*C, H, W)
            latent_est_reshaped = latent_est.view(1, b * c, h, w)
            
            # Convolve with psf
            conv_latent = F.conv2d(latent_est_reshaped, psf_conv, padding=(padding_h, padding_w), groups=b * c)[:, :, :h, :w]
            conv_latent = conv_latent.view(b, c, h, w)
            
            relative_blur = blurred_image / (conv_latent + 1e-8)
            
            # Reshape for correlation (convolution with flipped psf)
            relative_blur_reshaped = relative_blur.view(1, b * c, h, w)
            
            # Correlate
            error_corr = F.conv2d(relative_blur_reshaped, psf_flipped_conv, padding=(padding_h, padding_w), groups=b * c)[:, :, :h, :w]
            error_corr = error_corr.view(b, c, h, w)
            
            latent_est = latent_est * error_corr

        return latent_est

    def deblur_sv_patchwise(self, blurred_image, psfs_patches, iterations, patch_size, stride):
        """
        Performs spatially-varying Richardson-Lucy deconvolution using a patch-wise approach.

        Args:
            blurred_image (torch.Tensor): The blurred image (B, C, H, W).
            psfs_patches (torch.Tensor): The PSFs for each patch (B, num_patches, 1, kH, kW).
            iterations (int): The number of iterations.
            patch_size (int): The size of the patches.
            stride (int): The stride between patches.

        Returns:
            torch.Tensor: The deblurred image (B, C, H, W).
        """
        B, C, H, W = blurred_image.shape
        if len(psfs_patches.shape) == 6:
            psfs_patches = einops.rearrange(psfs_patches, 'B a b c h w -> B (a b) c h w')
        _, _, _, kH, kW = psfs_patches.shape
        
        # 1. Unfold the blurred image into patches
        patches = F.unfold(blurred_image, kernel_size=patch_size, stride=stride, padding=self.total_pad//2)
        pos = torch.stack(torch.meshgrid(torch.arange(0, H+self.total_pad-patch_size+1, stride+1), torch.arange(0, W+self.total_pad-patch_size+1, stride+1)))
        pos = pos.view(pos.shape[0], -1).permute(1,0)[None] 
        pos = pos.repeat(B, 1, 1)
        
        # Reshape patches to be a batch of images: (B, C*patch_size*patch_size, num_patches) -> (B*num_patches, C, patch_size, patch_size)
        patches = patches.view(B, C, patch_size, patch_size, -1)
        num_patches = patches.shape[-1]
        patches = patches.permute(0, 4, 1, 2, 3).contiguous()
        patches = patches.view(B * num_patches, C, patch_size, patch_size)
        
        # Reshape psfs to match the patches batch: (B, num_patches, 1, kH, kW) -> (B*num_patches, 1, kH, kW)
        psfs_patches = psfs_patches.view(B * num_patches, -1, kH, kW)

        # 2. Deblur the batch of patches using the spatially-invariant RL method
        deblurred_patches = self.deblur(patches, psfs_patches, iterations)
        
        # 3. Fold the deblurred patches back into an image
        # Reshape deblurred_patches back to (B, C*patch_size*patch_size, num_patches)
        deblurred_patches = deblurred_patches.view(B, num_patches, C, patch_size, patch_size)
        final_image = []
        for b in range(B):
            deblurred_image = merge_patches(deblurred_patches[b], pos[b])
            final_image.append(deblurred_image[:, self.total_pad//2:self.total_pad//2+H, self.total_pad//2:self.total_pad//2+W])
        final_image = torch.stack(final_image)

        # deblurred_patches = deblurred_patches.permute(0, 2, 3, 4, 1).contiguous()
        # deblurred_patches = deblurred_patches.view(B, C * patch_size * patch_size, num_patches)
        
        # # Use fold to stitch them back. It sums overlapping regions.
        # folded_image = F.fold(deblurred_patches, output_size=(H, W), kernel_size=patch_size, stride=stride, padding=self.total_pad//2)
        
        # # 4. Create a weight map to average overlapping regions
        # one_patches = torch.ones_like(deblurred_patches)
        # weight_map = F.fold(one_patches, output_size=(H, W), kernel_size=patch_size, stride=stride, padding=self.total_pad//2)
        # weight_map = torch.where(weight_map == 0, torch.ones_like(weight_map), weight_map)
        
        # final_image = folded_image / weight_map
        
        return final_image

    def feed_data(self, data, is_train=True):
        # In non-blind deblurring, 'blur' is the blurred image
        # and we need a psf.
        self.blurred = data['blur'].to(self.device)
        self.psf = data['psfs'].to(self.device) # For SV, this should contain patch PSFs
        self.psf_centers = data['psf_centers'].to(self.device)
        self.gt = data.get('gt', None)
        if self.gt is not None:
            self.gt = self.gt.to(self.device)

    @torch.no_grad()
    def validate_step(self, batch, idx, lq_key, gt_key):
        self.feed_data(batch, is_train=False)
        
        if self.spatially_varying:
            deblurred_image = self.deblur_sv_patchwise(self.blurred, self.psf, self.iterations, self.patch_size, self.stride)
        else:
            deblurred_image = self.deblur(self.blurred, self.psf, self.iterations)
            
        psf_pos_top_left = (self.psf_centers*0.5+0.5)*self.sensor_size-self.psf.shape[-1]//2 # x, y (x-y coordinate)
        psf_pos_top_left[..., 1] = self.sensor_size - psf_pos_top_left[..., 1] # j, i
        psf_pos_top_left[..., 0] += self.total_pad
        psf_pos_top_left = psf_pos_top_left.flip(dims=[-1]) # i, j
        
        # we can drop batch because all psfs in a batch are the same
        merged_psfs = merge_patches(einops.rearrange(self.psf[0], 'a b c h w -> (a b) c h w'), 
                                    einops.rearrange(psf_pos_top_left[0], 'a b n -> (a b) n').cpu().numpy())
        merged_psfs = merged_psfs[None].cpu().numpy()
        merged_psfs /= merged_psfs.max()

        lq = self.blurred.cpu().numpy()
        gt = self.gt.cpu().numpy()
        out = deblurred_image.cpu().numpy()
        for i in range(len(gt)):
            idx += 1
            image1 = np.stack([lq[i], gt[i], out[i]])
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx:04d}', self.global_step)
            log_image(self.opt, self.accelerator, np.clip(merged_psfs, 0,1), f'psfs_{idx:04d}', self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)        
        return idx

    def train(self):
        # Override the train method to just run validation since this model is not for training
        if not self.is_train:
            self.logger.info("====== Richardson-Lucy Deblurring (testing only) =======")
            self.validation()
        else:
            self.logger.warning("RLModel is not designed for training. Please set is_train=False.")

    # The following methods are overridden because RL model has no trainable parameters.
    def setup_optimizers(self):
        self.logger.info("RLModel has no parameters to optimize. Skipping optimizer setup.")
        self.optimizers = []

    def setup_schedulers(self):
        self.logger.info("RLModel has no optimizers. Skipping scheduler setup.")
        self.schedulers = []

    def optimize_parameters(self):
        # No-op
        return {} # must return a dict of losses
