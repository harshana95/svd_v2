import torch
import torch.nn.functional as F
import numpy as np
import einops
from torch.utils.data import DataLoader
from dataset import create_dataset
from models.base_model import BaseModel
from utils import log_image, log_metrics
from utils.dataset_utils import merge_patches
from models.NonBlindDeblurring.RL_model import RL_model

class Wiener_model(RL_model):
    """
    Wiener deconvolution model.
    This is a non-blind deconvolution algorithm, so it requires a PSF.
    The noise-to-signal power ratio (K) can be set in the options.
    This model supports both spatially-invariant and patch-wise spatially-varying deblurring.
    """

    def __init__(self, opt, logger=None):
        super().__init__(opt, logger)
        self.noise_level = self.opt.get('noise_level', 0.01)

    def deblur(self, blurred_image, psf):
        """
        Performs spatially-invariant Wiener deconvolution using a single PSF.

        Args:
            blurred_image (torch.Tensor): The blurred image (B, C, H, W).
            psf (torch.Tensor): The Point Spread Function (B, 1, kH, kW).

        Returns:
            torch.Tensor: The deblurred image (B, C, H, W).
        """
        b, c, h, w = blurred_image.shape
        if len(psf.shape) == 6:
            assert psf.shape[1] == 1 and psf.shape[2] == 1
            psf = psf[:, 0, 0]
        _, _, hk, wk = psf.shape

        # Pad PSF to match image size
        pad_h = h - hk
        pad_w = w - wk
        psf_padded = F.pad(psf, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

        # To frequency domain
        blurred_fft = torch.fft.fft2(blurred_image)
        psf_fft = torch.fft.fft2(psf_padded)

        # Wiener filter
        psf_fft_conj = torch.conj(psf_fft)
        wiener_filter = psf_fft_conj / (torch.abs(psf_fft)**2 + self.noise_level)

        # Apply filter
        deblurred_fft = blurred_fft * wiener_filter

        # Back to spatial domain
        deblurred_image = torch.fft.ifftshift(torch.fft.ifft2(deblurred_fft), dim=(-2, -1)).real
        
        return deblurred_image
    
    def train(self):
        # Override the train method to just run validation since this model is not for training
        if not self.is_train:
            self.logger.info("====== Wiener Deconvolution (testing only) =======")
            self.validation()
        else:
            self.logger.warning("WienerModel is not designed for training. Please set is_train=False.")
