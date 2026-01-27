import torch
import torch.nn.functional as F
import numpy as np
from models.NonBlindDeblurring.RL_model import RL_model

class FISTA_model(RL_model):
    """
    FISTA deconvolution model.
    This is a non-blind deconvolution algorithm, so it requires a PSF.
    It's an iterative algorithm. The number of iterations can be set in the options.
    This model supports both spatially-invariant and patch-wise spatially-varying deblurring.
    """

    def __init__(self, opt, logger=None):
        super().__init__(opt, logger)
        self.lambda_val = self.opt.get('lambda_val', 0.01)
        self.momentum = self.opt.get('momentum', 0.1)

    def deblur(self, blurred_image, psf):
        """
        Performs spatially-invariant FISTA deconvolution.

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

        x_k = blurred_image.clone()
        y_k = x_k.clone()
        t_k = 1

        if psf.shape[1] == 1:
            psf = psf.repeat(1, c, 1, 1)

        psf_conv = psf.view(b * c, 1, hk, wk)
        psf_flipped_conv = torch.flip(psf_conv, [2, 3])

        padding_h = (hk - 1) // 2
        padding_w = (wk - 1) // 2

        for i in range(self.iterations):
            # Reshape for grouped convolution
            y_k_reshaped = y_k.view(1, b * c, h, w)

            # Convolution with psf
            conv_y = F.conv2d(y_k_reshaped, psf_conv, padding=(padding_h, padding_w), groups=b*c)[:, :, :h, :w]
            conv_y = conv_y.view(b, c, h, w)

            # Gradient update
            grad = F.conv2d(
                (conv_y - blurred_image).view(1, b*c, h, w),
                psf_flipped_conv,
                padding=(padding_h, padding_w),
                groups=b*c
            )[:, :, :h, :w]
            grad = grad.view(b, c, h, w)

            x_k_1 = y_k - self.momentum * grad

            # Proximal operator (soft thresholding)
            x_k_1 = torch.sign(x_k_1) * torch.max(torch.abs(x_k_1) - self.lambda_val, torch.zeros_like(x_k_1))

            t_k_1 = (1 + np.sqrt(1 + 4 * t_k**2)) / 2

            y_k = x_k_1 + ((t_k - 1) / t_k_1) * (x_k_1 - x_k)

            t_k = t_k_1
            x_k = x_k_1

        return x_k
    
    def train(self):
        # Override the train method to just run validation since this model is not for training
        if not self.is_train:
            self.logger.info("====== FISTA Deblurring (testing only) =======")
            self.validation()
        else:
            self.logger.warning("FISTAModel is not designed for training. Please set is_train=False.")