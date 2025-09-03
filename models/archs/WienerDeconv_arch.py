import logging

import einops
import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from utils.dataset_utils import crop_arr


class WienerDeconvolution_config(PretrainedConfig):
    model_type = "WienerDeconvolutionModel"

    def __init__(self, psfs=None, is_trainable_psf=False, noise_weight=0.1, do_pad=True, do_clip=True, max_trainable=None, return_frequency=False, **kwargs):
        super().__init__(**kwargs)
        self.psfs = psfs
        self.is_trainable_psf = is_trainable_psf
        self.noise_weight = noise_weight
        self.do_pad = do_pad
        self.do_clip = do_clip
        self.max_trainable = max_trainable
        self.return_frequency = return_frequency


class WienerDeconvolution_arch(PreTrainedModel):
    """
    Performs Wiener Deconvolution in the frequency domain for each psf.

    psfs with shape (N inc outc n_psfs Hp Wp)
    initial_K has shape (1, 1, n_psfs) for each psf.

    """
    config_class = WienerDeconvolution_config
    def __init__(self, config):
        super().__init__(config)
        
        psfs = np.array(config.psfs, dtype=np.float32)
        psfs = psfs[:, :, None]

        self.is_trainable_psf = config.is_trainable_psf
        self.noise_weight = config.noise_weight
        self.do_pad = config.do_pad
        self.do_clip = config.do_clip
        self.max_trainable = config.max_trainable
        self.return_frequency = config.return_frequency

        h, w = psfs.shape[-2:]
        if self.do_pad:
            # make sure PSFs are padded.
            h1, h2 = int(np.floor(h / 2)), int(np.ceil(h / 2))
            w1, w2 = int(np.floor(w / 2)), int(np.ceil(w / 2))
            self.torch_pad = [h1, h2, w1, w2]

            psfs = crop_arr(psfs, h*2, w*2)
            h, w = psfs.shape[-2:]

        initial_Ks = torch.tensor(np.ones((1, 1, psfs.shape[-3], 1, 1)), dtype=torch.float32)
        self.Ks = nn.Parameter(initial_Ks, requires_grad=True)

        self.psfs = nn.Parameter(torch.from_numpy(psfs), requires_grad=False)
        if self.is_trainable_psf:
            if len(self.psfs.shape) == 5:
                self.psfs = einops.rearrange(self.psfs, '... -> 1 ...')

            if self.max_trainable is None:
                self._psfs_mask = torch.ones_like(self.psfs)
            else:
                self._psfs_mask = torch.zeros_like(self.psfs)
                hs = max(0, int(h//2 - np.ceil(self.max_trainable[0] / 2)))
                he = min(h, hs + self.max_trainable[0])
                ws = max(0, int(w//2 - np.ceil(self.max_trainable[1] / 2)))
                we = min(w, ws + self.max_trainable[1])
                self._psfs_mask[..., hs:he, ws:we] = 1
            self._psfs = nn.Parameter(self.psfs * self._psfs_mask, requires_grad=True)
        else:
            if len(self.psfs.shape) == 5:
                self.psfs[None]
            self._psfs = nn.Parameter(self.psfs, requires_grad=False)
        self._PSFS = None

    def forward(self, images, psfs=None):
        """
        y = (h*x) + n
        X = Y/H =1/H.Y
          = H'/(H^2).Y
          = H'Y/(|H^2|Y + N).Y
          = H'/(|H^2| + NSR).Y

        H shape (inc outc n h w)

        @param images: torch.Tensor
            input image with shape (N C H W)
        @param psfs: torch.Tensor
            psfs with shape (N inc outc n_psfs Hp Wp)
        @return: torch.Tensor
            deconvolve image with shape (N n_psfs C H W)
        """
        if psfs is not None:
            logging.warning("PSFs provided. PSFs in the WienerDeconv model will be ignored")
            self._PSFS = torch.fft.fft2(psfs, dim=(-2, -1))
        elif self.is_trainable_psf:
            self._PSFS = torch.fft.fft2(self._psfs * self._psfs_mask, dim=(-2, -1))
        else:
            if self._PSFS is None:
                self._PSFS = torch.fft.fft2(self._psfs, dim=(-2, -1))
            

        if self.do_pad:
            images = torch.nn.functional.pad(images, self.torch_pad, mode='reflect')  # this might move the tensor to cpu?
        b, c, h, w = images.shape
        _, inc, outc, npsfs, hp, wp = self._PSFS.shape
        assert outc == 1, "Not implemented for psf out channels > 1"
        assert inc == c or inc == 1 or c == 1, f"Not implemented for psf in channels ({inc}) != image channels ({c})"
        
        Y = torch.fft.fft2(images.type(torch.complex64), dim=(-2, -1))
        num = torch.conj(self._PSFS[:, :, 0]) * Y[:, :, None]
        den = torch.square(torch.abs(self._PSFS[:, :, 0])) + self.Ks*self.noise_weight
        # num = Y[:, :, None]
        # den = torch.ones((self._PSFS.shape[0],self._PSFS.shape[1], self._PSFS.shape[3], 1, 1), device=num.device)
        # print(num.shape, den.shape)
        X = num / den
        if self.do_pad:
            X = X[..., self.torch_pad[0]:-self.torch_pad[1], self.torch_pad[2]:-self.torch_pad[3]]
        if self.return_frequency:
            return X

        X = torch.real(torch.fft.ifftshift(torch.fft.ifft2(X, dim=(-2, -1)), dim=(-2, -1)))  # output horizontal flipped!
        if self.do_clip:
            X[X > 1] = 1
            X[X < -1] = -1
        return X
