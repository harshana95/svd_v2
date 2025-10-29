import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Optional

import PIL
import einops
import numpy as np
import scipy.io
import torch
from diffusers import T2IAdapter, MultiAdapter, AutoencoderKL
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor, LoRAXFormersAttnProcessor, \
    LoRAAttnProcessor2_0

def preprocess_adapter_input(image):
    # use a HPF to extract high-frequency components
    r = 15  # radius of the low-frequency center to be removed
    b, c, h, w = image.shape
    crow, ccol = h // 2, w // 2
    
    image_fft = torch.fft.fft2(image)
    image_fftshift = torch.fft.fftshift(image_fft)
    
    # create a mask first, center square is 0, remaining all ones
    mask = torch.ones((b, c, h, w), device=image.device)
    mask[:, :, crow - r:crow + r, ccol - r:ccol + r] = 0
    
    # apply mask and inverse FFT
    fshift = image_fftshift * mask
    f_ishift = torch.fft.ifftshift(fshift)
    img_back = torch.fft.ifft2(f_ishift)
    img_back = torch.real(img_back)
    return img_back

@dataclass
class PipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]
    timesteps: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]



class OSEDiffPipeline(
    DiffusionPipeline,
    FromSingleFileMixin,
):
    def __init__(
            self,
            vae: AutoencoderKL,
            unet,
            scheduler: KarrasDiffusionSchedulers,
            adapter: Optional[T2IAdapter] = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            adapter=adapter,
        )
        self.register_to_config()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = self.unet.config.sample_size

    @torch.no_grad()
    def __call__(
            self,
            image_1: PipelineImageInput,
            image_2: PipelineImageInput,
            prompt_embeds,
            added_cond_kwargs=None,
            timesteps=[999],
            latents=None,
            eta=0.0,
            *args,
            **kwargs):
        t = timesteps[0]
        batch_size = image_1.shape[0]
        device = image_1.device #self._execution_device
        if image_2 is not None:
            if image_2.shape[1] == 3:
                image_2 = image_2[:, 1:2]
            vae_input = torch.cat([image_1, image_2], dim=1)
        else:
            vae_input = image_1
        height, width = image_1.shape[-2:]
        
        # get latents
        latents = self.vae.encode(vae_input).latent_dist.sample() * self.vae.config.scaling_factor
        
        # expand the latents if we are doing classifier free guidance
        latent_model_input = latents
        # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        if self.adapter is not None:
            if image_2 is not None:
                down_block_additional_residuals = self.adapter(preprocess_adapter_input(image_2))
            else:
                down_block_additional_residuals = self.adapter(preprocess_adapter_input(image_1))
        else:
            down_block_additional_residuals = None

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds[:batch_size],
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals, 
            return_dict=False,
        )[0]
        
        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        out = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0].clip(-1, 1)
        # Offload all models
        self.maybe_free_model_hooks()
        return PipelineOutput(images=out,timesteps=timesteps)
