"""One-step diffusion pipeline for Stable Diffusion 3 Medium (flow matching, transformer)."""
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput

from .OSEDiffPipeline import PipelineOutput


class OSEDiffPipelineSD3(DiffusionPipeline):
    """One-step pipeline using SD3 transformer and flow-matching scheduler."""

    def __init__(
        self,
        vae: AutoencoderKL,
        transformer,
        scheduler: FlowMatchEulerDiscreteScheduler,
        concatenate_images: bool = False,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.concatenate_images = concatenate_images
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = getattr(
            self.transformer.config, "sample_size", 128
        )

    @torch.no_grad()
    def __call__(
        self,
        image_1: PipelineImageInput,
        image_2: PipelineImageInput,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        timesteps=None,
        latents=None,
        *args,
        **kwargs,
    ):
        if timesteps is None:
            timesteps = [self.scheduler.timesteps[0].item()]
        t = timesteps[0]
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=image_1.device, dtype=torch.float32)
        elif not isinstance(t, torch.Tensor):
            t = self.scheduler.timesteps[0].to(image_1.device)

        batch_size = image_1.shape[0]
        device = image_1.device
        vae_input = image_1
        if image_2 is not None and self.concatenate_images:
            vae_input = torch.cat([image_1, image_2], dim=1)

        latents = self.vae.encode(vae_input).latent_dist.sample() * self.vae.config.scaling_factor
        t_batch = t.expand(batch_size).to(device) if t.numel() == 1 else t

        model_pred = self.transformer(
            hidden_states=latents,
            timestep=t_batch,
            encoder_hidden_states=prompt_embeds[:batch_size],
            pooled_projections=pooled_prompt_embeds[:batch_size],
            return_dict=False,
        )[0]

        timestep_val = t_batch[0] if t_batch.numel() >= 1 else t
        latents = self.scheduler.step(model_pred, timestep_val, latents, return_dict=False)[0]
        self.scheduler._step_index = 0 # roll back scheduler step_index (only single step!)

        latents = latents / self.vae.config.scaling_factor
        if getattr(self.vae.config, "shift_factor", None) is not None:
            latents = latents + self.vae.config.shift_factor
        out = self.vae.decode(latents, return_dict=False)[0].clamp(-1, 1)

        self.maybe_free_model_hooks()
        return PipelineOutput(images=out, timesteps=timesteps)
