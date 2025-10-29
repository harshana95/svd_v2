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


@dataclass
class PipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]
    step_outputs: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]
    deblur_1: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]
    deblur_2: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]
    timesteps: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class DiffusionTwoImageLKPNPipeline(
    DiffusionPipeline,
    FromSingleFileMixin,
):
    def __init__(
            self,
            vae: AutoencoderKL,
            unet,
            lkpn_1,
            lkpn_2,
            eac,
            adapter1: Union[T2IAdapter, MultiAdapter, List[T2IAdapter]],
            adapter2: Union[T2IAdapter, MultiAdapter, List[T2IAdapter]],
            scheduler: KarrasDiffusionSchedulers,
            use_1_as_start=False,
            use_2_as_start=False,
            preprocessing_space='pixel'
    ):
        super().__init__()
        assert not (use_1_as_start and use_2_as_start), "Can't start from both"
        self.use_1_as_start = use_1_as_start
        self.use_2_as_start = use_2_as_start
        self.preprocessing_space = preprocessing_space
        self.register_modules(
            vae=vae,
            unet=unet,
            lkpn_1=lkpn_1,
            lkpn_2=lkpn_2,
            eac=eac,
            adapter1=adapter1,
            adapter2=adapter2,
            scheduler=scheduler,
        )
        self.register_to_config()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = self.unet.config.sample_size

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()
    def __call__(
            self,
            image_1: PipelineImageInput,
            image_2: PipelineImageInput,
            prompt_embeds,
            added_cond_kwargs,
            rescale_image_a, rescale_image_b,
            num_inference_steps=50,
            num_images_per_prompt=1,
            adapter_conditioning_scale=1.0,
            adapter_conditioning_factor=1.0,
            latents=None,
            generator=None,
            eta=0.0,
            output_intermediate_steps=True,
            n_output_intermediate_steps=10,
            *args,
            **kwargs):
        batch_size = image_1.shape[0]
        device = image_1.device #self._execution_device
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        ha, wa = int(image_1.shape[-2] * rescale_image_a), int(image_1.shape[-1] * rescale_image_a)
        hb, wb = int(image_2.shape[-2] * rescale_image_b), int(image_2.shape[-1] * rescale_image_b)
        image_1 = torch.nn.functional.interpolate(image_1, (ha, wa), mode='bilinear')
        image_2 = torch.nn.functional.interpolate(image_2, (hb, wb), mode='bilinear')
        image_2 = einops.repeat(image_2, 'b 1 h w -> b c h w', c=3)
        
        height, width = image_2.shape[-2:]

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps=None)

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=latents,
        )
        
        # get latents for img1 and img2
        z_1 = self.vae.encode(image_1).latent_dist.sample() * self.vae.config.scaling_factor
        z_2 = self.vae.encode(image_2).latent_dist.sample() * self.vae.config.scaling_factor
        
        lamb = 0.9
        if self.use_1_as_start:
            latents = lamb * latents + (1 - lamb) * z_1
        if self.use_2_as_start:
            latents = lamb * latents + (1 - lamb) * z_2
        
        # 6.1 Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings & adapter features
        step_outputs = []
        predeblurring_1 = []
        predeblurring_2 = []
        
        if output_intermediate_steps:
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            step_outputs.append(image)
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        if self.preprocessing_space == 'pixel':
            z_1 = image_1
            z_2 = image_2
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # calculate latent kernels
                if self.preprocessing_space == 'pixel':
                    z_t = self.vae.decode(latent_model_input / self.vae.config.scaling_factor, return_dict=False)[0]
                else:
                    z_t = latent_model_input
                k_1 = self.lkpn_1(z_t, z_1, torch.tensor([t]*batch_size, device=device))
                k_2 = self.lkpn_2(z_t, z_2, torch.tensor([t]*batch_size, device=device))

                # convolve latents with estimated kernels
                batch_size, channels, height, width = z_1.shape
                k_1 = k_1.view(batch_size, channels, self.lkpn_1.k, self.lkpn_1.k, height, width)
                k_2 = k_2.view(batch_size, channels, self.lkpn_2.k, self.lkpn_2.k, height, width)
                k_1 = k_1.permute(0, 1, 4, 5, 2, 3)
                k_2 = k_2.permute(0, 1, 4, 5, 2, 3)
                z_1_ref = self.eac(k_1, z_1)
                z_2_ref = self.eac(k_2, z_2)

                # generate adapter features
                adapter_state1 = self.adapter1(torch.cat([z_1, z_1_ref], dim=-3))
                adapter_state2 = self.adapter2(torch.cat([z_2, z_2_ref], dim=-3))
                adapter_state = []
                for _i in range(len(adapter_state1)):
                    adapter_state.append((adapter_state1[_i] * adapter_state2[_i]) * adapter_conditioning_scale)

                # predict the noise residual
                if i < int(num_inference_steps * adapter_conditioning_factor):
                    down_intrablock_additional_residuals = [state.clone() for state in adapter_state]
                else:
                    down_intrablock_additional_residuals = None

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds[:batch_size],
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    added_cond_kwargs={k: v[:batch_size] for k, v in added_cond_kwargs.items()},
                    return_dict=False,
                )[0]
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if (output_intermediate_steps and i % (len(timesteps) // n_output_intermediate_steps) == 0) or i == len(timesteps) - 1:
                    dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
                    if needs_upcasting:
                        self.upcast_vae()
                        latents = latents.to(dtype)
                        z_1_ref = z_1_ref.to(dtype)
                        z_2_ref = z_2_ref.to(dtype)

                    step_outputs.append(self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0])
                    if self.preprocessing_space == "latent":
                        predeblurring_1.append(self.vae.decode(z_1_ref / self.vae.config.scaling_factor, return_dict=False)[0])
                        predeblurring_2.append(self.vae.decode(z_2_ref / self.vae.config.scaling_factor, return_dict=False)[0])
                    else:
                        predeblurring_1.append(z_1_ref)
                        predeblurring_2.append(z_2_ref)
                    # cast back to fp16 if needed
                    if needs_upcasting:
                        self.vae.to(dtype=torch.float16)
                progress_bar.update()

        # Offload all models
        self.maybe_free_model_hooks()
        return PipelineOutput(images=step_outputs[-1], step_outputs=step_outputs, deblur_1=predeblurring_1, deblur_2=predeblurring_2, timesteps=timesteps)


class DiffusionSingleImageLKPNPipeline(
    DiffusionPipeline,
    FromSingleFileMixin,
):
    def __init__(
            self,
            vae: AutoencoderKL,
            unet,
            lkpn,
            eac,
            adapter1: Union[T2IAdapter, MultiAdapter, List[T2IAdapter]],
            scheduler: KarrasDiffusionSchedulers,
            use_1_as_start=False,
            preprocessing_space='pixel'
    ):
        super().__init__()
        self.use_1_as_start = use_1_as_start
        self.preprocessing_space = preprocessing_space
        self.register_modules(
            vae=vae,
            unet=unet,
            lkpn=lkpn,
            eac=eac,
            adapter1=adapter1,
            scheduler=scheduler,
        )
        self.register_to_config()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = self.unet.config.sample_size

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()
    def __call__(
            self,
            image_1: PipelineImageInput,
            prompt_embeds,
            added_cond_kwargs,
            rescale_image_a,
            num_inference_steps=50,
            num_images_per_prompt=1,
            adapter_conditioning_scale=1.0,
            adapter_conditioning_factor=1.0,
            latents=None,
            generator=None,
            eta=0.0,
            output_intermediate_steps=True,
            n_output_intermediate_steps=10,
            *args,
            **kwargs):
        batch_size = image_1.shape[0]
        device = image_1.device #self._execution_device
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        ha, wa = int(image_1.shape[-2] * rescale_image_a), int(image_1.shape[-1] * rescale_image_a)
        image_1 = torch.nn.functional.interpolate(image_1, (ha, wa), mode='bilinear')
        
        height, width = image_1.shape[-2:]

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps=None)

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=latents,
        )
        
        # get latents for img1 and img2
        z_1 = self.vae.encode(image_1).latent_dist.sample() * self.vae.config.scaling_factor
        
        lamb = 0.9
        if self.use_1_as_start:
            # image_2 = einops.repeat(image_2, 'b 1 h w -> b c h w', c=3)
            latents = lamb * latents + (1 - lamb) * z_1
        
        # 6.1 Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings & adapter features
        step_outputs = []
        predeblurring_1 = []
        if output_intermediate_steps:
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            step_outputs.append(image)
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        if self.preprocessing_space == 'pixel':
            z_1 = image_1
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # calculate latent kernels
                if self.preprocessing_space == 'pixel':
                    z_t = self.vae.decode(latent_model_input / self.vae.config.scaling_factor, return_dict=False)[0]
                else:
                    z_t = latent_model_input
                k_1 = self.lkpn(z_t, z_1, torch.tensor([t]*batch_size, device=device))

                # convolve latents with estimated kernels
                batch_size, channels, height, width = latent_model_input.shape
                k_1 = k_1.view(batch_size, channels, self.lkpn.k, self.lkpn.k, height, width)
                k_1 = k_1.permute(0, 1, 4, 5, 2, 3)
                z_1_ref = self.eac(k_1, z_1)

                # generate adapter features
                adapter_state1 = self.adapter1(torch.cat([z_1, z_1_ref], dim=-3))
                adapter_state = []
                for _i in range(len(adapter_state1)):
                    adapter_state.append((adapter_state1[_i]) * adapter_conditioning_scale)

                # predict the noise residual
                if i < int(num_inference_steps * adapter_conditioning_factor):
                    down_intrablock_additional_residuals = [state.clone() for state in adapter_state]
                else:
                    down_intrablock_additional_residuals = None

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds[:batch_size],
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    added_cond_kwargs={k: v[:batch_size] for k, v in added_cond_kwargs.items()},
                    return_dict=False,
                )[0]
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if (output_intermediate_steps and i % (len(timesteps) // n_output_intermediate_steps) == 0) or i == len(timesteps) - 1:
                    dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
                    if needs_upcasting:
                        self.upcast_vae()
                        latents = latents.to(dtype)
                        z_1_ref = z_1_ref.to(dtype)

                    step_outputs.append(self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0])
                    if self.preprocessing_space == "latent":
                        predeblurring_1.append(self.vae.decode(z_1_ref / self.vae.config.scaling_factor, return_dict=False)[0])
                    else:
                        predeblurring_1.append(z_1_ref)
                    # cast back to fp16 if needed
                    if needs_upcasting:
                        self.vae.to(dtype=torch.float16)
                progress_bar.update()

        # Offload all models
        self.maybe_free_model_hooks()
        return PipelineOutput(images=step_outputs[-1], step_outputs=step_outputs, deblur_1=predeblurring_1, deblur_2=[], timesteps=timesteps)
