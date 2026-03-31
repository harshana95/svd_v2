# LucidFlux denoising loop.
# Source: W2GenAI-Lab/LucidFlux  (Apache 2.0)
# Adapted for use without LucidFlux's HFEmbedder dependency.

import math
from typing import Callable

import torch
from einops import rearrange
from torch import Tensor


def get_noise(num_samples, height, width, device, dtype, seed):
    return torch.randn(
        num_samples, 16,
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device, dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def time_shift(mu: float, sigma: float, t: Tensor) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1=256, y1=0.5, x2=4096, y2=1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(num_steps, image_seq_len, base_shift=0.5, max_shift=1.15, shift=True):
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2, pw=2,
    )


def denoise_lucidflux(
    model,                # diffusers FluxTransformer2DModel
    dual_condition_model,  # DualConditionBranch
    img: Tensor,          # packed noise [B, seq, 64]
    img_ids: Tensor,     # [B, seq, 3]
    txt: Tensor,          # T5 text [B, txt_len, 4096]
    txt_ids: Tensor,
    siglip_txt: Tensor,   # cat([T5, Redux]) [B, txt_len+1024, 4096]
    siglip_txt_ids: Tensor,
    vec: Tensor,          # CLIP pooled [B, 768]
    timesteps: list,
    guidance: float = 4.0,
    condition_cond_lq: Tensor = None,   # [B, 3, H, W] pixel [-1,1]
    condition_cond_pre: Tensor = None,  # [B, 3, H, W] pixel [-1,1]
) -> Tensor:
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        dtype = img.dtype

        # ---- DualConditionBranch ----
        block_res_samples = dual_condition_model(
            img=img,
            img_ids=img_ids.to(dtype),
            condition_cond_lq=condition_cond_lq.to(dtype),
            condition_cond_pre=condition_cond_pre.to(dtype),
            txt=txt.to(dtype),
            txt_ids=txt_ids.to(dtype),
            y=vec.to(dtype),
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        for i, block_res_sample in enumerate(block_res_samples):
            block_res_samples[i] = block_res_sample.to(dtype)


        # ---- Main Flux transformer (diffusers) ----
        pred = model(
            hidden_states=img,
            encoder_hidden_states=siglip_txt.to(dtype),
            pooled_projections=vec.to(dtype),
            timestep=t_vec,
            guidance=guidance_vec,
            txt_ids=siglip_txt_ids.to(dtype),
            img_ids=img_ids.to(dtype)[0], # no need for batch dim
            controlnet_block_samples=block_res_samples,
            return_dict=False,
        )[0]

        img = img + (t_prev - t_curr) * pred

    return img
