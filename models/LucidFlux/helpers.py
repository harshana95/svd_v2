import torch
import torch.nn as nn
from diffusers import Flux2Pipeline
from peft import LoraConfig, get_peft_model

try:
    from diffusers import Flux2KleinPipeline
except ImportError:
    Flux2KleinPipeline = None

PROMPT = "restore this image"


def get_flux2_pipeline_class(model_id: str):
    if Flux2KleinPipeline is not None and "klein" in model_id.lower():
        return Flux2KleinPipeline
    return Flux2Pipeline


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def encode_vae_image(
    vae,
    image: torch.Tensor,
    generator: torch.Generator,
    flux2_pipe_cls=Flux2Pipeline,
):
    if image.ndim != 4:
        raise ValueError(f"Expected image dims 4, got {image.ndim}.")

    image_latents = retrieve_latents(
        vae.encode(image), generator=generator, sample_mode="argmax"
    )
    image_latents = flux2_pipe_cls._patchify_latents(image_latents)

    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
        image_latents.device, image_latents.dtype
    )
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    )
    image_latents = (image_latents - latents_bn_mean) / latents_bn_std
    return image_latents


def prepare_image_latents(
    vae,
    ref_images: torch.Tensor,
    batch_size: int,
    generator,
    device,
    dtype,
    flux2_pipe_cls=Flux2Pipeline,
):
    ref_images = ref_images.to(device=device, dtype=dtype)
    image_latents_batch = encode_vae_image(
        vae, ref_images, generator, flux2_pipe_cls=flux2_pipe_cls
    )
    image_latents = [
        image_latents_batch[i : i + 1] for i in range(image_latents_batch.shape[0])
    ]

    image_latent_ids = flux2_pipe_cls._prepare_image_ids([image_latents[0]])

    packed_latents = []
    for latent in image_latents:
        packed = flux2_pipe_cls._pack_latents(latent)
        packed = packed.squeeze(0)
        packed_latents.append(packed)
        
    image_latents = torch.stack(packed_latents, dim=0)

    image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
    image_latents = image_latents.to(device=device, dtype=dtype)
    image_latent_ids = image_latent_ids.to(device)

    return image_latents, image_latent_ids


def get_latent_token_layout(
    crop_size: int,
    vae: torch.nn.Module,
    transformer_config: dict | None = None,
    token_dim: int | None = None,
):
    """Return packed VAE token layout for a square crop."""
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_spatial_divisor = vae_scale_factor * 2
    lat_h = int(crop_size) // latent_spatial_divisor
    lat_w = int(crop_size) // latent_spatial_divisor
    num_slots = lat_h * lat_w
    if token_dim is None:
        if transformer_config is None:
            raise ValueError("token_dim or transformer_config is required")
        token_dim = transformer_config.get("in_channels")
        if token_dim is None:
            raise ValueError("transformer config missing in_channels")
    return num_slots, int(token_dim), latent_spatial_divisor, lat_h, lat_w


def encode_vae_packed_tokens(
    vae,
    image: torch.Tensor,
    flux2_pipe_cls=Flux2Pipeline,
    generator=None,
):
    """Encode RGB image to patchified latents and packed VAE tokens."""
    patchified_latents = encode_vae_image(
        vae, image, generator, flux2_pipe_cls=flux2_pipe_cls
    )
    packed_tokens = flux2_pipe_cls._pack_latents(patchified_latents)
    return packed_tokens, patchified_latents


def decode_packed_vae_tokens(
    packed_tokens: torch.Tensor,
    patchified_latents: torch.Tensor,
    vae: torch.nn.Module,
    flux2_pipe_cls=Flux2Pipeline,
):
    """Unpack VAE token sequence and decode to RGB."""
    latent_ids = flux2_pipe_cls._prepare_latent_ids(patchified_latents)
    latent_ids = latent_ids.to(device=packed_tokens.device, dtype=packed_tokens.dtype)
    spatial = flux2_pipe_cls._unpack_latents_with_ids(
        packed_tokens.to(patchified_latents.dtype),
        latent_ids,
    )
    return decode_latents_flux(spatial, vae, flux2_pipe_cls=flux2_pipe_cls)


def decode_latents_flux(
    latents: torch.Tensor, vae: torch.nn.Module, flux2_pipe_cls=Flux2Pipeline
):
    if hasattr(vae, "bn"):
        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = flux2_pipe_cls._unpatchify_latents(latents)
    else:
        shift = getattr(vae.config, "shift_factor", 0.0)
        scale = getattr(vae.config, "scaling_factor", 1.0)
        latents = latents / scale + shift

    image = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
    return image


def get_noisy_model_input_and_timesteps(
    noise_scheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
    discrete_flow_shift: float = 3.1582,
):
    batch_size = latents.shape[0]
    num_timesteps = noise_scheduler.config.num_train_timesteps

    sigmas = torch.sigmoid(torch.randn((batch_size,), device=device) + discrete_flow_shift)
    timesteps = (sigmas * num_timesteps).long()
    sigmas = sigmas.view(-1, 1, 1, 1)
    noisy_model_input = (1 - sigmas) * latents + sigmas * noise
    return noisy_model_input, timesteps, sigmas


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return float(image_seq_len) * m + b


def initialize_transformer_lora(
    transformer,
    rank=4,
    alpha=16,
    dropout=0.0,
    target_patterns=None,
):
    if target_patterns is None:
        target_patterns = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_out",
            "ff.net",
            "context_embedder",
            "x_embedder",
        ]

    matched_modules = []
    for name, module in transformer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(pattern in name for pattern in target_patterns):
            matched_modules.append(name)

    if not matched_modules:
        raise ValueError(
            "No Linear modules matched LoRA target patterns in Flux2 transformer. "
            "Check `train.lora_target_modules`."
        )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=matched_modules,
        lora_dropout=dropout,
        bias="none",
        init_lora_weights="gaussian",
    )
    return get_peft_model(transformer, lora_config)
