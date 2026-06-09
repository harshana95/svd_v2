# LucidFlux utilities — input preparation, model loading, and helpers.
# Source: W2GenAI-Lab/LucidFlux (Apache 2.0)

import numpy as np
import sys
import types
import torch
import torch.nn as nn
from einops import rearrange, repeat
from PIL import Image


def _expand_batch(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return repeat(tensor, "1 ... -> b ...", b=batch_size)
    raise ValueError(
        f"{name} batch size {tensor.shape[0]} does not match expected {batch_size}"
    )


def move_modules_to_device(device: torch.device | str, *modules: nn.Module) -> None:
    for module in modules:
        module.to(device)


def prepare_with_embeddings(
    img: torch.Tensor,
    precomputed_txt: torch.Tensor,
    precomputed_vec: torch.Tensor,
) -> dict:
    """Pack latents into sequence form and prepare all inputs for the Flux forward pass."""
    bs, _, h, w = img.shape

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    img_ids = torch.zeros(h // 2, w // 2, 3, device=img.device, dtype=img.dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=img.device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=img.device)[None, :]
    # img_ids = rearrange(img_ids, "h w c -> (h w) c")
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    txt = _expand_batch(precomputed_txt, bs, "precomputed_txt").to(img.device)
    vec = _expand_batch(precomputed_vec, bs, "precomputed_vec").to(img.device)
    # FluxTransformer2DModel expects txt_ids as 2D: [seq_len, 3]
    txt_ids = torch.zeros(txt.shape[1], 3, device=img.device, dtype=img.dtype)

    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "vec": vec,
    }


def load_lucidflux_weights(weights_path: str) -> dict:
    return torch.load(weights_path, map_location="cpu", weights_only=False)


def load_precomputed_embeddings(
    embeddings_path: str, device: torch.device | str
) -> dict:
    embeddings_data = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    result = {
        "txt": embeddings_data["txt"].to(device),
        "vec": embeddings_data["vec"].to(device) if embeddings_data.get("vec") is not None else None,
        "prompt": embeddings_data.get("prompt", "Unknown prompt"),
    }
    if "txt_ids" in embeddings_data and embeddings_data["txt_ids"] is not None:
        result["txt_ids"] = embeddings_data["txt_ids"].to(device)
    return result


def load_siglip_model(
    siglip_ckpt: str,
    device: torch.device | str,
    dtype: torch.dtype,
    offload: bool = False,
) -> nn.Module:
    from transformers import SiglipVisionModel

    siglip_model = SiglipVisionModel.from_pretrained(siglip_ckpt)
    siglip_model.eval()
    target = "cpu" if offload else device
    siglip_model.to(target).to(dtype=dtype)
    return siglip_model


def load_swinir(
    device: torch.device | str,
    checkpoint_path: str,
    offload: bool = False,
) -> nn.Module:
    """Load SwinIR super-resolution prior model."""
    if checkpoint_path is None:
        raise ValueError("SwinIR checkpoint path is not provided")

    # Try loading from the LucidFlux bundled SwinIR first, then fall back to basicsr
    try:
        from .swinir import SwinIR
    except ImportError:
        try:
            # basicsr still imports `torchvision.transforms.functional_tensor`
            # on some releases; torchvision>=0.23 moved rgb_to_grayscale.
            if "torchvision.transforms.functional_tensor" not in sys.modules:
                from torchvision.transforms.functional import rgb_to_grayscale

                shim = types.ModuleType("torchvision.transforms.functional_tensor")
                shim.rgb_to_grayscale = rgb_to_grayscale
                sys.modules["torchvision.transforms.functional_tensor"] = shim
            from basicsr.archs.swinir_arch import SwinIR
        except ImportError:
            raise ImportError(
                "SwinIR not found. Install basicsr or provide src/flux/swinir.py"
            )

    swinir = SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        sf=8,
        img_range=1.0,
        upsampler="nearest+conv",
        resi_connection="1conv",
        unshuffle=True,
        unshuffle_scale=8,
    )
    ckpt_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt_obj.get("state_dict", ckpt_obj)
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    swinir.load_state_dict(new_state, strict=False)
    swinir.eval()
    for p in swinir.parameters():
        p.requires_grad_(False)
    target = "cpu" if offload else device
    return swinir.to(target)


def siglip_from_unit_tensor(
    x: torch.Tensor,
    size: tuple[int, int] = (512, 512),
    device: torch.device | str | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    strict_range: bool = True,
) -> torch.Tensor:
    """Convert [0,1] tensor to SigLIP-compatible [-1,1] pixel values with PIL resize."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    assert x.ndim == 4 and x.shape[1] == 3, "Expected shape (B,3,H,W)"

    if strict_range:
        minv, maxv = float(x.min()), float(x.max())
        if minv < -1e-3 or maxv > 1 + 1e-3:
            raise ValueError(
                f"Input must be in [0,1], got range [{minv:.4f},{maxv:.4f}]"
            )

    batch_size = x.shape[0]
    processed_batch = []

    for i in range(batch_size):
        img_tensor = x[i].permute(1, 2, 0)  # CHW -> HWC
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode="RGB")

        if pil_img.size != size:
            pil_img = pil_img.resize(size, resample=2)  # BILINEAR

        img_array = np.array(pil_img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
        img_tensor = (img_tensor - 0.5) / 0.5  # normalize to [-1, 1]
        processed_batch.append(img_tensor)

    result = torch.stack(processed_batch, dim=0)
    if device is not None:
        result = result.to(device=device, dtype=out_dtype)
    else:
        result = result.to(dtype=out_dtype)
    return result


def preprocess_lq_image(
    image_path: str, width: int = 512, height: int = 512
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height))
    return image
