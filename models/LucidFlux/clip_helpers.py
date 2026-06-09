import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPVisionModel

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def get_clip_patch_size(model_name: str, fallback_patch_size: int = 14) -> int:
    name = model_name.lower()
    # "patch32" → 32, "patch16" → 16, "patch14" → 14
    for token in name.replace("-", "_").split("_"):
        if token.startswith("patch") and token[5:].isdigit():
            return int(token[5:])
    # fallback: parse the trailing digits after "vit-b/32" style slash
    for part in name.split("/"):
        if part.isdigit():
            return int(part)
    return fallback_patch_size


def get_clip_num_patches(model_name: str, image_size: int, fallback_patch_size: int = 14) -> int:
    patch_size = get_clip_patch_size(model_name, fallback_patch_size=fallback_patch_size)
    grid = image_size // patch_size
    return grid * grid


def get_clip_feature_dim(model_name: str, fallback_dim: int = 1024) -> int:
    name = model_name.lower()
    if "vit-b/32" in name or "vit_b_32" in name or "vitb32" in name or "patch32" in name:
        return 768
    if "vit-b/16" in name or "vit_b_16" in name or "vitb16" in name or "patch16" in name:
        return 768
    if "vit-l/14" in name or "vit_l_14" in name or "vitl14" in name or "large" in name:
        return 1024
    if "vit-h" in name or "huge" in name:
        return 1280
    return fallback_dim


def load_clip_model(
    model_name: str,
    device: torch.device | str,
    dtype: torch.dtype,
) -> nn.Module:
    model = CLIPVisionModel.from_pretrained(model_name)
    model.eval()
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model


def clip_from_unit_tensor(
    x: torch.Tensor,
    size: int = 224,
    device: torch.device | str | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert [0,1] BCHW tensor to CLIP-normalized pixels."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    assert x.ndim == 4 and x.shape[1] == 3, "Expected shape (B,3,H,W)"

    resize = transforms.Resize(
        (size, size),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)
    x = resize(x.float())
    x = (x - mean.to(x.device)) / std.to(x.device)

    if device is not None:
        x = x.to(device=device, dtype=out_dtype)
    else:
        x = x.to(dtype=out_dtype)
    return x


def extract_clip_tokens(
    model: nn.Module,
    pixel_values: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run CLIPVisionModel and return patch tokens [B, N, D] (CLS stripped)."""
    if pixel_values.ndim != 4:
        raise ValueError(
            f"Expected CLIP pixel_values to be 4D (B,C,H,W), got {pixel_values.ndim}D"
        )

    outputs = model(pixel_values.to(device=device, dtype=dtype))
    if hasattr(outputs, "last_hidden_state"):
        feats = outputs.last_hidden_state
    elif isinstance(outputs, dict):
        feats = outputs.get("last_hidden_state")
        if feats is None:
            raise ValueError(f"Unsupported CLIP dict output keys: {list(outputs.keys())}")
    elif isinstance(outputs, torch.Tensor):
        feats = outputs
    else:
        raise TypeError(f"Unsupported CLIP output type: {type(outputs)}")

    if feats.ndim != 3:
        raise ValueError(f"Unsupported CLIP feature shape: {tuple(feats.shape)}")

    # CLIP ViT prepends a CLS token at index 0.
    return feats[:, 1:, :]
