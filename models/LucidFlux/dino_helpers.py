import torch
import torch.nn as nn
from transformers import AutoModel


def get_dino_patch_size(model_name: str, fallback_patch_size: int = 14) -> int:
    name = model_name.lower()
    for token in name.split("-"):
        if token.startswith("vit") and token[-2:].isdigit():
            return int(token[-2:])
    return fallback_patch_size


def get_dino_num_patches(model_name: str, image_size: int, fallback_patch_size: int = 14) -> int:
    patch_size = get_dino_patch_size(model_name, fallback_patch_size=fallback_patch_size)
    grid = image_size // patch_size
    return grid * grid


def get_dino_feature_dim(model_name: str, fallback_dim: int = 1024) -> int:
    name = model_name.lower()
    # HuggingFace repo ids: facebook/dinov2-{small,base,large,giant}
    if "dinov2-small" in name or "vits14" in name or "vits16" in name:
        return 384
    if "dinov2-base" in name or "vitb14" in name or "vitb16" in name:
        return 768
    if "dinov2-large" in name or "vitl14" in name or "vitl16" in name:
        return 1024
    if "dinov2-giant" in name or "vitg14" in name or "vitg16" in name:
        return 1536
    return fallback_dim


def load_dinov2_model(
    model_name: str,
    device: torch.device | str,
    dtype: torch.dtype,
) -> nn.Module:
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    return model


def extract_dino_tokens(
    model: nn.Module,
    pixel_values: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run a DINOv2 encoder and return patch tokens [B, N, D] (CLS skipped)."""
    if pixel_values.ndim != 4:
        raise ValueError(
            f"Expected DINO pixel_values to be 4D (B,C,H,W), got {pixel_values.ndim}D"
        )

    outputs = model(pixel_values.to(device=device, dtype=dtype))
    if hasattr(outputs, "last_hidden_state"):
        feats = outputs.last_hidden_state
    elif isinstance(outputs, dict):
        feats = outputs.get("last_hidden_state")
        if feats is None:
            raise ValueError(
                f"Unsupported DINO dict output keys: {list(outputs.keys())}"
            )
    elif isinstance(outputs, torch.Tensor):
        feats = outputs
    else:
        raise TypeError(f"Unsupported DINO output type: {type(outputs)}")

    if feats.ndim != 3:
        raise ValueError(f"Unsupported DINO feature shape: {tuple(feats.shape)}")

    # DINOv2 prepends a CLS token at index 0.
    return feats[:, 1:, :]
