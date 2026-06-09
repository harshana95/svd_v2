import torch
import torch.nn as nn
from einops import rearrange
from torchvision import transforms


def get_vjepa_patch_size(hub_entry: str, fallback_patch_size: int = 16) -> int:
    if "patch" in hub_entry:
        for part in hub_entry.split("_"):
            if part.isdigit():
                return int(part)
    return fallback_patch_size


def get_vjepa_num_patches(hub_entry: str, image_size: int, fallback_patch_size: int = 16) -> int:
    patch_size = get_vjepa_patch_size(hub_entry, fallback_patch_size=fallback_patch_size)
    grid = image_size // patch_size
    return grid * grid


def get_vjepa_feature_dim(hub_entry: str, fallback_dim: int = 1024) -> int:
    if "vit_base" in hub_entry:
        return 768
    if "vit_large" in hub_entry:
        return 1024
    if "vit_giant" in hub_entry and "gigantic" not in hub_entry:
        return 1280
    if "vit_gigantic" in hub_entry:
        return 1664
    return fallback_dim


def load_vjepa2_model(
    hub_entry: str,
    device: torch.device | str,
    dtype: torch.dtype,
    offload: bool = False,
    pretrained: bool = False,
    **hub_kwargs,
) -> nn.Module:
    model_obj = torch.hub.load(
        "facebookresearch/vjepa2",
        hub_entry,
        pretrained=pretrained,
        **hub_kwargs,
    )
    vjepa_model = model_obj[0] if isinstance(model_obj, (tuple, list)) else model_obj
    vjepa_model.eval()
    target = "cpu" if offload else device
    vjepa_model.to(target).to(dtype=dtype)
    return vjepa_model


def load_vjepa2_weights(model: nn.Module, checkpoint_path: str) -> None:
    param = next(model.parameters(), None)
    target_device = param.device if param is not None else None
    target_dtype = param.dtype if param is not None else None

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("V-JEPA checkpoint must be a dict-like object.")

    source = None
    for key in ("ema_encoder", "target_encoder", "encoder", "model", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            source = ckpt[key]
            break
    if source is None:
        source = ckpt

    cleaned = {}
    for key, val in source.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = val
    model.load_state_dict(cleaned, strict=False)

    if target_device is not None or target_dtype is not None:
        model.to(device=target_device, dtype=target_dtype)


def vjepa_from_unit_tensor(
    x: torch.Tensor,
    size: int = 384,
    device: torch.device | str | None = None,
    out_dtype: torch.dtype = torch.float32,
    strict_range: bool = True,
) -> torch.Tensor:
    """Convert [0,1] tensor to V-JEPA-compatible ImageNet-normalized pixels."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    assert x.ndim == 4 and x.shape[1] == 3, "Expected shape (B,3,H,W)"

    if strict_range:
        minv, maxv = float(x.min()), float(x.max())
        if minv < -1e-3 or maxv > 1 + 1e-3:
            raise ValueError(
                f"Input must be in [0,1], got range [{minv:.4f},{maxv:.4f}]"
            )

    resize = transforms.Resize(
        (size, size),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    x = resize(x.float())
    x = (x - mean.to(x.device)) / std.to(x.device)

    if device is not None:
        x = x.to(device=device, dtype=out_dtype)
    else:
        x = x.to(dtype=out_dtype)
    return x


def extract_vjepa_tokens(
    model: nn.Module,
    pixel_values: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run a V-JEPA encoder and flatten its output to [B, N, D]."""
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(2)
    if pixel_values.ndim != 5:
        raise ValueError(
            f"Expected V-JEPA pixel_values to be 4D or 5D, got {pixel_values.ndim}D"
        )

    outputs = model(pixel_values.to(device=device, dtype=dtype))
    if isinstance(outputs, torch.Tensor):
        feats = outputs
    elif isinstance(outputs, dict):
        feats = outputs.get("last_hidden_state", outputs.get("x", outputs.get("features")))
        if feats is None:
            raise ValueError(f"Unsupported V-JEPA dict output keys: {list(outputs.keys())}")
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        feats = outputs[0]
    elif hasattr(outputs, "last_hidden_state"):
        feats = outputs.last_hidden_state
    else:
        raise TypeError(f"Unsupported V-JEPA output type: {type(outputs)}")

    if feats.ndim == 5:
        feats = rearrange(feats, "b t h w d -> b (t h w) d")
    elif feats.ndim == 4:
        feats = rearrange(feats, "b t n d -> b (t n) d")
    elif feats.ndim != 3:
        raise ValueError(f"Unsupported V-JEPA feature shape: {tuple(feats.shape)}")
    return feats
