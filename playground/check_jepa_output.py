#!/usr/bin/env python3
"""
Quick utility to inspect JEPA outputs for a single image.

Examples:
  # HuggingFace I-JEPA (recommended default)
  python playground/check_jepa_output.py \
      --image /path/to/input.png \
      --model facebook/ijepa_vith14_22k \
      --backend hf \
      --save-dir playground/jepa_debug

  # Torch hub V-JEPA2.1
  python playground/check_jepa_output.py \
      --image /path/to/input.png \
      --backend hub \
      --hub-entry vjepa2_1_vit_large_384 \
      --save-dir playground/jepa_debug
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel
# from transformers.models.ijepa import IJepaModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run I-JEPA on one image.")
    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/ijepa_vith14_22k",
        help="HF model id or local JEPA model path.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="hf",
        choices=["hf", "hub"],
        help="Model backend: HuggingFace (hf) or torch hub V-JEPA (hub).",
    )
    parser.add_argument(
        "--hub-entry",
        type=str,
        default="vjepa2_1_vit_large_384",
        help="torch.hub entrypoint used when --backend hub.",
    )
    parser.add_argument(
        "--hub-checkpoint",
        type=str,
        default="",
        help="Optional local .pt checkpoint for hub backend (avoids hub download).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize image to this square size before JEPA.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Optional output folder to save numpy dumps and token visualization.",
    )
    parser.add_argument(
        "--channels-per-image",
        type=int,
        default=16,
        help="Kept for compatibility; not used in GIF export.",
    )
    parser.add_argument(
        "--gif-duration-ms",
        type=int,
        default=80,
        help="Frame duration for channel GIF in milliseconds.",
    )
    return parser.parse_args()


def make_jepa_preprocess(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def infer_token_grid(n_tokens: int, image_h: int, image_w: int) -> tuple[int, int]:
    """
    Find (grid_h, grid_w) such that grid_h * grid_w = n_tokens and
    grid_w / grid_h is closest to image_w / image_h.
    """
    target_ratio = float(image_w) / float(max(image_h, 1))
    best_h, best_w = 1, n_tokens
    best_score = abs((best_w / best_h) - target_ratio)

    for h in range(1, int(np.sqrt(n_tokens)) + 1):
        if n_tokens % h != 0:
            continue
        w = n_tokens // h
        score_hw = abs((w / h) - target_ratio)
        if score_hw < best_score:
            best_h, best_w = h, w
            best_score = score_hw

        score_wh = abs((h / w) - target_ratio)
        if score_wh < best_score:
            best_h, best_w = w, h
            best_score = score_wh

    return best_h, best_w


def make_token_rgb(tokens: torch.Tensor, grid_h: int, grid_w: int) -> np.ndarray:
    """
    Convert [N, D] token matrix into a pseudo RGB image using PCA.
    """
    centered = tokens - tokens.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=3, center=False)
    rgb = centered @ v[:, :3]  # [N, 3]

    rgb = rgb - rgb.min(dim=0, keepdim=True).values
    denom = rgb.max(dim=0, keepdim=True).values.clamp_min(1e-8)
    rgb = rgb / denom

    return rgb.reshape(grid_h, grid_w, 3).cpu().numpy()


def save_channel_gif(
    tokens: torch.Tensor,
    grid_h: int,
    grid_w: int,
    save_path: Path,
    duration_ms: int,
) -> None:
    """
    Save one GIF frame per latent channel (lossless indexed-color GIF).
    """
    maps = tokens.reshape(grid_h, grid_w, tokens.shape[1]).permute(2, 0, 1).contiguous()  # [D, H, W]
    frames: list[Image.Image] = []
    for idx in range(maps.shape[0]):
        ch = maps[idx]
        ch = ch - ch.min()
        ch = ch / ch.max().clamp_min(1e-8)
        ch_u8 = (ch * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        frames.append(Image.fromarray(ch_u8, mode="L"))

    if not frames:
        return

    frames[0].save(
        save_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=max(1, duration_ms),
        optimize=False,
        disposal=2,
    )


def _clean_backbone_key(state_dict: dict) -> dict:
    cleaned = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = val
    return cleaned


def _load_hub_encoder_weights(encoder: torch.nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint is not a state-dict-like dict.")

    candidate_keys = ["ema_encoder", "target_encoder", "encoder", "model", "state_dict"]
    src = None
    for key in candidate_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            src = ckpt[key]
            break
    if src is None:
        src = ckpt

    state = _clean_backbone_key(src)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(
        f"Loaded hub encoder checkpoint: {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )




def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device(args.device)
    image = Image.open(image_path).convert("RGB")
    if args.backend == "hf":
        preprocess = make_jepa_preprocess(args.image_size)
        model = AutoModel.from_pretrained(args.model).to(device).eval()
        pixel_values = preprocess(image).unsqueeze(0).to(device)  # [B, C, H, W]
    else:
        try:
            preprocess = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor")
            # Force pretrained=False to avoid localhost mirror download in this environment.
            model_obj = torch.hub.load(
                "facebookresearch/vjepa2",
                args.hub_entry,
                pretrained=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize V-JEPA from torch.hub. "
                "Use --backend hf, or ensure torch.hub repo is available."
            ) from exc
        if isinstance(model_obj, (tuple, list)):
            model = model_obj[0]
        else:
            model = model_obj
        model = model.to(device).eval()

        if args.hub_checkpoint:
            _load_hub_encoder_weights(model, args.hub_checkpoint)
        else:
            print(
                "Warning: hub backend is running without pretrained weights. "
                "Pass --hub-checkpoint /path/to/*.pt for meaningful features."
            )

        # V-JEPA torch.hub preprocess expects a video clip (list of frames).
        pre_out = preprocess([np.array(image)])
        if isinstance(pre_out, list):
            pre_out = pre_out[0]
        if not isinstance(pre_out, torch.Tensor):
            pre_out = torch.as_tensor(pre_out)

        # Normalize to [B, C, T, H, W] expected by V-JEPA.
        if pre_out.ndim == 3:  # [C, H, W]
            pre_out = pre_out.unsqueeze(1)  # [C, 1, H, W]
        if pre_out.ndim == 4:  # [C, T, H, W] or [T, C, H, W]
            if pre_out.shape[0] in (1, 3):
                pre_out = pre_out.unsqueeze(0)  # [1, C, T, H, W]
            else:
                pre_out = pre_out.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
        if pre_out.ndim != 5:
            raise ValueError(f"Unexpected preprocessed tensor shape: {tuple(pre_out.shape)}")

        pixel_values = pre_out.to(device)

    
    with torch.no_grad():
        # V-JEPA hub models may return Tensor / dict / tuple instead of HF-style output.
        if args.backend == "hf":
            out = model(pixel_values=pixel_values)
        else:
            out = model(pixel_values)
        if isinstance(out, torch.Tensor):
            feat = out
        elif isinstance(out, dict):
            feat = out.get("last_hidden_state", out.get("x", out.get("features")))
            if feat is None:
                raise ValueError(f"Unsupported dict output keys: {list(out.keys())}")
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            feat = out[0]
        elif hasattr(out, "last_hidden_state"):
            feat = out.last_hidden_state
        else:
            raise TypeError(f"Unsupported model output type: {type(out)}")

        # Normalize to [N, D] token matrix.
        if feat.ndim == 3:
            tokens = feat[0]  # [B, N, D] -> [N, D]
        elif feat.ndim == 2:
            tokens = feat  # already [N, D]
        else:
            raise ValueError(f"Expected 2D/3D features, got shape {tuple(feat.shape)}")

    print(f"Backend: {args.backend}")
    print(f"Model: {args.model if args.backend == 'hf' else args.hub_entry}")
    print(f"Input image: {image_path}")
    print(f"Device: {device}")
    print(f"Token tensor shape: {tuple(tokens.shape)}")
    print(f"Token value range: [{tokens.min().item():.6f}, {tokens.max().item():.6f}]")
    print(f"Token mean/std: {tokens.mean().item():.6f} / {tokens.std().item():.6f}")
    grid_h, grid_w = infer_token_grid(tokens.shape[0], image.height, image.width)
    print(f"Inferred token grid (H, W): ({grid_h}, {grid_w})")

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        tokens_cpu = tokens.float().cpu()
        np.save(save_dir / "jepa_tokens.npy", tokens_cpu.numpy())
        np.save(save_dir / "jepa_pixel_values.npy", pixel_values.float().cpu().numpy())
        
        print(tokens_cpu.shape, tokens_cpu.dtype, tokens_cpu.min(), tokens_cpu.max())
        tokens_cpu_norm = (tokens_cpu - tokens_cpu.min()) / (
            tokens_cpu.max() - tokens_cpu.min() + 1e-8
        )
        tokens_cpu_u8 = (
            (tokens_cpu_norm.reshape(grid_h, grid_w, tokens_cpu.shape[1]).mean(dim=2) * 255.0)
            .clamp(0, 255)
            .to(torch.uint8)
            .numpy()
        )
        Image.fromarray(tokens_cpu_u8, mode="L").save(save_dir / "jepa_tokens_norm.png")

        token_rgb = make_token_rgb(tokens_cpu, grid_h, grid_w)
        token_rgb_u8 = (token_rgb * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(token_rgb_u8).save(save_dir / "jepa_tokens_pca_rgb.png")
        save_channel_gif(
            tokens_cpu,
            grid_h,
            grid_w,
            save_dir / "jepa_channels.gif",
            args.gif_duration_ms,
        )
        print(f"Saved outputs to: {save_dir}")


if __name__ == "__main__":
    main()
