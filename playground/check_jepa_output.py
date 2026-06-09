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
import torch.nn.functional as F
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
        default="hub",
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
        default=512,
        help="Resize shortest side to this size before JEPA.",
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
    parser.add_argument(
        "--interpolate-pos-encoding",
        action="store_true",
        help=(
            "Enable positional embedding interpolation for HF I-JEPA forward pass. "
            "Needed for non-native input resolutions (e.g., 384, 512)."
        ),
    )
    parser.add_argument(
        "--decode-embeddings",
        action="store_true",
        help="Optimize pixels to match extracted JEPA embeddings (feature inversion).",
    )
    parser.add_argument("--decode-steps", type=int, default=1000, help="Inversion optimization steps.")
    parser.add_argument("--decode-lr", type=float, default=0.05, help="Inversion learning rate.")
    parser.add_argument(
        "--decode-init",
        type=str,
        default="noise",
        choices=["input", "noise"],
        help="Initialization for inversion: input image or random noise.",
    )
    parser.add_argument(
        "--decode-cosine-weight",
        type=float,
        default=1.0,
        help="Weight for cosine embedding loss during inversion.",
    )
    parser.add_argument(
        "--decode-mse-weight",
        type=float,
        default=0.25,
        help="Weight for MSE embedding loss during inversion.",
    )
    parser.add_argument(
        "--decode-tv-weight",
        type=float,
        default=1e-4,
        help="Weight for total variation regularization.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=0,
        help=(
            "If > 0, keep only this many principal components of token embeddings "
            "(per-image PCA across tokens) before decoding and for token RGB viz. "
            "0 = use full embedding dimension (no PCA)."
        ),
    )
    return parser.parse_args()


def make_jepa_preprocess(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                image_size,
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


def make_token_rgb(
    tokens: torch.Tensor,
    grid_h: int,
    grid_w: int,
    n_components: int = 3,
) -> np.ndarray:
    """
    Convert [N, D] token matrix into a pseudo RGB image using PCA.

    Computes PCA with up to ``n_components`` directions (clamped by N and D).
    The image uses the first three score dimensions as R, G, B; if ``n_components`` < 3,
    remaining channels are zero-padded before per-channel min–max normalization.
    """
    n_tokens, dim = tokens.shape[0], tokens.shape[1]
    n_comp = max(1, min(int(n_components), dim, max(1, n_tokens - 1)))
    q = max(n_comp, min(3, dim, max(1, n_tokens - 1)))

    centered = tokens - tokens.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=q, center=False)
    rgb = centered @ v[:, : min(3, n_comp)]  # [N, min(3, n_comp)]

    if rgb.shape[1] < 3:
        pad = rgb.new_zeros(rgb.shape[0], 3 - rgb.shape[1])
        rgb = torch.cat([rgb, pad], dim=1)

    rgb = rgb - rgb.min(dim=0, keepdim=True).values
    denom = rgb.max(dim=0, keepdim=True).values.clamp_min(1e-8)
    rgb = rgb / denom

    return rgb.reshape(grid_h, grid_w, 3).cpu().numpy()


def _tokens_pca_reconstruct(tokens: torch.Tensor, n_components: int) -> torch.Tensor:
    """
    Per-image PCA across tokens: reconstruct each row using only the top ``n_components``
    principal directions (same subspace as ``make_token_rgb`` when q matches).
    """
    n_tokens, dim = tokens.shape[0], tokens.shape[1]
    n_comp = max(1, min(int(n_components), dim, max(1, n_tokens - 1)))
    mean = tokens.mean(dim=0, keepdim=True)
    centered = tokens - mean
    _, _, v = torch.pca_lowrank(centered, q=n_comp, center=False)
    scores = centered @ v  # [N, n_comp]
    return scores @ v.T + mean


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


def _extract_tokens_from_output(out) -> torch.Tensor:
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

    if feat.ndim == 3:
        return feat[0]  # [B, N, D] -> [N, D]
    if feat.ndim == 2:
        return feat  # already [N, D]
    raise ValueError(f"Expected 2D/3D features, got shape {tuple(feat.shape)}")


def _forward_tokens(model: torch.nn.Module, pixel_values: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.backend == "hf":
        out = model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=(args.interpolate_pos_encoding or args.image_size != 224),
        )
    else:
        out = model(pixel_values)
    return _extract_tokens_from_output(out)


def _total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    # avoid dividing by 0 by adding a small epsilon before mean
    if x.shape[2] == 1:
        x = x[:, :, 0]
    if x.ndim == 4:  # [B, C, H, W]
        tv_h = ((x[:, :, 1:, :] - x[:, :, :-1, :]).abs() + 1e-8).mean()
        tv_w = ((x[:, :, :, 1:] - x[:, :, :, :-1]).abs() + 1e-8).mean()
        return tv_h + tv_w
    if x.ndim == 5:  # [B, C, T, H, W]
        tv_t = ((x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).abs() + 1e-8).mean()
        tv_h = ((x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs() + 1e-8).mean()
        tv_w = ((x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs() + 1e-8).mean()
        return tv_t + tv_h + tv_w
    return x.new_tensor(0.0)


def _denorm_for_save(pixel_values: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device).view(1, 3, 1, 1)

    if pixel_values.ndim == 5:
        img = pixel_values[:, :, 0]  # first frame
    else:
        img = pixel_values
    img = (img * std + mean).clamp(0, 1)
    img = (img[0].permute(1, 2, 0) * 255.0).to(torch.uint8).cpu().numpy()
    return img


def _invert_embeddings(
    model: torch.nn.Module,
    target_tokens: torch.Tensor,
    pixel_values: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict]:
    if args.decode_init == "noise":
        x_opt = torch.randn_like(pixel_values)
    else:
        x_opt = pixel_values.clone()
    x_opt = x_opt.detach().requires_grad_(True)

    optimizer = torch.optim.Adam([x_opt], lr=args.decode_lr)
    target = target_tokens.detach()
    if getattr(args, "pca_components", 0) and args.pca_components > 0:
        print(f"Using PCA with {args.pca_components} components")
        target = _tokens_pca_reconstruct(target, args.pca_components)
    else:
        print("Using full embedding dimension")
    target_norm = F.normalize(target, dim=-1)

    stats = {"final_loss": None, "final_cosine": None, "final_mse": None, "final_tv": None}
    for step in range(args.decode_steps):
        optimizer.zero_grad(set_to_none=True)
        pred = _forward_tokens(model, x_opt, args)
        pred_norm = F.normalize(pred, dim=-1)

        loss_cos = 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()
        loss_mse = F.mse_loss(pred, target)
        loss_tv = _total_variation_loss(x_opt)
        loss = (
            args.decode_cosine_weight * loss_cos
            + args.decode_mse_weight * loss_mse
            + args.decode_tv_weight * loss_tv
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_opt.clamp_(-3.0, 3.0)

        if (step + 1) % 25 == 0 or step == 0 or step + 1 == args.decode_steps:
            print(
                f"[decode] step {step + 1:4d}/{args.decode_steps} "
                f"loss={loss.item():.6f} cos={loss_cos.item():.6f} "
                f"mse={loss_mse.item():.6f} tv={loss_tv.item():.6f}"
            )

    stats["final_loss"] = float(loss.item())
    stats["final_cosine"] = float(loss_cos.item())
    stats["final_mse"] = float(loss_mse.item())
    stats["final_tv"] = float(loss_tv.item())
    return x_opt.detach(), stats




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
            # preprocess = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor")
            preprocess = make_jepa_preprocess(args.image_size)
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
        pre_out = preprocess(image)
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
        tokens = _forward_tokens(model, pixel_values, args)

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

        # save original image
        image.save(save_dir / "original_image.png")

        # save normalized tokens
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
    
        # save PCA top 3 components as RGB
        token_rgb = make_token_rgb(tokens_cpu, grid_h, grid_w, n_components=3)
        token_rgb_u8 = (token_rgb * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(token_rgb_u8).save(save_dir / "jepa_tokens_pca_rgb.png")

        # save channels as GIF
        save_channel_gif(
            tokens_cpu,
            grid_h,
            grid_w,
            save_dir / "jepa_channels.gif",
            args.gif_duration_ms,
        )

        if args.decode_embeddings:
            model.eval()
            decoded_values, decode_stats = _invert_embeddings(model, tokens, pixel_values, args)
            decoded_img = _denorm_for_save(decoded_values)
            Image.fromarray(decoded_img).save(save_dir / "jepa_decoded_from_embeddings.png")
            np.save(save_dir / "jepa_decoded_pixel_values.npy", decoded_values.float().cpu().numpy())
            print(
                "Decoded image saved to "
                f"{save_dir / 'jepa_decoded_from_embeddings.png'} "
                f"(final loss={decode_stats['final_loss']:.6f})"
            )
        print(f"Saved outputs to: {save_dir}")


if __name__ == "__main__":
    main()
