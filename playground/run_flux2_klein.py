#!/usr/bin/env python3
"""
Run FLUX.2-klein with diffusers (text-to-image, or text + single reference image).

Examples:
  python playground/run_flux2_klein.py --prompt "a photo of a corgi in space" --out out.png
  python playground/run_flux2_klein.py --prompt "same scene at golden hour" --image ref.png --out out.png
"""

from __future__ import annotations

import argparse
import inspect
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

# add parent directory to path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.LucidFlux.src.flux.lucidflux import (
    extract_vjepa_tokens,
    load_vjepa2_model,
    load_vjepa2_weights,
    vjepa_from_unit_tensor,
)


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _pick_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype '{dtype}'. Use bf16|fp16|fp32.")


def _round_down_to_multiple(value: int, multiple: int = 16) -> int:
    return max(multiple, (value // multiple) * multiple)


def _load_flux2_klein_pipeline(model_id: str, torch_dtype: torch.dtype) -> Any:
    """
    Try a few imports so this script works across diffusers versions.
    """
    # Preferred path for FLUX.2-klein in newer diffusers.
    try:
        from diffusers import Flux2KleinPipeline  # type: ignore

        return Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    except Exception:
        pass

    # Fallback for builds that only expose Flux2Pipeline.
    try:
        from diffusers import Flux2Pipeline  # type: ignore

        return Flux2Pipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    except Exception:
        pass

    # Older/common fallback.
    try:
        from diffusers import FluxPipeline  # type: ignore

        return FluxPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    except Exception:
        pass

    # Alternate import paths used by some builds/forks.
    try:
        from diffusers.pipelines.flux import Flux2KleinPipeline  # type: ignore

        return Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    except Exception:
        pass

    try:
        from diffusers.pipelines.flux import Flux2Pipeline  # type: ignore

        return Flux2Pipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    except Exception:
        pass

    try:
        from diffusers.pipelines.flux import FluxPipeline  # type: ignore

        return FluxPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    except Exception as e:
        raise RuntimeError(
            "Could not import a FLUX pipeline (Flux2KleinPipeline/Flux2Pipeline/FluxPipeline) "
            "from diffusers for FLUX.2-klein. Your diffusers version may not include this "
            "pipeline. Try upgrading diffusers or verify the pipeline class name."
        ) from e


def get_vjepa2_model(
    hub_entry: str,
    dtype: torch.dtype,
    pretrained: bool = False,
    vjepa_checkpoint_path: str | None = None,
) -> Any:
    vjepa = load_vjepa2_model(
        hub_entry,
        "cpu",
        dtype,
        offload=False,
        pretrained=pretrained,
    )
    if vjepa_checkpoint_path and os.path.exists(vjepa_checkpoint_path):
        load_vjepa2_weights(vjepa, vjepa_checkpoint_path)
    vjepa.requires_grad_(False)
    vjepa.eval()
    return vjepa


def main() -> None:
    p = argparse.ArgumentParser(description="Run FLUX.2-klein with diffusers.")
    p.add_argument(
        "--model",
        default=os.environ.get("FLUX2_KLEIN_MODEL", "black-forest-labs/FLUX.2-klein-4B"),
        help="HuggingFace model id or local path. Env: FLUX2_KLEIN_MODEL",
    )
    p.add_argument("--prompt", default="", help="Text prompt for image generation.")
    p.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    p.add_argument(
        "--image",
        default=None,
        help="Path to one reference image (RGB). Passed as `image` to the pipeline for conditioning.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height in pixels (optional; with --image, defaults from the reference after pipeline crop).",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width in pixels (optional; with --image, defaults from the reference after pipeline crop).",
    )
    p.add_argument("--out", default="flux2_klein_out.png", help="Output path (png recommended).")

    p.add_argument("--steps", type=int, default=50, help="Number of denoising steps.")
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale (higher -> stronger prompt adherence).",
    )
    p.add_argument("--seed", type=int, default=12345, help="Random seed (0 means random).")

    p.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"], help="Device.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Computation dtype.")
    p.add_argument(
        "--offload",
        default="model",
        choices=["none", "model", "sequential"],
        help=(
            "Offload mode: none (fastest, highest VRAM), "
            "model (balanced), sequential (lowest VRAM, slowest)."
        ),
    )
    p.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Backward-compatible alias for --offload sequential.",
    )

    p.add_argument("--use-jepa", action="store_true", help="Use V-JEPA 2.1 encoder.")

    args = p.parse_args()

    device = _pick_device(args.device)
    dtype = _pick_dtype(args.dtype)

    ref_pil = None
    if args.image:
        ref_pil = Image.open(args.image).convert("RGB")

    if args.use_jepa:
        if ref_pil is None:
            raise ValueError("--use-jepa requires --image.")
        # V-JEPA 2.1 RoPE mixes float32 position grids with activations; running the
        # encoder in bf16/fp16 leaves Q/K float32 and V reduced precision, which breaks
        # scaled_dot_product_attention. LucidFlux JEPA models use fp32 for V-JEPA only.
        print("Using V-JEPA 2.1 encoder (float32).")
        vjepa_dtype = torch.float32
        vjepa = get_vjepa2_model("vjepa2_1_vit_large_384", vjepa_dtype, True, None).to(device)
        vjepa_input = torch.from_numpy(np.array(ref_pil)).permute(2, 0, 1).to(device) / 255.0
        vjepa_input = vjepa_from_unit_tensor(
            vjepa_input, size=384, device=device, out_dtype=vjepa_dtype
        )

        with torch.no_grad():
            vjepa_tokens = extract_vjepa_tokens(
                vjepa,
                vjepa_input,
                device=device,
                dtype=vjepa_dtype,
            )
        print(f"V-JEPA tokens shape: {vjepa_tokens.shape}")

    pipe = _load_flux2_klein_pipeline(args.model, torch_dtype=dtype)
    offload_mode = "sequential" if args.cpu_offload and args.offload == "none" else args.offload
    if offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload_mode == "model":
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            print(
                "Warning: enable_model_cpu_offload() not available in this diffusers build; "
                "falling back to sequential offload."
            )
            pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    generator = None
    if args.seed and int(args.seed) != 0:
        generator = torch.Generator(device=device).manual_seed(int(args.seed))

    # FLUX.2-klein defaults in diffusers use (9, 18, 27) for better prompt adherence.
    max_sequence_length = 512
    text_encoder_out_layers = (9, 18, 27)
    common_kwargs: dict[str, Any] = dict(
        prompt=args.prompt,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        text_encoder_out_layers=text_encoder_out_layers,
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
        generator=generator,
    )
    if ref_pil is not None:
        common_kwargs["image"] = ref_pil
    if args.height is not None:
        common_kwargs["height"] = int(args.height)
    if args.width is not None:
        common_kwargs["width"] = int(args.width)
    if args.negative_prompt is not None:
        common_kwargs["negative_prompt"] = args.negative_prompt

    # Filter kwargs to only those supported by this diffusers version.
    try:
        sig = inspect.signature(pipe.__call__)
        valid_keys = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in common_kwargs.items() if k in valid_keys}
    except (TypeError, ValueError):
        filtered_kwargs = common_kwargs

    if device == "cuda":
        torch.cuda.empty_cache()

    with torch.inference_mode():
        result = pipe(**filtered_kwargs)

    out_img = getattr(result, "images", result)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_img.save(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
