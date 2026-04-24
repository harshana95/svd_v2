#!/usr/bin/env python3
"""
Run FLUX Kontext (image-to-image editing) with diffusers.

Example:
  python playground/run_flux_kontext.py --image path/to/input.png --prompt "remove the people, keep the background" --out out.png
  /depot/chan129/data/GoPro/test/GOPR0384_11_00/blur/000001.png
"""

from __future__ import annotations

import argparse
import os
from typing import Any
import inspect

import torch
from PIL import Image


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


def _load_kontext_pipeline(model_id: str, torch_dtype: torch.dtype) -> Any:
    """
    Diffusers has renamed/introduced Flux Kontext pipelines across versions.
    We try a few imports so this script stays usable in research envs.
    """
    try:
        from diffusers import FluxKontextPipeline  # type: ignore

        return FluxKontextPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    except Exception:
        pass

    # Fallback: some versions expose it under diffusers.pipelines.flux
    try:
        from diffusers.pipelines.flux import FluxKontextPipeline  # type: ignore

        return FluxKontextPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    except Exception as e:
        raise RuntimeError(
            "Could not import FluxKontextPipeline from diffusers. "
            "Your diffusers version may not include Flux Kontext. "
            "Try upgrading diffusers or verify the pipeline class name."
        ) from e


def main() -> None:
    p = argparse.ArgumentParser(description="Run FLUX Kontext with diffusers.")
    p.add_argument(
        "--model",
        default=os.environ.get("FLUX_KONTEXT_MODEL", "black-forest-labs/FLUX.1-Kontext-dev"),
        help="HuggingFace model id or local path. Env: FLUX_KONTEXT_MODEL",
    )
    p.add_argument("--image", required=True, help="Input image path (png/jpg/webp).")
    p.add_argument("--prompt", required=True, help="Edit instruction / prompt.")
    p.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    p.add_argument("--out", default="kontext_out.png", help="Output path (png recommended).")

    p.add_argument("--steps", type=int, default=28, help="Number of denoising steps.")
    p.add_argument("--guidance-scale", type=float, default=3.5, help="Classifier-free guidance scale.")
    p.add_argument("--strength", type=float, default=0.8, help="Edit strength (0-1).")
    p.add_argument("--seed", type=int, default=0, help="Random seed (0 means random).")

    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Computation dtype.")
    p.add_argument("--cpu-offload", action="store_true", help="Enable sequential CPU offload (saves VRAM).")

    args = p.parse_args()

    device = _pick_device(args.device)
    dtype = _pick_dtype(args.dtype)

    img = Image.open(args.image).convert("RGB")

    pipe = _load_kontext_pipeline(args.model, torch_dtype=dtype)
    if args.cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    generator = None
    if args.seed and int(args.seed) != 0:
        generator = torch.Generator(device=device).manual_seed(int(args.seed))

    # Different diffusers builds may expose slightly different kwarg names.
    common_kwargs: dict[str, Any] = dict(
        prompt=args.prompt,
        image=img,
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
        strength=float(args.strength),
        generator=generator,
    )
    if args.negative_prompt is not None:
        common_kwargs["negative_prompt"] = args.negative_prompt

    # Filter kwargs to only those supported by this diffusers version.
    try:
        sig = inspect.signature(pipe.__call__)
        valid_keys = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in common_kwargs.items() if k in valid_keys}
    except (TypeError, ValueError):
        # Fallback: if we can't inspect the signature, just use the original kwargs.
        filtered_kwargs = common_kwargs

    with torch.inference_mode():
        result = pipe(**filtered_kwargs)

    out_img = getattr(result, "images", result)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_img.save(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

