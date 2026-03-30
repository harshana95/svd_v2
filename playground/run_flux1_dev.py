#!/usr/bin/env python3
"""
Run FLUX.1-dev (text-to-image) with diffusers.

Example:
  python playground/run_flux1_dev.py --prompt "a photo of a corgi in space" --out out.png
"""

from __future__ import annotations

import argparse
import os
from typing import Any
import inspect

import torch


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


def _load_flux1_dev_pipeline(model_id: str, torch_dtype: torch.dtype) -> Any:
    """
    Try a few imports so this script works across diffusers versions.
    """
    # Newer diffusers exposes it directly.
    try:
        from diffusers import FluxPipeline  # type: ignore

        return FluxPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    except Exception:
        pass

    # Some forks/versions may expose it under diffusers.pipelines.flux
    try:
        from diffusers.pipelines.flux import FluxPipeline  # type: ignore

        return FluxPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    except Exception as e:
        raise RuntimeError(
            "Could not import FluxPipeline for FLUX.1-dev from diffusers. "
            "Your diffusers version may not include this pipeline. "
            "Try upgrading diffusers or verify the pipeline class name."
        ) from e


def main() -> None:
    p = argparse.ArgumentParser(description="Run FLUX.1-dev with diffusers.")
    p.add_argument(
        "--model",
        default=os.environ.get("FLUX1_DEV_MODEL", "black-forest-labs/FLUX.1-dev"),
        help="HuggingFace model id or local path. Env: FLUX1_DEV_MODEL",
    )
    p.add_argument("--prompt", required=True, help="Text prompt for image generation.")
    p.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    p.add_argument("--out", default="flux1_dev_out.png", help="Output path (png recommended).")

    p.add_argument("--steps", type=int, default=50, help="Number of denoising steps.")
    p.add_argument("--guidance-scale", type=float, default=3.5, help="Classifier-free guidance scale.")
    p.add_argument("--seed", type=int, default=0, help="Random seed (0 means random).")

    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Computation dtype.")
    p.add_argument("--cpu-offload", action="store_true", help="Enable sequential CPU offload (saves VRAM).")

    args = p.parse_args()

    device = _pick_device(args.device)
    dtype = _pick_dtype(args.dtype)

    pipe = _load_flux1_dev_pipeline(args.model, torch_dtype=dtype)
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
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
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
        filtered_kwargs = common_kwargs

    with torch.inference_mode():
        result = pipe(**filtered_kwargs)

    out_img = getattr(result, "images", result)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_img.save(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

