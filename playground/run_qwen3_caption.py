#!/usr/bin/env python3
"""
Generate an image caption with a Qwen vision-language model.

Example:
  python playground/run_qwen3_caption.py \
    --image /path/to/image.jpg \
    --model Qwen/Qwen3-VL-8B-Instruct
"""

from __future__ import annotations

import argparse
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def pick_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def build_messages(instruction: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption an image with Qwen3-VL.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN3_VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        help="HuggingFace model id or local path. Env: QWEN3_VL_MODEL",
    )
    parser.add_argument(
        "--instruction",
        default="Describe this image in one concise caption.",
        help="Prompt instruction for captioning.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Computation dtype used while loading the model.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["auto", "cuda", "cpu"],
        help="Execution device.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    dtype = pick_dtype(args.dtype)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    if device == "cpu":
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to("cpu")
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    image = Image.open(args.image).convert("RGB")
    messages = build_messages(args.instruction)
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
    )

    if device == "cpu":
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
    else:
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = generated[:, prompt_len:]
    caption = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    print(caption)


if __name__ == "__main__":
    main()
