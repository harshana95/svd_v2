import argparse
from pathlib import Path

import torch
from diffusers import Flux2Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and save Flux 2 text embeddings."
    )
    parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.2-dev",
        help="Flux 2 pretrained model name or local path.",
    )
    parser.add_argument(
        "--output",
        default="null_prompt_embeddings_flux2.pt",
        help="Output .pt path for saved embeddings.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Prompt to encode. Defaults to the empty prompt.",
    )   
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run text encoding on.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16" if torch.cuda.is_available() else "fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Torch dtype for the text encoders.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="Optional max sequence length override for prompt encoding.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping[name]


def encode_prompt(pipe: Flux2Pipeline, prompt: str, max_sequence_length: int | None):
    kwargs = {
        "prompt": [prompt],
        "num_images_per_prompt": 1,
        "do_classifier_free_guidance": False,
    }
    if max_sequence_length is not None:
        kwargs["max_sequence_length"] = max_sequence_length

    with torch.no_grad():
        try:
            result = pipe.encode_prompt(**kwargs)
            if isinstance(result, tuple):
                prompt_embeds = result[0]
                pooled_prompt_embeds = result[1] if len(result) > 1 else None
            elif isinstance(result, dict):
                prompt_embeds = result.get("prompt_embeds")
                pooled_prompt_embeds = result.get("pooled_prompt_embeds")
            else:
                prompt_embeds = result
                pooled_prompt_embeds = None
        except TypeError:
            # Older/newer diffusers builds may expose a slightly different Flux 2 API.
            fallback = pipe.encode_prompt(
                [prompt],
            )
            if not isinstance(fallback, tuple):
                raise RuntimeError("Unexpected Flux 2 encode_prompt return type.")
            prompt_embeds = fallback[0]
            pooled_prompt_embeds = fallback[1] if len(fallback) > 1 else None

    if prompt_embeds is None:
        raise RuntimeError("Flux 2 prompt encoding did not return `prompt_embeds`.")

    if pooled_prompt_embeds is None:
        pooled_prompt_embeds = torch.zeros(
            (prompt_embeds.shape[0], 768),
            dtype=prompt_embeds.dtype,
            device=prompt_embeds.device,
        )

    return prompt_embeds, pooled_prompt_embeds


def main():
    args = parse_args()
    dtype = resolve_dtype(args.dtype)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = Flux2Pipeline.from_pretrained(
        args.model,
        transformer=None,
        vae=None,
        torch_dtype=dtype,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        pipe=pipe,
        prompt=args.prompt,
        max_sequence_length=args.max_sequence_length,
    )

    payload = {
        "txt": prompt_embeds.detach().cpu(),
        "vec": pooled_prompt_embeds.detach().cpu(),
        "prompt": args.prompt,
        "model": args.model,
        "dtype": args.dtype,
    }
    torch.save(payload, output_path)

    print(f"Saved Flux 2 text embeddings to {output_path}")
    print(f"txt shape: {tuple(payload['txt'].shape)} dtype: {payload['txt'].dtype}")
    print(f"vec shape: {tuple(payload['vec'].shape)} dtype: {payload['vec'].dtype}")


if __name__ == "__main__":
    main()
