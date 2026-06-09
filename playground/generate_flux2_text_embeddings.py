import argparse
from pathlib import Path

import torch
from diffusers import Flux2Pipeline

try:
    from diffusers import Flux2KleinPipeline
except ImportError:
    Flux2KleinPipeline = None


def get_flux2_pipeline_class(model_id: str):
    if Flux2KleinPipeline is not None and "klein" in model_id.lower():
        return Flux2KleinPipeline
    return Flux2Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and save Flux 2 text embeddings."
    )
    parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="Flux 2 pretrained model name or local path.",
    )
    parser.add_argument(
        "--output",
        default="null_prompt_embeddings_flux2_klein.pt",
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


def encode_prompt(pipe, prompt: str, max_sequence_length: int | None):
    kwargs = {
        "prompt": [prompt],
        "num_images_per_prompt": 1,
    }
    if max_sequence_length is not None:
        kwargs["max_sequence_length"] = max_sequence_length

    with torch.no_grad():
        prompt_embeds, text_ids = pipe.encode_prompt(**kwargs)

    if prompt_embeds is None:
        raise RuntimeError("Flux 2 prompt encoding did not return `prompt_embeds`.")

    return prompt_embeds, text_ids


def main():
    args = parse_args()
    dtype = resolve_dtype(args.dtype)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipe_cls = get_flux2_pipeline_class(args.model)
    pipe = pipe_cls.from_pretrained(
        args.model,
        transformer=None,
        vae=None,
        torch_dtype=dtype,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    prompt_embeds, text_ids = encode_prompt(
        pipe=pipe,
        prompt=args.prompt,
        max_sequence_length=args.max_sequence_length,
    )

    payload = {
        "txt": prompt_embeds.detach().cpu(),
        "txt_ids": text_ids.detach().cpu(),
        "prompt": args.prompt,
        "model": args.model,
        "dtype": args.dtype,
    }
    torch.save(payload, output_path)

    print(f"Saved Flux 2 text embeddings to {output_path}")
    print(f"txt shape: {tuple(payload['txt'].shape)} dtype: {payload['txt'].dtype}")
    print(f"txt_ids shape: {tuple(payload['txt_ids'].shape)} dtype: {payload['txt_ids'].dtype}")


if __name__ == "__main__":
    main()
