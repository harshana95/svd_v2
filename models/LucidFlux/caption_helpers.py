"""Image captioning and Flux2 text-token encoding for projector alignment."""

from __future__ import annotations

import torch
from PIL import Image

from .cross_attn_projector import resample_text_tokens
from .helpers import get_flux2_pipeline_class

__all__ = [
    "ImageCaptioner",
    "Flux2TextEncoder",
    "encode_flux2_text_tokens",
    "resample_text_tokens",
    "tensor_to_pil",
]


def encode_flux2_text_tokens(pipe, captions: list[str], device, dtype) -> torch.Tensor:
    """Encode captions with Flux2KleinPipeline / Flux2Pipeline text stack."""
    with torch.inference_mode():
        prompt_embeds, _ = pipe.encode_prompt(
            captions,
            num_images_per_prompt=1,
        )
    return prompt_embeds.to(device=device, dtype=dtype)


def resolve_caption_dtype(
    model_name: str,
    device: torch.device | str,
    requested_dtype: torch.dtype,
) -> torch.dtype:
    """Gemma 3 VL models need bfloat16 on GPU; float16 yields empty captions on A100."""
    if str(device) != "cpu" and "gemma" in model_name.lower():
        return torch.bfloat16
    return requested_dtype


def tensor_to_pil(image_01: torch.Tensor) -> Image.Image:
    """Convert [3, H, W] float01 tensor to RGB PIL."""
    arr = image_01.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0).round().astype("uint8")
    return Image.fromarray(arr, mode="RGB")


class ImageCaptioner:
    """Lazy-loaded VLM for GT image captions (default Qwen3-VL)."""

    def __init__(
        self,
        model_name: str,
        instruction: str,
        device: torch.device | str,
        dtype: torch.dtype,
        max_new_tokens: int = 96,
    ):
        self.model_name = model_name
        self.instruction = instruction
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor

        load_dtype = resolve_caption_dtype(self.model_name, self.device, self.dtype)
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if str(self.device) == "cpu":
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=load_dtype,
                trust_remote_code=True,
            ).to("cpu")
        else:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=load_dtype,
                device_map={"": self.device},
                trust_remote_code=True,
            )
        self._model.eval()

    def caption_batch(self, images_01: torch.Tensor) -> list[str]:
        """Caption a batch of images in [0, 1], shape [B, 3, H, W]."""
        self._ensure_loaded()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.instruction},
                ],
            }
        ]
        text_prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        pil_images = [tensor_to_pil(images_01[i]) for i in range(images_01.shape[0])]
        inputs = self._processor(
            text=[text_prompt] * len(pil_images),
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )
        model_device = next(self._model.parameters()).device
        model_dtype = next(self._model.parameters()).dtype
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = generated[:, prompt_len:]
        return self._processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )


class Flux2TextEncoder:
    """Lazy-loaded Flux2 text encoder for caption embedding targets."""

    def __init__(self, pretrained_model_name_or_path: str, device, dtype):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = device
        self.dtype = dtype
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        pipe_cls = get_flux2_pipeline_class(self.pretrained_model_name_or_path)
        self._pipe = pipe_cls.from_pretrained(
            self.pretrained_model_name_or_path,
            transformer=None,
            vae=None,
            torch_dtype=self.dtype,
        ).to(self.device)
        self._pipe.set_progress_bar_config(disable=True)

    def encode(self, captions: list[str]) -> torch.Tensor:
        self._ensure_loaded()
        return encode_flux2_text_tokens(
            self._pipe,
            captions,
            self.device,
            self.dtype,
        )
