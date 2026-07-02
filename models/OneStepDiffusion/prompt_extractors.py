"""Pluggable image-to-text prompt extractors for OSEDiff SD2 models."""

from __future__ import annotations

import torch
from torchvision import transforms

from ram import inference_ram as inference


CAPTION_MODEL_NAMES = {
    "qwen": "Qwen/Qwen3-VL-8B-Instruct",
    "gemma": "google/gemma-3-4b-it",
    "florence": "microsoft/Florence-2-large",
}

EXTRACTOR_DEFAULTS = {
    "qwen": {
        "caption_instruction": "Describe this image in one concise caption.",
        "caption_max_new_tokens": 96,
    },
    "gemma": {
        "caption_instruction": "Describe this image in one concise caption.",
        "caption_max_new_tokens": 96,
    },
    "florence": {
        "florence_task": "<CAPTION>",
        "caption_max_new_tokens": 1024,
    },
}


def _cfg(opt, key, default=None):
    value = getattr(opt, key, default)
    if value is None and hasattr(opt, "get"):
        value = opt.get(key, default)
    return default if value is None else value


def get_caption_generator(model_path, **kwargs):
    from ram.models.ram_lora import ram

    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model_vlm = ram(
        pretrained=model_path,
        pretrained_condition=kwargs.get("dape_path"),
        image_size=384,
        vit="swin_l",
    )
    return model_vlm, ram_transforms


class BasePromptExtractor:
    def setup(self, device):
        pass

    def get_captions(self, images: torch.Tensor, bsz: int) -> list[str]:
        raise NotImplementedError


class NullPromptExtractor(BasePromptExtractor):
    def get_captions(self, images, bsz):
        return [""] * bsz


class RamDapePromptExtractor(BasePromptExtractor):
    def __init__(self, vlm_model_path: str, dape_path: str):
        self.vlm_model_path = vlm_model_path
        self.dape_path = dape_path
        self.model_vlm = None
        self.transforms = None
        self._device = None

    def setup(self, device):
        self._device = device
        self.model_vlm, self.transforms = get_caption_generator(
            self.vlm_model_path,
            dape_path=self.dape_path,
        )
        self.model_vlm.eval()
        self.model_vlm.to(device, dtype=torch.float16)

    def get_captions(self, images, bsz):
        ram_images = self.transforms(images * 0.5 + 0.5)
        return list(inference(ram_images.to(dtype=torch.float16), self.model_vlm))


class ChatVLMExtractor(BasePromptExtractor):
    def __init__(self, model_name: str, instruction: str, max_new_tokens: int):
        self.model_name = model_name
        self.instruction = instruction
        self.max_new_tokens = max_new_tokens
        self._captioner = None
        self._device = None

    def setup(self, device):
        self._device = device

    def _ensure_loaded(self):
        if self._captioner is not None:
            return
        from models.LucidFlux.caption_helpers import ImageCaptioner

        self._captioner = ImageCaptioner(
            model_name=self.model_name,
            instruction=self.instruction,
            device=self._device,
            dtype=torch.float16,
            max_new_tokens=self.max_new_tokens,
        )

    def get_captions(self, images, bsz):
        self._ensure_loaded()
        images_01 = images * 0.5 + 0.5
        return self._captioner.caption_batch(images_01)


class FlorenceExtractor(BasePromptExtractor):
    def __init__(self, model_name: str, task: str, max_new_tokens: int):
        self.model_name = model_name
        self.task = task
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._device = None

    def setup(self, device):
        self._device = device

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(self._device)
        self._model.eval()

    def get_captions(self, images, bsz):
        from models.LucidFlux.caption_helpers import tensor_to_pil

        self._ensure_loaded()
        images_01 = images * 0.5 + 0.5
        model_dtype = next(self._model.parameters()).dtype
        captions = []
        for i in range(bsz):
            pil_image = tensor_to_pil(images_01[i])
            inputs = self._processor(
                text=self.task,
                images=pil_image,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False,
                )
            generated_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
            parsed = self._processor.post_process_generation(
                generated_text,
                task=self.task,
                image_size=(pil_image.width, pil_image.height),
            )
            captions.append(parsed[self.task])
        return captions


def resolve_extractor_type(opt) -> str:
    extractor_type = _cfg(opt, "text_prompt_extractor")
    if extractor_type is None:
        return "ram_dape" if _cfg(opt, "dape_path") is not None else "null"
    return extractor_type


def build_prompt_extractor(opt) -> BasePromptExtractor:
    extractor_type = resolve_extractor_type(opt)
    supported = ("null", "ram_dape", "qwen", "gemma", "florence")
    if extractor_type not in supported:
        raise ValueError(
            f"Unknown text_prompt_extractor '{extractor_type}'. "
            f"Supported: {list(supported)}"
        )

    if extractor_type == "null":
        return NullPromptExtractor()

    if extractor_type == "ram_dape":
        vlm_model_path = _cfg(opt, "vlm_model_path")
        dape_path = _cfg(opt, "dape_path")
        if not vlm_model_path or not dape_path:
            raise ValueError("text_prompt_extractor=ram_dape requires vlm_model_path and dape_path")
        return RamDapePromptExtractor(vlm_model_path, dape_path)

    if extractor_type in ("qwen", "gemma"):
        defaults = EXTRACTOR_DEFAULTS[extractor_type]
        return ChatVLMExtractor(
            model_name=CAPTION_MODEL_NAMES[extractor_type],
            instruction=_cfg(opt, "caption_instruction", defaults["caption_instruction"]),
            max_new_tokens=int(_cfg(opt, "caption_max_new_tokens", defaults["caption_max_new_tokens"])),
        )

    defaults = EXTRACTOR_DEFAULTS["florence"]
    return FlorenceExtractor(
        model_name=CAPTION_MODEL_NAMES["florence"],
        task=_cfg(opt, "florence_task", defaults["florence_task"]),
        max_new_tokens=int(_cfg(opt, "caption_max_new_tokens", defaults["caption_max_new_tokens"])),
    )
