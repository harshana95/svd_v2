"""Helpers for frozen Open-VLJEPA semantic conditioning."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer

from .openvljepa.models.vljepa import OpenVLJEPA

VLJEPA_EMBED_DIM = 1536
LLAMA32_PAD = "<|finetune_right_pad_id|>"


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token is not None:
        return tokenizer
    vocab = tokenizer.get_vocab()
    if LLAMA32_PAD in vocab:
        tokenizer.pad_token = LLAMA32_PAD
        tokenizer.pad_token_id = vocab[LLAMA32_PAD]
    else:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_ram_tag_candidates(repo_root: str | None = None) -> list[str]:
    root = repo_root or Path(__file__).resolve().parents[2]
    tag_path = Path(root) / "ram" / "data" / "ram_tag_list.txt"
    if not tag_path.is_file():
        raise FileNotFoundError(f"RAM tag list not found: {tag_path}")
    with open(tag_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def infer_num_text_slots(
    precomputed_embeddings_path: str | None,
    default: int = 512,
) -> int:
    if precomputed_embeddings_path and os.path.exists(precomputed_embeddings_path):
        data = torch.load(precomputed_embeddings_path, map_location="cpu", weights_only=False)
        txt = data.get("txt")
        if txt is not None and txt.ndim >= 2:
            return int(txt.shape[1])
    return default


def load_open_vljepa(
    checkpoint_path: str,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[OpenVLJEPA, dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Open-VLJEPA checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    try:
        model = OpenVLJEPA(
            cfg["encoder"],
            cfg["y_encoder"],
            cfg["predictor"],
            torch_dtype=dtype,
        )
    except OSError as exc:
        raise OSError(
            "Failed to load Open-VLJEPA backbone weights from Hugging Face. "
            "Ensure `huggingface-cli login` is configured and you have accepted "
            "the licenses for google/embeddinggemma-300m and meta-llama/Llama-3.2-1B."
        ) from exc
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    model.requires_grad_(False)
    for module in (model.x_encoder, model.predictor, model.y_encoder):
        module.eval()
        module.requires_grad_(False)

    model = model.to(device=device, dtype=dtype)
    return model, cfg


def load_vljepa_tokenizers(cfg: dict):
    query_tokenizer = _ensure_pad_token(
        AutoTokenizer.from_pretrained(cfg["predictor"]["llama_name"])
    )
    target_tokenizer = _ensure_pad_token(
        AutoTokenizer.from_pretrained(cfg["y_encoder"]["model_name"])
    )
    return query_tokenizer, target_tokenizer


def preprocess_vljepa_image(
    x: torch.Tensor,
    num_frames: int = 1,
    image_size: int = 256,
    device: torch.device | str | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert [0,1] BCHW tensor to Open-VLJEPA input (B, T, C, H, W)."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    assert x.ndim == 4 and x.shape[1] == 3, "Expected shape (B, 3, H, W)"

    resize = transforms.Resize(
        (image_size, image_size),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    frames = []
    for b in range(x.shape[0]):
        frame = resize(x[b : b + 1].float())
        frame = (frame - mean.to(frame.device)) / std.to(frame.device)
        if num_frames > 1:
            frames.append(frame.expand(num_frames, -1, -1, -1))
        else:
            frames.append(frame)

    out = torch.stack(frames, dim=0)
    if device is not None:
        out = out.to(device=device, dtype=out_dtype)
    else:
        out = out.to(dtype=out_dtype)
    return out


def tokenize_vljepa_query(
    query: str,
    query_tokenizer,
    max_length: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    enc = query_tokenizer(
        query,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.inference_mode()
def predict_semantic_embedding(
    model: OpenVLJEPA,
    pixel_values: torch.Tensor,
    query_ids: torch.Tensor,
    query_mask: torch.Tensor,
) -> torch.Tensor:
    return model.predict_embedding(pixel_values, query_ids, query_mask)


class VLJEPACaptionReadout:
    """Frozen Y-encoder nearest-neighbor retrieval over a candidate tag/caption bank."""

    def __init__(
        self,
        model: OpenVLJEPA,
        target_tokenizer,
        candidates: Iterable[str],
        device: torch.device | str,
        max_caption_len: int = 128,
        batch_size: int = 64,
    ):
        self.model = model
        self.target_tokenizer = target_tokenizer
        self.candidates = list(candidates)
        self.device = device
        self.max_caption_len = max_caption_len
        self.candidate_embeds = self._encode_candidates(batch_size)

    @torch.inference_mode()
    def _encode_candidates(self, batch_size: int) -> torch.Tensor:
        embeds = []
        for start in range(0, len(self.candidates), batch_size):
            chunk = self.candidates[start : start + batch_size]
            enc = self.target_tokenizer(
                chunk,
                max_length=self.max_caption_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            emb = self.model.y_encoder(ids, mask)
            embeds.append(F.normalize(emb.float(), dim=-1).cpu())
        return torch.cat(embeds, dim=0)

    @torch.inference_mode()
    def retrieve(self, s_y: torch.Tensor) -> list[str]:
        if s_y.ndim == 1:
            s_y = s_y.unsqueeze(0)
        pred = F.normalize(s_y.float(), dim=-1).cpu()
        sims = pred @ self.candidate_embeds.T
        indices = sims.argmax(dim=-1).tolist()
        return [self.candidates[i] for i in indices]
