"""Shared per-forward conditioning state for blur-map + JEPA adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BlurContext:
    enabled: bool = False
    t: torch.Tensor | None = None  # [B] flow time in [0, 1]
    blur_tokens: torch.Tensor | None = None  # [B, N_img, Cblur]
    jepa_tokens: torch.Tensor | None = None  # [B, N_img, dim]
    blur_pooled: torch.Tensor | None = None  # [B, Cblur]
    g: torch.Tensor | None = None  # [B, Cg]
    r: torch.Tensor | None = None  # [B, K]
    num_img_tokens: int = 0  # noisy image token count
    num_cond_tokens: int = 0  # LQ condition tokens appended after noise

    # Module refs filled once at install time (not reset per forward).
    modulation: object | None = None
    pcm: object | None = None
    coupler: object | None = None

    def reset(self) -> None:
        self.enabled = False
        self.t = None
        self.blur_tokens = None
        self.jepa_tokens = None
        self.blur_pooled = None
        self.g = None
        self.r = None
        self.num_img_tokens = 0
        self.num_cond_tokens = 0
