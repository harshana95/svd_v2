"""Shared per-forward conditioning state.

A single mutable ``LumaContext`` is populated by :class:`LumaFlux` at the start of
every transformer forward and read by the per-block wrappers (PCM / Coupler) and
the PGA attention processors.  This lets us inject conditioning into the frozen
diffusers FLUX backbone without changing its block call signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LumaContext:
    enabled: bool = False
    t: torch.Tensor | None = None              # [B] flow time in [0, 1]
    tphys_tokens: torch.Tensor | None = None   # [B, N_img, Cphys]
    tperc_tokens: torch.Tensor | None = None   # [B, N_img, model_dim]
    tphys_pooled: torch.Tensor | None = None   # [B, Cphys]
    g: torch.Tensor | None = None              # [B, Cg]
    r: torch.Tensor | None = None              # [B, K]
    num_img_tokens: int = 0

    def reset(self) -> None:
        self.enabled = False
        self.t = None
        self.tphys_tokens = None
        self.tperc_tokens = None
        self.tphys_pooled = None
        self.g = None
        self.r = None
        self.num_img_tokens = 0
