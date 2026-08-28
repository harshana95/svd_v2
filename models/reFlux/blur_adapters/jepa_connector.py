"""Trainable connector mapping frozen V-JEPA tokens into Flux2 inner_dim."""

from __future__ import annotations

import torch
import torch.nn as nn


class JEPAPerceptualEncoder(nn.Module):
    """Linear projector: JEPA patch tokens ``[B, N, Dj]`` -> ``[B, N, dim]``."""

    def __init__(self, jepa_dim: int, model_dim: int):
        super().__init__()
        self.connector = nn.Linear(jepa_dim, model_dim)

    def forward(self, jepa_tokens: torch.Tensor) -> torch.Tensor:
        return self.connector(jepa_tokens.float())
