"""Project VL-JEPA semantic embedding S_Y into Flux 2 text conditioning tokens."""

import torch
import torch.nn as nn


class VLJEPAProjector(nn.Module):
    """Map a single VL-JEPA embedding [B, D] to Flux text slots [B, num_slots, seq_dim]."""

    def __init__(
        self,
        vljepa_dim: int = 1536,
        seq_dim: int = 15360,
        num_slots: int = 512,
        hidden_dim: int = 2048,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_slots = num_slots

        self.query_tokens = nn.Parameter(torch.randn(1, num_slots, hidden_dim) * 0.02)
        self.embed_proj = nn.Linear(vljepa_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.seq_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, seq_dim),
        )

    def forward(self, s_y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_y: [B, vljepa_dim]
        Returns:
            [B, num_slots, seq_dim]
        """
        bsz = s_y.shape[0]
        kv = self.norm_k(self.embed_proj(s_y).unsqueeze(1))
        q = self.norm_q(self.query_tokens.expand(bsz, -1, -1))
        out, _ = self.cross_attn(q, kv, kv)
        return self.seq_out(out)
