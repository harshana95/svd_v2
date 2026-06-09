"""Cross-attention projectors mapping encoder features into Flux 2 text conditioning tokens."""

import torch
import torch.nn as nn


class PatchCrossAttnProjector(nn.Module):
    """Map encoder features to Flux text slots via learned-query cross-attention.

    Accepts either a global embedding [B, embed_dim] or patch tokens [B, N, embed_dim].
    """

    def __init__(
        self,
        embed_dim: int,
        seq_dim: int,
        num_slots: int,
        hidden_dim: int = 2048,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_slots = num_slots

        self.query_tokens = nn.Parameter(torch.randn(1, num_slots, hidden_dim) * 0.02)
        self.embed_proj = nn.Linear(embed_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.seq_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, seq_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, embed_dim] or [B, N, embed_dim]
        Returns:
            [B, num_slots, seq_dim]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        bsz = x.shape[0]
        kv = self.norm_k(self.embed_proj(x))
        q = self.norm_q(self.query_tokens.expand(bsz, -1, -1))
        out, _ = self.cross_attn(q, kv, kv)
        return self.seq_out(out)


class VLJEPAProjector(PatchCrossAttnProjector):
    """Backward-compatible alias for VL-JEPA global-embedding conditioning."""

    def __init__(
        self,
        vljepa_dim: int = 1536,
        seq_dim: int = 15360,
        num_slots: int = 512,
        hidden_dim: int = 2048,
        num_heads: int = 8,
    ):
        super().__init__(
            embed_dim=vljepa_dim,
            seq_dim=seq_dim,
            num_slots=num_slots,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
