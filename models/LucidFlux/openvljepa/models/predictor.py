"""Predictor: last 8 transformer layers of Llama-3.2-1B (pretrained), non-causal."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class Predictor(nn.Module):
    """V-JEPA2 visual embeddings + query tokens → predicted target embedding."""

    def __init__(
        self,
        llama_name: str = "meta-llama/Llama-3.2-1B",
        n_keep_layers: int = 8,
        vis_dim: int = 1024,
        proj_dim: int = 1536,
        disable_causal_mask: bool = True,
        torch_dtype=None,
    ):
        super().__init__()
        full = AutoModelForCausalLM.from_pretrained(llama_name, torch_dtype=torch_dtype)
        base = full.model
        self.hidden_size = base.config.hidden_size

        self.embed_tokens = base.embed_tokens
        self.embed_tokens.requires_grad_(False)
        self.layers = nn.ModuleList(list(base.layers[-n_keep_layers:]))
        self.norm = base.norm
        self.rotary_emb = getattr(base, "rotary_emb", None)

        self.vis_proj = nn.Linear(vis_dim, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, proj_dim)

        self.disable_causal_mask = disable_causal_mask

        base.layers = nn.ModuleList()
        base.embed_tokens = None
        base.norm = None
        if hasattr(base, "rotary_emb"):
            base.rotary_emb = None
        del full

    def _build_bidirectional_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        B, _ = attention_mask.shape
        pad = (1.0 - attention_mask.to(torch.float32))
        neg_inf = torch.finfo(dtype).min
        mask = (pad * neg_inf).to(dtype)
        return mask[:, None, None, :].expand(B, 1, attention_mask.shape[1], attention_mask.shape[1]).contiguous()

    def forward(
        self,
        vis_embeds: torch.Tensor,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert query_mask is not None, "query_mask is required for Open-VLJEPA predictor."

        B, Nv, _ = vis_embeds.shape

        v = self.vis_proj(vis_embeds)
        q = self.embed_tokens(query_ids)
        h = torch.cat([v, q], dim=1)
        seq_len = h.shape[1]

        vis_mask = torch.ones(B, Nv, device=h.device, dtype=torch.float32)
        full_pad_mask = torch.cat([vis_mask, query_mask.to(torch.float32)], dim=1)

        if self.disable_causal_mask:
            attn_mask_4d = self._build_bidirectional_mask(full_pad_mask, h.dtype)
        else:
            neg_inf = torch.finfo(h.dtype).min
            causal = torch.triu(
                torch.full((seq_len, seq_len), neg_inf, dtype=h.dtype, device=h.device),
                diagonal=1,
            )
            pad = (1.0 - full_pad_mask.to(torch.float32)).to(h.dtype) * neg_inf
            attn_mask_4d = (causal[None, None, :, :] + pad[:, None, None, :]).contiguous()

        position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0).expand(B, -1)
        position_embeddings = self.rotary_emb(h, position_ids) if self.rotary_emb is not None else None

        for layer in self.layers:
            layer_out = layer(
                h,
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            h = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        h = self.norm(h)
        mask = full_pad_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.out_proj(pooled)
