"""Physically-Guided Adaptation (PGA) for Flux2 attention processors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.transformers.transformer_flux2 import (
    _get_qkv_projections,
    apply_rotary_emb,
    dispatch_attention_fn,
)

from .context import BlurContext
from .modulation import BlockModulation


class PGAModule(nn.Module):
    """Per-block trainable PGA parameters (gated low-rank residual on V)."""

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        blur_dim: int,
        global_dim: int,
        num_bands: int,
        rank: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.rank = rank

        self.A_v = nn.Parameter(torch.randn(dim, rank) * (1.0 / dim**0.5))
        self.B_v = nn.Parameter(torch.zeros(rank, dim))
        self.P_v = nn.Linear(blur_dim + global_dim, heads)
        self.W_r = nn.Linear(num_bands, heads)

    def value_residual(
        self,
        h: torch.Tensor,
        mod: BlockModulation,
        blur_pooled: torch.Tensor,
        g: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """h: ``[B, M, dim]`` (image tokens) -> residual ``[B, M, dim]``."""
        b, m, d = h.shape
        hf = h.float()
        base = (hf @ self.A_v) @ self.B_v
        dv = mod.alpha_pga[:, None] * base + mod.beta_pga[:, None] * hf
        dv = dv.view(b, m, self.heads, self.head_dim)

        gate_phys = torch.sigmoid(
            self.P_v(torch.cat([blur_pooled.float(), g.float()], dim=-1))
        )
        gate_spec = F.softplus(self.W_r(r.float()))
        scale = gate_phys * (1.0 + mod.n_spec * gate_spec)
        dv = dv * scale[:, None, :, None]
        return dv.reshape(b, m, d).to(h.dtype)


class Flux2PGAAttnProcessor:
    """Flux2 double-stream attention with PGA residual on image values."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, pga: PGAModule, context: BlurContext):
        self.pga = pga
        self.context = context
        self._mod: BlockModulation | None = None

    def set_step(self, mod: BlockModulation) -> None:
        self._mod = mod

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ):
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        ctx = self.context
        if ctx.enabled and self._mod is not None:
            n_img = ctx.num_img_tokens
            # Double-stream image layout: [noise; cond]. Modulate noise prefix only.
            noise_h = hidden_states[:, :n_img, :]
            dv = self.pga.value_residual(
                noise_h, self._mod, ctx.blur_pooled, ctx.g, ctx.r
            )
            value = value.clone()
            value[:, :n_img, :] = value[:, :n_img, :] + dv

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


class Flux2ParallelPGAAttnProcessor:
    """Flux2 single-stream parallel attention with PGA on the image-token tail."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, pga: PGAModule, context: BlurContext):
        self.pga = pga
        self.context = context
        self._mod: BlockModulation | None = None

    def set_step(self, mod: BlockModulation) -> None:
        self._mod = mod

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ):
        # Parallel in (QKV + MLP in) projection
        projected = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            projected,
            [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor],
            dim=-1,
        )
        query, key, value = qkv.chunk(3, dim=-1)

        ctx = self.context
        if ctx.enabled and self._mod is not None:
            n_img = ctx.num_img_tokens
            n_cond = ctx.num_cond_tokens
            # Sequence layout: [txt; noise; cond].
            noise_start = hidden_states.shape[1] - n_img - n_cond
            noise_end = noise_start + n_img
            img_h = hidden_states[:, noise_start:noise_end, :]
            dv = self.pga.value_residual(
                img_h, self._mod, ctx.blur_pooled, ctx.g, ctx.r
            )
            value = value.clone()
            value[:, noise_start:noise_end, :] = (
                value[:, noise_start:noise_end, :] + dv
            )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        attn_out = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attn_out = attn_out.flatten(2, 3).to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        out = torch.cat([attn_out, mlp_hidden_states], dim=-1)
        return attn.to_out(out)
