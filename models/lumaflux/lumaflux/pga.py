"""Physically-Guided Adaptation (PGA) -- paper Sec. 4.3 / App. D.1.

PGA injects physically grounded luminance cues into attention through a *gated
low-rank residual* on the value projection of each MM-DiT attention block:

    R_base_v = A_v B_v            (rank r << d)
    G_phys   = diag(sigmoid(P_v [Tphys || g]))      per-head intensity gate
    g_FFT    = softplus(W_r r)                       per-head spectral gate
    R^{t,l}_v = (alpha_pga R_base_v + beta_pga I) G_phys (I + n_spec diag(g_FFT))
    W_V <- W^0_V + R^{t,l}_v

i.e. the value of the image tokens becomes  V <- V + dV  where ``dV`` is the
low-rank, per-head-gated, timestep/layer-scaled residual.  Implemented as a drop-in
``FluxAttnProcessor`` replacement that reproduces the diffusers FLUX attention math
and only perturbs the value tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformers.transformer_flux import (
    _get_projections,
    _get_fused_projections,
    apply_rotary_emb,
    dispatch_attention_fn,
)

from .context import LumaContext
from .modulation import BlockModulation


class PGAModule(nn.Module):
    """Per-block trainable PGA parameters."""

    def __init__(self, dim: int, heads: int, head_dim: int, phys_dim: int,
                 global_dim: int, num_bands: int, rank: int = 8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.rank = rank

        # Low-rank residual A_v B_v (B_v init zero -> starts as identity mapping).
        self.A_v = nn.Parameter(torch.randn(dim, rank) * (1.0 / dim ** 0.5))
        self.B_v = nn.Parameter(torch.zeros(rank, dim))
        # Per-head gates from physical cues / spectral bands.
        self.P_v = nn.Linear(phys_dim + global_dim, heads)
        self.W_r = nn.Linear(num_bands, heads)

    def value_residual(self, h: torch.Tensor, mod: BlockModulation,
                       tphys_pooled: torch.Tensor, g: torch.Tensor,
                       r: torch.Tensor) -> torch.Tensor:
        """h: ``[B, M, dim]`` (image tokens) -> residual ``[B, M, dim]``."""
        b, m, d = h.shape
        hf = h.float()
        base = (hf @ self.A_v) @ self.B_v                      # [B, M, dim]
        dv = mod.alpha_pga[:, None] * base + mod.beta_pga[:, None] * hf
        dv = dv.view(b, m, self.heads, self.head_dim)

        gate_phys = torch.sigmoid(self.P_v(torch.cat([tphys_pooled.float(), g.float()], dim=-1)))
        gate_spec = F.softplus(self.W_r(r.float()))                                # [B, H]
        scale = gate_phys * (1.0 + mod.n_spec * gate_spec)                         # [B, H]
        dv = dv * scale[:, None, :, None]
        return dv.reshape(b, m, d).to(h.dtype)


class PGAAttnProcessor:
    """FLUX attention processor that adds the PGA value residual.

    Mirrors ``diffusers`` ``FluxAttnProcessor`` exactly, perturbing only ``value``
    for the image tokens.  Conditioning for the current step is set via
    :meth:`set_step` by the block wrapper before each call.
    """

    def __init__(self, pga: PGAModule, context: LumaContext):
        self.pga = pga
        self.context = context
        self._mod: BlockModulation | None = None

    def set_step(self, mod: BlockModulation) -> None:
        self._mod = mod

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        ctx = self.context
        is_single = encoder_hidden_states is None

        if getattr(attn, "fused_projections", False):
            query, key, value, e_q, e_k, e_v = _get_fused_projections(
                attn, hidden_states, encoder_hidden_states)
        else:
            query, key, value, e_q, e_k, e_v = _get_projections(
                attn, hidden_states, encoder_hidden_states)

        # --- PGA: gated low-rank residual on the image-token values. ---
        if ctx.enabled and self._mod is not None:
            if is_single:
                # hidden_states == [text ; image]; perturb only the image tail.
                n_img = ctx.num_img_tokens
                img_h = hidden_states[:, -n_img:, :]
                dv = self.pga.value_residual(img_h, self._mod, ctx.tphys_pooled, ctx.g, ctx.r)
                value = value.clone()
                value[:, -n_img:, :] = value[:, -n_img:, :] + dv
            else:
                dv = self.pga.value_residual(hidden_states, self._mod, ctx.tphys_pooled, ctx.g, ctx.r)
                value = value + dv

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            e_q = e_q.unflatten(-1, (attn.heads, -1))
            e_k = e_k.unflatten(-1, (attn.heads, -1))
            e_v = e_v.unflatten(-1, (attn.heads, -1))
            e_q = attn.norm_added_q(e_q)
            e_k = attn.norm_added_k(e_k)
            query = torch.cat([e_q, query], dim=1)
            key = torch.cat([e_k, key], dim=1)
            value = torch.cat([e_v, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        out = dispatch_attention_fn(query, key, value, attn_mask=attention_mask)
        out = out.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            enc, img = out.split_with_sizes(
                [encoder_hidden_states.shape[1], out.shape[1] - encoder_hidden_states.shape[1]], dim=1)
            img = attn.to_out[0](img.contiguous())
            img = attn.to_out[1](img)
            enc = attn.to_add_out(enc.contiguous())
            return img, enc
        return out
