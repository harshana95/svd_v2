"""Physically-Guided-Adaptation-style dynamics injection for Wan self-attention.

Ports the zero-init, per-layer + per-timestep gated, low-rank value-residual
idea from `models/reFlux/blur_adapters/pga.py` to Wan's self-attention
(`WanAttention.attn1`). Unlike the Flux2 original, there is no hand-crafted
blur map here, so there is no physical/spectral gate (`P_v`/`W_r`) -- the
residual is computed directly from the dynamics tokens themselves (fixing a
bug found in the Flux2 original, where `PGAModule.value_residual` never
actually consumed `jepa_tokens` -- only the out-of-scope-here PCM/Coupler
modules did), gated purely by `TimestepLayerModulation`'s zero-init
`alpha_pga`/`beta_pga` (`alpha_pga` starts at ~0.69*softplus... no -- see
`TimestepLayerModulation.head`, which is explicitly zero-initialized, so
both are exactly 0 at init and the residual is a true no-op until trained).

`WanAttention.attn1` is always self-attention with `encoder_hidden_states=None`
and `added_kv_proj_dim=None` (confirmed in
`diffusers/models/transformers/transformer_wan.py`'s `WanTransformerBlock`),
so unlike Flux2's dual-stream processor there is no noise/cond token
prefix-slicing needed and no "image kv" branch to handle.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import _get_qkv_projections

from models.reFlux.blur_adapters import BlockModulation

from .context import WanDynamicsContext


class WanPGAModule(nn.Module):
    """Per-block trainable low-rank value residual, driven by dynamics tokens."""

    def __init__(self, dim: int, rank: int = 8):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.A_v = nn.Parameter(torch.randn(dim, rank) * dim**-0.5)
        self.B_v = nn.Parameter(torch.zeros(rank, dim))  # zero-init

    def value_residual(self, dynamics_full: torch.Tensor, mod: BlockModulation) -> torch.Tensor:
        """dynamics_full: [B, N, dim] (N == attn1's token count) -> residual [B, N, dim]."""
        df = dynamics_full.float()
        dv = mod.alpha_pga[:, None] * ((df @ self.A_v) @ self.B_v) + mod.beta_pga[:, None] * df
        return dv


def _apply_wan_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    """Copied verbatim from WanAttnProcessor.__call__ -- Wan's rotary formula is a
    local closure there, not exported at module level, and differs from Flux2's."""
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class WanPGAAttnProcessor:
    """Wan self-attention with a PGA value residual driven by dynamics tokens."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, pga: WanPGAModule, context: WanDynamicsContext):
        self.pga = pga
        self.context = context
        self._mod: BlockModulation | None = None

    def set_step(self, mod: BlockModulation) -> None:
        self._mod = mod

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb=None,
    ) -> torch.Tensor:
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        ctx = self.context
        if ctx.enabled and self._mod is not None and ctx.dynamics_tokens is not None:
            dv = self.pga.value_residual(ctx.dynamics_tokens, self._mod)
            value = value + dv.to(value.dtype)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = _apply_wan_rotary_emb(query, *rotary_emb)
            key = _apply_wan_rotary_emb(key, *rotary_emb)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanDynamicsCrossAttn(nn.Module):
    """Alternative to WanPGAModule: inject dynamics by CROSS-ATTENTION instead of
    a value residual.

    The PGA path adds a fixed per-token vector to `value` inside self-attention:
    every spatial position receives the same additive signal for a given frame
    (with the pooled connector it is literally the same vector broadcast across
    space). Cross-attention instead lets each position *query* the dynamics set
    and pull what it needs, so different positions can read different content --
    which only pays off if the dynamics set itself has structure to attend over,
    i.e. pair it with the `factored` connector; with `pooled` the set is just T
    distinct vectors and this is close to degenerate.

    Attends to the COMPACT dynamics set (`ctx.dynamics_kv`, [B, M, dim] with
    M = T for pooled or T*H*W for factored), not the spatially-broadcast
    [B, N, dim] tensor -- same information, far cheaper, and each dynamics entry
    stays a distinct key instead of H*W duplicates.

    Runs attention in a LOW-RANK bottleneck (dim -> rank -> dim). A full-width
    version costs ~50M params per block (~1.5B over 30 blocks), which would
    dwarf the 11.8M LoRA it sits next to; at rank 64 this is ~0.9M per block,
    the same order as the existing PGA/LoRA adapters.

    `to_out` is zero-init, so the branch is an exact no-op until trained and a
    checkpoint resumed into this mode starts unperturbed.
    """

    def __init__(self, dim: int, rank: int = 64, heads: int = 4,
                 n_tubelets: int = 15, grid_h: int = 24, grid_w: int = 24,
                 max_tubelets: int | None = None):
        super().__init__()
        if rank % heads != 0:
            raise ValueError(f"rank={rank} must be divisible by heads={heads}")
        self.n_tubelets, self.grid_h, self.grid_w = n_tubelets, grid_h, grid_w
        self.heads = heads
        self.head_dim = rank // heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, rank, bias=False)
        self.to_k = nn.Linear(dim, rank, bias=False)
        self.to_v = nn.Linear(dim, rank, bias=False)
        self.to_out = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.to_out.weight)  # zero-init -> exact no-op at start
        # FACTORIZED positional embedding over the raw V-JEPA grid: one
        # embedding per tubelet index + one per spatial cell, summed. Without
        # positions the KV is an unordered bag and attention cannot tell
        # tubelet 1 from tubelet 5 -- discarding exactly the temporal ordering
        # the burst construction exists to create. Factorized rather than a
        # flat M-entry table because M = n_tubelets*grid_h*grid_w = 8640 for
        # burst-3, which as one table would cost ~16.6M params over 30 blocks
        # for positions alone; this costs ~1.1M and encodes the structure
        # explicitly (which tubelet, which cell) instead of leaving the model
        # to rediscover it.
        #
        # pos_t is allocated at max_tubelets and SLICED at runtime, so one
        # module serves every burst size. Cross-attention KV is a SET whose
        # length never has to match Wan's token count, so a varying tubelet
        # count is otherwise free -- pos_t was the only thing pinning it.
        self.max_tubelets = int(max_tubelets or n_tubelets)
        if self.max_tubelets < n_tubelets:
            raise ValueError(f"max_tubelets={self.max_tubelets} < n_tubelets={n_tubelets}")
        self.pos_t = nn.Parameter(torch.zeros(self.max_tubelets, rank))
        self.pos_s = nn.Parameter(torch.zeros(grid_h * grid_w, rank))
        nn.init.normal_(self.pos_t, std=0.02)
        nn.init.normal_(self.pos_s, std=0.02)

    def _kv_pos(self, m: int, device) -> torch.Tensor:
        """[1, M, rank] = pos_t[tubelet] + pos_s[cell], flattened (t, h, w) to
        match how the V-JEPA grid itself is flattened.

        The tubelet count is DERIVED from the incoming KV length rather than
        asserted, so a single module handles a burst size that varies per step
        (dynamic_burst). Only divisibility by the spatial grid and the
        max_tubelets bound are enforced -- both are real errors, whereas an
        exact-length check would just forbid the variation."""
        cells = self.grid_h * self.grid_w
        if m % cells != 0:
            raise ValueError(
                f"dynamics_kv length {m} is not divisible by grid_h*grid_w={cells}; "
                "the cross-attention KV must be the RAW unpooled V-JEPA grid."
            )
        t_count = m // cells
        if t_count > self.max_tubelets:
            raise ValueError(
                f"dynamics_kv has {t_count} tubelets but pos_t was allocated for "
                f"max_tubelets={self.max_tubelets}. Raise max_tubelets "
                f"(= per_segment_tubelets * max burst size) and retrain/extend pos_t."
            )
        t = self.pos_t[:t_count, None, :]               # [T,1,rank]
        sp = self.pos_s[None, :, :]                     # [cells,rank] -> [1,cells,rank]
        return (t + sp).reshape(1, m, -1).to(device)

    def forward(self, hidden_states: torch.Tensor, dynamics_kv: torch.Tensor) -> torch.Tensor:
        """hidden_states [B,N,dim], dynamics_kv [B,M,dim] -> residual [B,N,dim]."""
        b, n, _ = hidden_states.shape
        m = dynamics_kv.shape[1]
        q = self.to_q(self.norm_q(hidden_states.float()))
        kv = self.norm_kv(dynamics_kv.float())
        pos = self._kv_pos(m, kv.device)
        k = self.to_k(kv) + pos
        v = self.to_v(kv) + pos
        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, m, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, m, self.heads, self.head_dim).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim)
        return self.to_out(out)


class WanDynamicsCrossAttnStack(nn.ModuleList):
    """A plain nn.ModuleList, but with its OWN class name.

    base_model.py's save/load hooks key checkpoint folders on
    `model._get_name()` plus an occurrence counter, walking `self.models` in
    reverse. The PGA stack is a bare `nn.ModuleList` and every existing
    checkpoint therefore stores it as `ModuleList_1`. Registering a second bare
    `nn.ModuleList` for cross-attention would make the newer one claim
    `ModuleList_1` on load (it is popped first) and shift PGA to `ModuleList_2`,
    which does not exist in any prior checkpoint -- and since the loader's
    failure path only PRINTS, PGA's trained weights would be silently dropped
    while the run looked healthy. Subclassing purely to change the name keeps
    PGA's folder stable and gives this stack its own.
    """
