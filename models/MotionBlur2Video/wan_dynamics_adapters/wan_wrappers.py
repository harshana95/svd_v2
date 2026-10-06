"""Wan block wrapper + top-level installer for the dynamics-injection adapter."""

from __future__ import annotations

import torch.nn as nn

from models.reFlux.blur_adapters import TimestepLayerModulation

from .context import WanDynamicsContext
from .wan_pga import (
    WanDynamicsCrossAttn,
    WanDynamicsCrossAttnStack,
    WanPGAAttnProcessor,
    WanPGAModule,
)


class WanBlockWrapper(nn.Module):
    """Wraps a WanTransformerBlock; gates its attn1 processor's per-step residual."""

    def __init__(self, block: nn.Module, ell: int, context: WanDynamicsContext):
        super().__init__()
        self.block = block
        self.ell = ell
        self.context = context

    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        ctx = self.context
        if ctx.enabled:
            mod = ctx.modulation(ctx.t, self.ell)
            proc = self.block.attn1.processor
            if hasattr(proc, "set_step"):
                proc.set_step(mod)
            # Blur conditioning enters ONCE, at the input to the first block --
            # the same place widening patch_embedding would have put it.
            if self.ell == 0 and ctx.blur_tokens is not None:
                hidden_states = hidden_states + ctx.blur_tokens.to(hidden_states.dtype)
            # Cross-attention injection (alternative to the PGA value residual).
            if ctx.xattn is not None and ctx.dynamics_kv is not None:
                kv = ctx.dynamics_kv
                if isinstance(kv, (list, tuple)):   # multi-layer V-JEPA: block group g -> KV set g
                    kv = kv[min(len(kv) - 1, self.ell * len(kv) // max(ctx.num_blocks, 1))]
                res = ctx.xattn[self.ell](hidden_states, kv)
                hidden_states = hidden_states + res.to(hidden_states.dtype)
        return self.block(hidden_states, encoder_hidden_states, temb, rotary_emb)


def install_wan_dynamics_adapters(
    transformer: nn.Module, *, rank: int = 8, cross_attention: bool = False,
    xattn_rank: int = 64, n_tubelets: int = 15, grid_h: int = 24, grid_w: int = 24,
    max_tubelets: int | None = None,
) -> tuple[WanDynamicsContext, TimestepLayerModulation, nn.ModuleList, nn.ModuleList | None]:
    """Install a PGA-style dynamics residual on every Wan self-attention block.

    Returns (context, modulation, pga) -- the trainable module list is `pga`;
    `modulation` is also trainable (its head is zero-init, everything else
    frozen-shape but trained).
    """
    heads = transformer.config.num_attention_heads
    head_dim = transformer.config.attention_head_dim
    dim = heads * head_dim

    num_blocks = len(transformer.blocks)

    context = WanDynamicsContext()
    modulation = TimestepLayerModulation(num_blocks)
    pga = nn.ModuleList([WanPGAModule(dim, rank=rank) for _ in range(num_blocks)])
    context.modulation = modulation
    context.num_blocks = num_blocks

    xattn = WanDynamicsCrossAttnStack([
        WanDynamicsCrossAttn(dim, rank=xattn_rank, n_tubelets=n_tubelets,
                             max_tubelets=max_tubelets,
                             grid_h=grid_h, grid_w=grid_w)
        for _ in range(num_blocks)
    ]) if cross_attention else None

    for ell in range(num_blocks):
        block = transformer.blocks[ell]
        block.attn1.processor = WanPGAAttnProcessor(pga[ell], context)
        transformer.blocks[ell] = WanBlockWrapper(block, ell, context)

    context.xattn = xattn  # referenced, NOT registered as a submodule
    return context, modulation, pga, xattn
