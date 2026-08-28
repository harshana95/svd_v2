"""Flux2 block wrappers that apply PCM + Coupler before frozen blocks."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .context import BlurContext
from .coupler import BlurResidualCoupler
from .modulation import TimestepLayerModulation
from .pcm import PCMModule
from .pga import Flux2PGAAttnProcessor, Flux2ParallelPGAAttnProcessor, PGAModule


def _modulate_noise_prefix(h: torch.Tensor, ctx: BlurContext, ell: int, mod):
    """Apply PCM + Coupler to the noisy image prefix; leave LQ condition tokens alone.

    Double-stream ``h`` layout: ``[noise; cond]``.
    """
    n_img = ctx.num_img_tokens
    noise = h[:, :n_img, :]
    noise = ctx.pcm[ell](noise, ctx.jepa_tokens, mod)
    noise = ctx.coupler[ell](noise, ctx.blur_tokens, ctx.jepa_tokens, mod)
    if h.shape[1] == n_img:
        return noise
    return torch.cat([noise, h[:, n_img:, :]], dim=1)


def _modulate_noise_in_concat(h: torch.Tensor, ctx: BlurContext, ell: int, mod):
    """Apply PCM + Coupler to noise tokens inside ``[txt; noise; cond]``."""
    n_img = ctx.num_img_tokens
    n_cond = ctx.num_cond_tokens
    noise_start = h.shape[1] - n_img - n_cond
    noise_end = noise_start + n_img
    noise = h[:, noise_start:noise_end, :]
    noise = ctx.pcm[ell](noise, ctx.jepa_tokens, mod)
    noise = ctx.coupler[ell](noise, ctx.blur_tokens, ctx.jepa_tokens, mod)
    return torch.cat([h[:, :noise_start, :], noise, h[:, noise_end:, :]], dim=1)


class Flux2DoubleBlockWrapper(nn.Module):
    """Wraps a Flux2 double-stream block; modulates noisy image tokens only."""

    def __init__(self, block: nn.Module, ell: int, context: BlurContext):
        super().__init__()
        self.block = block
        self.ell = ell
        self.context = context

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_img: torch.Tensor,
        temb_mod_txt: torch.Tensor,
        image_rotary_emb=None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ):
        ctx = self.context
        if ctx.enabled:
            mod = ctx.modulation(ctx.t, self.ell)
            hidden_states = _modulate_noise_prefix(
                hidden_states, ctx, self.ell, mod
            )
            proc = self.block.attn.processor
            if hasattr(proc, "set_step"):
                proc.set_step(mod)
        return self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_img=temb_mod_img,
            temb_mod_txt=temb_mod_txt,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )


class Flux2SingleBlockWrapper(nn.Module):
    """Wraps a Flux2 single-stream block; modulates noisy image tokens only."""

    def __init__(self, block: nn.Module, ell: int, context: BlurContext):
        super().__init__()
        self.block = block
        self.ell = ell
        self.context = context

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb_mod: torch.Tensor,
        image_rotary_emb=None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        split_hidden_states: bool = False,
        text_seq_len: int | None = None,
    ):
        ctx = self.context
        if ctx.enabled:
            mod = ctx.modulation(ctx.t, self.ell)
            if encoder_hidden_states is None:
                # Sequence is [txt; noise; cond] after outer concat in Flux2.forward.
                hidden_states = _modulate_noise_in_concat(
                    hidden_states, ctx, self.ell, mod
                )
            else:
                hidden_states = _modulate_noise_prefix(
                    hidden_states, ctx, self.ell, mod
                )
            proc = self.block.attn.processor
            if hasattr(proc, "set_step"):
                proc.set_step(mod)
        return self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod=temb_mod,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
            split_hidden_states=split_hidden_states,
            text_seq_len=text_seq_len,
        )


def install_blur_adapters(
    transformer: nn.Module,
    *,
    blur_dim: int = 64,
    global_dim: int = 64,
    num_bands: int = 16,
    pga_rank: int = 8,
    adapter_hidden: int = 64,
) -> tuple[
    BlurContext,
    TimestepLayerModulation,
    nn.ModuleList,
    nn.ModuleList,
    nn.ModuleList,
]:
    """Install PCM, Coupler, and PGA on every Flux2 double/single block.

    Returns context + trainable module lists (modulation, pga, pcm, coupler).
    """
    heads = transformer.config.num_attention_heads
    head_dim = transformer.config.attention_head_dim
    dim = heads * head_dim

    num_dual = len(transformer.transformer_blocks)
    num_single = len(transformer.single_transformer_blocks)
    num_blocks = num_dual + num_single

    context = BlurContext()
    modulation = TimestepLayerModulation(num_blocks)
    pga = nn.ModuleList(
        [
            PGAModule(
                dim, heads, head_dim, blur_dim, global_dim, num_bands, pga_rank
            )
            for _ in range(num_blocks)
        ]
    )
    pcm = nn.ModuleList(
        [PCMModule(dim, adapter_hidden) for _ in range(num_blocks)]
    )
    coupler = nn.ModuleList(
        [
            BlurResidualCoupler(dim, blur_dim, adapter_hidden)
            for _ in range(num_blocks)
        ]
    )

    context.modulation = modulation
    context.pcm = pcm
    context.coupler = coupler

    idx = 0
    for j in range(num_dual):
        block = transformer.transformer_blocks[j]
        block.attn.processor = Flux2PGAAttnProcessor(pga[idx], context)
        transformer.transformer_blocks[j] = Flux2DoubleBlockWrapper(
            block, idx, context
        )
        idx += 1

    for j in range(num_single):
        block = transformer.single_transformer_blocks[j]
        block.attn.processor = Flux2ParallelPGAAttnProcessor(pga[idx], context)
        transformer.single_transformer_blocks[j] = Flux2SingleBlockWrapper(
            block, idx, context
        )
        idx += 1

    return context, modulation, pga, pcm, coupler
