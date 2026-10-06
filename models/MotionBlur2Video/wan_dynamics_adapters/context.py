"""Shared per-forward conditioning state for the Wan dynamics-injection adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WanDynamicsContext:
    enabled: bool = False
    t: torch.Tensor | None = None  # [B] flow time in [0, 1] (or whatever scale `modulation` expects)
    dynamics_tokens: torch.Tensor | None = None  # [B, num_tokens, dim], already matching attn1's value token count

    # [B, num_tokens, dim] added ONCE to the hidden states entering block 0.
    # Carries the blurred input conditioning, temporally laid out so each burst
    # segment sits on the frames it was averaged from (see
    # Wan22VJEPADynamics_model._burst_blur_condition_tokens). Adding at block 0
    # is exactly equivalent to widening patch_embedding's input channels and
    # concatenating, because that conv is linear in its channel dim -- but it
    # keeps the new weights in a separately-registered, separately-saved module
    # instead of inside the PeftModel, where a newly-trainable submodule would
    # be silently dropped from every checkpoint (LoRA saves adapters only, and
    # this repo's apply_lora_to_model sets no modules_to_save).
    blur_tokens: torch.Tensor | None = None

    # Compact dynamics set for CROSS-ATTENTION injection: [B, M, dim] with
    # M = T (pooled) or T*H*W (factored) -- i.e. before the spatial broadcast
    # that `dynamics_tokens` applies. Only set when dynamics_injection is
    # "cross_attention"; the value-residual path ignores it.
    dynamics_kv: "torch.Tensor | list | None" = None   # a list of K tensors under multi-layer V-JEPA conditioning
    num_blocks: int = 1

    # Module refs filled once at install time (not reset per forward).
    modulation: object | None = None
    # The cross-attention stack is held HERE rather than as a WanBlockWrapper
    # submodule on purpose. Anything inside transformer.blocks becomes part of
    # the PeftModel, and get_peft_model freezes every base-model parameter --
    # which silently froze 89% of this stack (and, because its submodules are
    # named to_q/to_k/to_v/to_out, PEFT also substring-matched them against
    # lora_target_modules and injected adapters INTO it). PGA avoids the same
    # fate only because it is referenced from a plain non-nn.Module attention
    # processor. Keeping the stack off the module tree keeps it 100% trainable
    # and separately checkpointed.
    xattn: object | None = None

    def reset(self) -> None:
        self.enabled = False
        self.t = None
        self.dynamics_tokens = None
        self.blur_tokens = None
        self.dynamics_kv = None
