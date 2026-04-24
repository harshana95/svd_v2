# LucidFlux conditioning components.
# Source: W2GenAI-Lab/LucidFlux (Apache 2.0)

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    MLPEmbedder,
    timestep_embedding,
)


@dataclass
class FluxParams:
    in_channels: int = 64
    cond_in_channels: int = 64
    cond_patch_size: int = 2
    vec_in_dim: int = 512*4
    context_in_dim: int = 4096
    text_in_dim: int = 15360
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: tuple = (16, 56, 56)
    theta: int = 10_000
    qkv_bias: bool = True
    guidance_embed: bool = True



# ---------------------------------------------------------------------------
# ConditionBranchFlux2 — top-level trainable module
# ---------------------------------------------------------------------------

class ConditionBranchFlux2(nn.Module):
    def __init__(
        self,
        vision_feature_adapter: nn.Module | None = None,
    ):
        super().__init__()
        self.vision_feature_adapter = (
            vision_feature_adapter if vision_feature_adapter is not None else nn.Identity()
        )
    def forward(
        self,
        *,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor,
        vision_image_pre_fts: torch.Tensor | None = None,
    ):
        if vision_image_pre_fts is not None:
            adapted_vision_fts = self.vision_feature_adapter(
                vision_image_pre_fts.to(
                    device=txt.device,
                    dtype=getattr(getattr(self.vision_feature_adapter, "weight", None),"dtype",),
                )
            )
            
            imgtxt_redux = torch.cat([txt, adapted_vision_fts], dim=1)

            extra_seq_len = adapted_vision_fts.shape[1]
            _, channels = txt_ids.shape
            extra_ids = torch.zeros(
                (extra_seq_len, channels),
                device=txt_ids.device,
                dtype=txt_ids.dtype,
            )
            imgtxt_redux_ids = torch.cat([txt_ids, extra_ids], dim=0)

            # imgtxt_redux = txt
            # imgtxt_redux_ids = txt_ids
        

        return imgtxt_redux, imgtxt_redux_ids


# ---------------------------------------------------------------------------
# Weight loading / saving utilities
# ---------------------------------------------------------------------------


def build_condition_branch_fresh(
    params: FluxParams,
    device: torch.device | str,
    connector_dtype: torch.dtype = torch.float32,
    vision_feature_dim: int = 1152,
):
    """Build a fresh (randomly initialized) ConditionBranchWithRedux for training from scratch."""
    
    redux_in_dim = params.text_in_dim
    if vision_feature_dim == redux_in_dim:
        vision_feature_adapter = nn.Identity()
    else:
        vision_feature_adapter = nn.Linear(vision_feature_dim, redux_in_dim).to(
            device, dtype=connector_dtype
        )

    return ConditionBranchFlux2(
        vision_feature_adapter=vision_feature_adapter
    )
