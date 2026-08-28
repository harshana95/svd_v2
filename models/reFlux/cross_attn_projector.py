"""Cross-attention projectors mapping encoder features into Flux 2 text conditioning tokens."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_helpers import get_clip_num_patches
from .dino_helpers import get_dino_num_patches
from models.LucidFlux.jepa_helpers import get_vjepa_num_patches

logger = logging.getLogger(__name__)


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


class VaeToTextTargetBridge(nn.Module):
    """Resample VAE packed tokens to vision patch grid and project into Flux text space."""

    def __init__(
        self,
        latent_token_dim: int,
        text_dim: int,
        lat_h: int,
        lat_w: int,
        target_h: int,
        target_w: int,
    ):
        super().__init__()
        self.lat_h = lat_h
        self.lat_w = lat_w
        self.target_h = target_h
        self.target_w = target_w
        self.proj = nn.Linear(latent_token_dim, text_dim)

    def forward(self, vae_tokens: torch.Tensor) -> torch.Tensor:
        bsz, _, dim = vae_tokens.shape
        x = vae_tokens.reshape(bsz, self.lat_h, self.lat_w, dim).permute(0, 3, 1, 2)
        x = F.interpolate(
            x,
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        )
        x = x.permute(0, 2, 3, 1).reshape(bsz, -1, dim)
        return self.proj(x)


def resample_text_tokens(text_tokens: torch.Tensor, num_slots: int) -> torch.Tensor:
    """Resample [B, S, D] text tokens along sequence length to num_slots."""
    if text_tokens.shape[1] == num_slots:
        return text_tokens
    x = text_tokens.permute(0, 2, 1)
    x = F.interpolate(x, size=num_slots, mode="linear", align_corners=False)
    return x.permute(0, 2, 1)


class DualHeadPatchCrossAttnProjector(nn.Module):
    """Shared trunk with separate text and VAE output heads for Flux conditioning."""

    def __init__(
        self,
        embed_dim: int,
        text_seq_dim: int,
        text_num_slots: int,
        vae_seq_dim: int,
        vae_num_slots: int,
        hidden_dim: int = 2048,
        num_heads: int = 8,
    ):
        super().__init__()
        self.text_num_slots = text_num_slots
        self.vae_num_slots = vae_num_slots

        self.embed_proj = nn.Linear(embed_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.text_norm_q = nn.LayerNorm(hidden_dim)
        self.vae_norm_q = nn.LayerNorm(hidden_dim)

        self.text_query_tokens = nn.Parameter(
            torch.randn(1, text_num_slots, hidden_dim) * 0.02
        )
        self.vae_query_tokens = nn.Parameter(
            torch.randn(1, vae_num_slots, hidden_dim) * 0.02
        )
        self.text_seq_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, text_seq_dim),
        )
        self.vae_seq_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vae_seq_dim),
        )

    def _encode_kv(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.norm_k(self.embed_proj(x))

    def forward(self, x: torch.Tensor, heads: str = "both"):
        """
        Args:
            x: [B, embed_dim] or [B, N, embed_dim]
            heads: "both", "text", or "vae"
        Returns:
            dict with "text" and/or "vae" keys when heads="both",
            otherwise a single [B, num_slots, seq_dim] tensor.
        """
        if x.ndim == 2:
            bsz = x.shape[0]
        else:
            bsz = x.shape[0]
        kv = self._encode_kv(x)
        outputs = {}
        if heads in ("both", "text"):
            q = self.text_norm_q(self.text_query_tokens.expand(bsz, -1, -1))
            attn_out, _ = self.cross_attn(q, kv, kv)
            outputs["text"] = self.text_seq_out(attn_out)
        if heads in ("both", "vae"):
            q = self.vae_norm_q(self.vae_query_tokens.expand(bsz, -1, -1))
            attn_out, _ = self.cross_attn(q, kv, kv)
            outputs["vae"] = self.vae_seq_out(attn_out)
        if heads == "text":
            return outputs["text"]
        if heads == "vae":
            return outputs["vae"]
        return outputs


def build_dual_head_projector(
    embed_dim: int,
    text_seq_dim: int,
    text_num_slots: int,
    vae_seq_dim: int,
    vae_num_slots: int,
    hidden_dim: int = 2048,
) -> DualHeadPatchCrossAttnProjector:
    return DualHeadPatchCrossAttnProjector(
        embed_dim=embed_dim,
        text_seq_dim=text_seq_dim,
        text_num_slots=text_num_slots,
        vae_seq_dim=vae_seq_dim,
        vae_num_slots=vae_num_slots,
        hidden_dim=hidden_dim,
    )


def resolve_transformer_context_dim(opt, transformer_config, default: int = 15360) -> int:
    text_in_dim = opt.get("text_in_dim") if hasattr(opt, "get") else None
    if text_in_dim in (None, "~"):
        text_in_dim = transformer_config.get("joint_attention_dim")
    if text_in_dim in (None, "~"):
        text_in_dim = default
    return int(text_in_dim)


def _resolve_patch_projector_override(opt):
    override = opt.get("patch_projector_num_slots") if hasattr(opt, "get") else None
    if override not in (None, "~"):
        return int(override)
    return None


def resolve_vjepa_projector_slots(opt, hub_entry: str, image_size: int) -> int:
    override = _resolve_patch_projector_override(opt)
    if override is not None:
        return override
    return get_vjepa_num_patches(hub_entry, image_size)


def resolve_dino_projector_slots(opt, model_name: str, image_size: int) -> int:
    override = _resolve_patch_projector_override(opt)
    if override is not None:
        return override
    return get_dino_num_patches(model_name, image_size)


def resolve_clip_projector_slots(opt, model_name: str, image_size: int) -> int:
    override = _resolve_patch_projector_override(opt)
    if override is not None:
        return override
    return get_clip_num_patches(model_name, image_size)


def build_flux_patch_projector(
    embed_dim: int,
    transformer_context_dim: int,
    num_slots: int,
    hidden_dim: int = 2048,
) -> PatchCrossAttnProjector:
    return PatchCrossAttnProjector(
        embed_dim=embed_dim,
        seq_dim=transformer_context_dim,
        num_slots=num_slots,
        hidden_dim=hidden_dim,
    )


def load_jepa_projector_weights(projector: nn.Module, weights_path: str):
    """Load projector weights; strict when shapes match, partial fallback for legacy checkpoints."""
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model_sd = projector.state_dict()
    try:
        projector.load_state_dict(state, strict=True)
        logger.info(
            "Loaded %d/%d jepa_projector keys from %s (strict)",
            len(model_sd),
            len(model_sd),
            weights_path,
        )
    except RuntimeError:
        compatible = {
            k: v for k, v in state.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        missing = [k for k in model_sd if k not in compatible]
        projector.load_state_dict(compatible, strict=False)
        logger.warning(
            "Partial load %d/%d jepa_projector keys from %s (old checkpoint? skipped: %s)",
            len(compatible),
            len(model_sd),
            weights_path,
            missing,
        )
