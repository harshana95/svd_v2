"""Resample a V-JEPA space-time token grid into dynamics tokens for Wan's
attention, via two alternative pipelines (selected by
`dynamics_connector_type` on Wan22VJEPADynamics_model):

- "pooled" (the original pipeline, `pool_dynamics_to_frame_tokens` +
  `broadcast_frame_tokens_to_video_tokens` + `JEPAPerceptualEncoder`, the
  latter imported from models.reFlux.blur_adapters): global-average-pools
  V-JEPA's spatial grid away entirely (one vector per frame, broadcast
  uniformly across all of that frame's spatial tokens), then projects via a
  single dense Linear. Kept here, unchanged, so checkpoints already trained
  with it (e.g. Stage A, the ablation, the first LoRA runs, the
  deblur-prefix run) remain re-loadable -- a checkpoint's saved module is
  keyed by class name, so switching the *default* pipeline out from under
  an old checkpoint silently leaves its connector randomly initialized
  instead of loading the real trained weights (a real mistake made and
  caught during this project, see findings/Wan22VJEPADynamics_findings.md).
- "factored" (the newer default for fresh runs): resample_dynamics_grid
  preserves V-JEPA's spatial grid (bilinearly resized, in either direction,
  to match Wan's own post-patch H,W) instead of pooling it away, and
  JEPAFactoredConnector projects via two small linear heads (per-channel,
  per-attention-head) combined by outer product instead of one dense
  matrix -- see each function's own docstring for the full rationale.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_dynamics_to_frame_tokens(
    jepa_grid: torch.Tensor,
    vjepa_num_tubelets: int,
    vjepa_grid_h: int,
    vjepa_grid_w: int,
    target_num_frames: int,
) -> torch.Tensor:
    """jepa_grid: [B, vjepa_num_tubelets*vjepa_grid_h*vjepa_grid_w, D] -> [B, target_num_frames, D]."""
    b, n, d = jepa_grid.shape
    expected_n = vjepa_num_tubelets * vjepa_grid_h * vjepa_grid_w
    if n != expected_n:
        raise ValueError(
            f"pool_dynamics_to_frame_tokens: got N={n} tokens, expected "
            f"vjepa_num_tubelets*vjepa_grid_h*vjepa_grid_w={expected_n}"
        )
    grid = jepa_grid.reshape(b, vjepa_num_tubelets, vjepa_grid_h, vjepa_grid_w, d)
    per_tubelet = grid.mean(dim=(2, 3))  # [B, vjepa_num_tubelets, D] -- global spatial avg-pool

    # Temporal resample vjepa_num_tubelets -> target_num_frames.
    per_tubelet_t = per_tubelet.transpose(1, 2)  # [B, D, vjepa_num_tubelets]
    if target_num_frames == vjepa_num_tubelets:
        resampled = per_tubelet_t
    else:
        resampled = F.interpolate(
            per_tubelet_t.float(), size=target_num_frames, mode="linear", align_corners=False
        ).to(per_tubelet_t.dtype)
    return resampled.transpose(1, 2)  # [B, target_num_frames, D]


def broadcast_frame_tokens_to_video_tokens(
    dynamics_per_frame: torch.Tensor, tokens_per_frame: int
) -> torch.Tensor:
    """[B, T, D] -> [B, T*tokens_per_frame, D], matching Wan's `(T,H,W)`-flattened token order."""
    b, t, d = dynamics_per_frame.shape
    return dynamics_per_frame.unsqueeze(2).expand(-1, -1, tokens_per_frame, -1).reshape(b, t * tokens_per_frame, d)


def resample_dynamics_grid(
    jepa_grid: torch.Tensor,
    vjepa_num_tubelets: int,
    vjepa_grid_h: int,
    vjepa_grid_w: int,
    target_num_frames: int,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """[B, vjepa_num_tubelets*vjepa_grid_h*vjepa_grid_w, D] ->
    [B, target_num_frames, target_h, target_w, D].

    Temporal axis resampled via linear interpolation (tubelets ->
    target_num_frames), spatial axes resized via bilinear interpolation
    (vjepa_grid_h x vjepa_grid_w -> target_h x target_w, upsampling or
    downsampling as needed -- a no-op when the sizes already match). Both
    resamples are no-ops when the source/target sizes are equal.
    """
    b, n, d = jepa_grid.shape
    expected_n = vjepa_num_tubelets * vjepa_grid_h * vjepa_grid_w
    if n != expected_n:
        raise ValueError(
            f"resample_dynamics_grid: got N={n} tokens, expected "
            f"vjepa_num_tubelets*vjepa_grid_h*vjepa_grid_w={expected_n}"
        )
    grid = jepa_grid.reshape(b, vjepa_num_tubelets, vjepa_grid_h, vjepa_grid_w, d)

    if target_num_frames == vjepa_num_tubelets:
        grid_t = grid
    else:
        # Move the tubelet axis last so it's the only one F.interpolate
        # touches; everything else (B, grid_h, grid_w, D) rides along as
        # "batch".
        moved = grid.permute(0, 2, 3, 4, 1).reshape(
            b * vjepa_grid_h * vjepa_grid_w * d, 1, vjepa_num_tubelets
        )
        resampled = F.interpolate(
            moved.float(), size=target_num_frames, mode="linear", align_corners=False
        )
        grid_t = resampled.reshape(b, vjepa_grid_h, vjepa_grid_w, d, target_num_frames).permute(
            0, 4, 1, 2, 3
        ).to(grid.dtype)

    if vjepa_grid_h == target_h and vjepa_grid_w == target_w:
        return grid_t

    spatial = grid_t.permute(0, 1, 4, 2, 3).reshape(
        b * target_num_frames, d, vjepa_grid_h, vjepa_grid_w
    )
    resized = F.interpolate(spatial.float(), size=(target_h, target_w), mode="bilinear", align_corners=False)
    return resized.reshape(b, target_num_frames, d, target_h, target_w).permute(0, 1, 3, 4, 2).to(grid_t.dtype)


class JEPAFactoredConnector(nn.Module):
    """Projects V-JEPA features [..., jepa_dim] to Wan's per-block value
    width [..., num_heads*head_dim] via two small linear heads -- one
    predicting a per-channel (head_dim) pattern, one predicting a
    per-attention-head scale -- combined by an outer product, instead of one
    dense [jepa_dim, num_heads*head_dim] matrix. This matches how the output
    is actually consumed (added to `value` then split into num_heads x
    head_dim) rather than treating it as an arbitrary flat vector, and uses
    far fewer parameters (jepa_dim*(head_dim+num_heads) vs.
    jepa_dim*num_heads*head_dim)."""

    def __init__(self, jepa_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_head_dim = nn.Linear(jepa_dim, head_dim)
        self.to_num_heads = nn.Linear(jepa_dim, num_heads)

    def forward(self, jepa_tokens: torch.Tensor) -> torch.Tensor:
        jepa_tokens = jepa_tokens.float()
        per_channel = self.to_head_dim(jepa_tokens)  # [..., head_dim]
        per_head = self.to_num_heads(jepa_tokens)  # [..., num_heads]
        out = per_head.unsqueeze(-1) * per_channel.unsqueeze(-2)  # [..., num_heads, head_dim]
        return out.flatten(-2)  # [..., num_heads*head_dim]


class MultiLayerDynamicsConnector(nn.Module):
    """K V-JEPA depths -> K cross-attention KV sets.

    Input [B, N, D*K] (K layers concatenated on the feature dim, as produced by
    extract_vjepa_tokens_layers); each slot gets its OWN LayerNorm (intermediate V-JEPA blocks are the
    raw, un-normed residual stream, so their scale differs from the final normed output) and its own
    Linear(D -> out_dim). Returns a list of K tensors [B, N, out_dim]; the Wan block wrapper routes
    group g of blocks to KV set g (shallow V-JEPA depth -> early Wan blocks, deep -> late).
    """

    def __init__(self, jepa_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.jepa_dim = jepa_dim
        self.num_layers = num_layers
        self.norms = nn.ModuleList([nn.LayerNorm(jepa_dim) for _ in range(num_layers)])
        self.proj = nn.ModuleList([nn.Linear(jepa_dim, out_dim) for _ in range(num_layers)])

    def forward(self, tokens: torch.Tensor):
        tokens = tokens.float()
        chunks = tokens.split(self.jepa_dim, dim=-1)
        if len(chunks) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} x {self.jepa_dim} features, got {tokens.shape[-1]}")
        return [self.proj[k](self.norms[k](c)) for k, c in enumerate(chunks)]


class WanBlurCondEmbed(nn.Module):
    """Embeds a blurred-input latent video into Wan's post-patch token space.

    Mirrors Wan's own `patch_embedding` exactly (Conv3d, kernel == stride ==
    patch_size, z_dim -> inner_dim) and is **zero-initialized**, so a model that
    installs it starts bit-identical to one that never had it -- which is what
    makes a fine-tune resumed from a pre-existing checkpoint a valid continuation
    rather than a perturbed restart.

    Why a separate conv instead of widening patch_embedding's in_channels and
    concatenating: a conv is linear in its input channels, so
    `patch_embed(cat([x, b]))` with the new half zero-init is identically
    `patch_embed(x) + blur_embed(b)`. The separate module is registered and saved
    like any other trainable module here, whereas widening patch_embedding would
    bury new weights inside the PeftModel, which saves adapter tensors only.
    """

    def __init__(self, z_dim: int, inner_dim: int, patch_size):
        super().__init__()
        p_t, p_h, p_w = patch_size
        self.proj = nn.Conv3d(z_dim, inner_dim, kernel_size=(p_t, p_h, p_w), stride=(p_t, p_h, p_w))
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, blur_latent: torch.Tensor) -> torch.Tensor:
        """blur_latent: [B, z, T_lat, H_lat, W_lat] -> [B, T'*H'*W', inner_dim],
        flattened in the same order Wan patchifies (t, then h, then w)."""
        x = self.proj(blur_latent)                      # [B, inner, T', H', W']
        b, c = x.shape[0], x.shape[1]
        return x.reshape(b, c, -1).transpose(1, 2)      # [B, N, inner]


class DynamicsPredictorWrapper(nn.Module):
    """Holds the (PEFT-adapted) V-JEPA dynamics PREDICTOR under a UNIQUE class name.

    base_model's save/load hooks key checkpoint folders on `model._get_name()` plus an occurrence counter, walking
    `self.models` in reverse. A bare PeftModel here would collide with the TRANSFORMER's PeftModel: on load, Stage A's
    transformer LoRA could be poured into the predictor (or vice-versa) with no error -- the module-swap failure mode
    recorded in memory (module_swap_checkpoint_compat) and the same trick used for WanDynamicsCrossAttnStack. Only
    used when `dynamics_predictor_trainable: true` (joint fine-tune); the frozen predictor stays a bare PeftModel.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)
