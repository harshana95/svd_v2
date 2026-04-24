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
    vec_in_dim: int = 768
    context_in_dim: int = 4096
    text_in_dim: int = 4096
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: tuple = (16, 56, 56)
    theta: int = 10_000
    qkv_bias: bool = True
    guidance_embed: bool = True


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# ---------------------------------------------------------------------------
# Timestep / control-index embeddings for adaptive modulation
# ---------------------------------------------------------------------------

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        act_map = {"silu": nn.SiLU, "mish": nn.Mish, "gelu": nn.GELU, "relu": nn.ReLU}
        self.act = act_map.get(act_fn, nn.SiLU)()

        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        self.post_act = None
        if post_act_fn is not None:
            self.post_act = act_map.get(post_act_fn, nn.SiLU)()

    def forward(self, sample: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        proj_dtype = self.linear_1.weight.dtype
        sample = sample.to(dtype=proj_dtype)
        if condition is not None:
            condition = condition.to(dtype=proj_dtype)
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


# ---------------------------------------------------------------------------
# Condition-level adaptive modulation (timestep + control_index)
# ---------------------------------------------------------------------------

class ConditionModulation(nn.Module):
    """Timestep & layer-adaptive modulation for DualConditionBranch outputs."""

    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim, bias=bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=dim)
        self.control_index_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=dim)

    def forward(
        self, x: torch.Tensor, timestep: torch.Tensor, control_index: torch.Tensor
    ) -> torch.Tensor:
        mod_dtype = self.linear.weight.dtype
        x = x.to(dtype=mod_dtype)
        timesteps_proj = self.time_proj(timestep * 1000)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=mod_dtype))

        if control_index.dim() == 0:
            control_index = control_index.repeat(x.shape[0])
        elif control_index.dim() == 1 and control_index.shape[0] != x.shape[0]:
            control_index = control_index.expand(x.shape[0])
        control_index = control_index.to(device=x.device, dtype=mod_dtype)
        control_index_proj = self.time_proj(control_index)
        control_index_emb = self.control_index_embedder(control_index_proj.to(dtype=mod_dtype))
        timesteps_emb = timesteps_emb + control_index_emb

        emb = self.linear(self.silu(timesteps_emb))
        shift_msa, scale_msa = emb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]


# ---------------------------------------------------------------------------
# SingleConditionBranch — lightweight ControlNet-like branch
# ---------------------------------------------------------------------------

class SingleConditionBranch(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams, controlnet_depth: int = 2):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.cond_in_channels = params.cond_in_channels
        self.cond_patch_size = params.cond_patch_size
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by "
                f"num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=list(params.axes_dim)
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if params.guidance_embed
            else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(controlnet_depth)
            ]
        )

        # ControlNet zero-initialized output projections
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(controlnet_depth):
            controlnet_block = nn.Linear(self.hidden_size, self.hidden_size)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)

        self.pos_embed_input = nn.Linear(
            self.cond_in_channels, self.hidden_size, bias=True
        )
        self.gradient_checkpointing = False

        # Pixel-space conditioning encoder
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(nn.Conv2d(16, 16, 3, padding=1)),
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        controlnet_cond: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> tuple:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        branch_dtype = self.img_in.weight.dtype
        img = img.to(dtype=branch_dtype)
        txt = txt.to(dtype=branch_dtype)
        y = y.to(dtype=branch_dtype)
        timesteps = timesteps.to(dtype=branch_dtype)
        if guidance is not None:
            guidance = guidance.to(dtype=branch_dtype)
        controlnet_cond = controlnet_cond.to(dtype=self.input_hint_block[0].weight.dtype)

        # FluxTransformer2DModel deprecated batched txt_ids, but this conditioner still
        # needs batched ids to concatenate with img_ids for positional embeddings.
        if txt_ids.ndim == 2:
            # [T, 3] -> [B, T, 3]
            txt_ids = repeat(txt_ids, "t c -> b t c", b=img.shape[0]).to(
                device=img.device, dtype=img.dtype
            )
        elif txt_ids.ndim != 3:
            raise ValueError(f"Expected txt_ids to be 2D or 3D, got {txt_ids.ndim}D")

        if img_ids.ndim != 3:
            raise ValueError(f"Expected img_ids to be 3D [B, S, 3], got {img_ids.ndim}D")

        # The LucidFlux conditioner uses its own RoPE layout, which is independent
        # from the Flux2 transformer's 4D positional ids.
        target_dims = len(self.pe_embedder.axes_dim)

        def _match_id_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
            if x.shape[-1] > target_dims:
                return x[..., :target_dims]
            if x.shape[-1] < target_dims:
                pad = torch.zeros(
                    *x.shape[:-1],
                    target_dims - x.shape[-1],
                    device=x.device,
                    dtype=x.dtype,
                )
                return torch.cat([x, pad], dim=-1)
            return x

        txt_ids = _match_id_dims(txt_ids, target_dims)
        img_ids = _match_id_dims(img_ids, target_dims)

        img = self.img_in(img)
        controlnet_cond = self.input_hint_block(controlnet_cond)
        if self.cond_patch_size == 1:
            controlnet_cond = rearrange(controlnet_cond, "b c h w -> b (h w) c")
        else:
            controlnet_cond = rearrange(
                controlnet_cond,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=self.cond_patch_size,
                pw=self.cond_patch_size,
            )
        controlnet_cond = self.pos_embed_input(
            controlnet_cond.to(dtype=self.pos_embed_input.weight.dtype)
        )
        img = img + controlnet_cond

        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        block_res_samples = ()
        for block in self.double_blocks:
            if self.training and self.gradient_checkpointing:
                def _double_fwd(img_, txt_, vec_, pe_, _block=block):
                    return _block(img=img_, txt=txt_, vec=vec_, pe=pe_)

                try:
                    img, txt = torch.utils.checkpoint.checkpoint(
                        _double_fwd, img, txt, vec, pe, use_reentrant=False
                    )
                except TypeError:
                    img, txt = torch.utils.checkpoint.checkpoint(
                        _double_fwd, img, txt, vec, pe
                    )
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            block_res_samples += (img,)

        controlnet_block_res_samples = ()
        for block_res_sample, controlnet_block in zip(
            block_res_samples, self.controlnet_blocks
        ):
            block_res_sample = controlnet_block(block_res_sample)
            controlnet_block_res_samples += (block_res_sample,)

        return controlnet_block_res_samples


# ---------------------------------------------------------------------------
# DualConditionBranch — two branches + adaptive modulation
# ---------------------------------------------------------------------------

class DualConditionBranch(nn.Module):
    def __init__(
        self,
        condition_branch_lq: nn.Module,
        condition_branch_pre: nn.Module,
        modulation_lq: nn.Module,
        modulation_pre: nn.Module,
        depth: int,
    ):
        super().__init__()
        self.lq = condition_branch_lq
        self.pre = condition_branch_pre
        self.modulation_lq = modulation_lq
        self.modulation_pre = modulation_pre
        self.depth = depth

    def forward(
        self,
        *,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        y: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: torch.Tensor,
        condition_cond_lq: torch.Tensor,
        condition_cond_pre: torch.Tensor,
        return_last_only: bool = False,
    ) -> list[torch.Tensor] | torch.Tensor:
        out_lq = self.lq(
            img=img,
            img_ids=img_ids,
            controlnet_cond=condition_cond_lq,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=timesteps,
            guidance=guidance,
        )
        out_pre = self.pre(
            img=img,
            img_ids=img_ids,
            controlnet_cond=condition_cond_pre,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=timesteps,
            guidance=guidance,
        )

        out = []
        last_out = None
        num_blocks = self.depth  # Flux double-stream depth
        for i in range(num_blocks // 2 + 1):
            for control_index, (lq, pre) in enumerate(zip(out_lq, out_pre)):
                control_index_t = torch.tensor(
                    control_index, device=timesteps.device, dtype=timesteps.dtype
                )
                lq = self.modulation_lq(lq, timesteps, i * 2 + control_index_t)
                if len(out) == num_blocks:
                    break
                pre = self.modulation_pre(pre, timesteps, i * 2 + control_index_t)
                combined = lq + pre
                if return_last_only:
                    last_out = combined
                else:
                    out.append(combined)
        return last_out if return_last_only else out


# ---------------------------------------------------------------------------
# ConditionBranchWithRedux — top-level trainable module
# ---------------------------------------------------------------------------

class ConditionBranchWithRedux(nn.Module):
    def __init__(
        self,
        condition_branch: DualConditionBranch,
        redux_image_encoder: nn.Module,
        vision_feature_adapter: nn.Module | None = None,
        text_state_adapter: nn.Module | None = None,
    ):
        super().__init__()
        self.condition_branch = condition_branch
        self.redux_image_encoder = redux_image_encoder
        self.vision_feature_adapter = (
            vision_feature_adapter if vision_feature_adapter is not None else nn.Identity()
        )
        self.text_state_adapter = text_state_adapter if text_state_adapter is not None else nn.Identity()
    def forward(
        self,
        *,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        y: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: torch.Tensor,
        condition_cond_lq: torch.Tensor,
        condition_cond_pre: torch.Tensor | None = None,
        vision_image_pre_fts: torch.Tensor | None = None,
        siglip_image_pre_fts: torch.Tensor | None = None,
        return_last_only: bool = False,
    ):
        imgtxt_redux = txt
        siglip_txt_ids = txt_ids
        if vision_image_pre_fts is None:
            vision_image_pre_fts = siglip_image_pre_fts

        if vision_image_pre_fts is not None:
            enc_dtype = self.redux_image_encoder.redux_up.weight.dtype
            adapted_vision_fts = self.vision_feature_adapter(
                vision_image_pre_fts.to(
                    device=txt.device,
                    dtype=getattr(
                        getattr(self.vision_feature_adapter, "weight", None),
                        "dtype",
                        enc_dtype,
                    ),
                )
            )
            adapted_text_fts = self.text_state_adapter(txt)
            
            image_embeds = self.redux_image_encoder(
                adapted_vision_fts.to(device=txt.device, dtype=enc_dtype)
            )["image_embeds"].to(dtype=txt.dtype)

            imgtxt_redux = torch.cat([adapted_text_fts, image_embeds], dim=1)

            # vision_image_pre_fts: [B, 576, 1024]
            # adapted_vision_fts: [B, 576, 1152]
            # image_embeds: [B, 576, 4096]

            # txt: [B, 512, 15360]
            # adapted_text_fts: [B, 512, 4096]
            # imgtxt_redux: [B, 1088, 4096]

            extra_seq_len = image_embeds.shape[1]
            if txt_ids.ndim == 2:
                _, channels = txt_ids.shape
                extra_ids = torch.zeros(
                    (extra_seq_len, channels),
                    device=txt_ids.device,
                    dtype=txt_ids.dtype,
                )
                imgtxt_redux_ids = torch.cat([txt_ids, extra_ids], dim=0)
            elif txt_ids.ndim == 3:
                batch_size, _, channels = txt_ids.shape
                extra_ids = torch.zeros(
                    (batch_size, extra_seq_len, channels),
                    device=txt_ids.device,
                    dtype=txt_ids.dtype,
                )
                imgtxt_redux_ids = torch.cat([txt_ids, extra_ids], dim=1)
            else:
                raise ValueError(
                    f"Expected txt_ids to be 2D or 3D, got {txt_ids.ndim}D"
                )

        block_res_samples = self.condition_branch(
            img=img,
            img_ids=img_ids,
            txt=adapted_text_fts,
            txt_ids=txt_ids,
            y=y,
            timesteps=timesteps,
            guidance=guidance,
            condition_cond_lq=condition_cond_lq,
            condition_cond_pre=condition_cond_pre
            if condition_cond_pre is not None
            else condition_cond_lq,
            return_last_only=return_last_only,
        )

        return block_res_samples, imgtxt_redux, imgtxt_redux_ids


# ---------------------------------------------------------------------------
# Weight loading / saving utilities
# ---------------------------------------------------------------------------

def load_single_condition_branch(
    params: FluxParams, device: torch.device | str, transformer=None
) -> SingleConditionBranch:
    with torch.device(device):
        controlnet = SingleConditionBranch(params)
    if transformer is not None:
        controlnet.load_state_dict(transformer.state_dict(), strict=False)
    return controlnet


def load_dual_condition_branch(
    params: FluxParams,
    lucidflux_weights: dict,
    device: torch.device | str,
    offload: bool = False,
    branch_dtype: torch.dtype = torch.bfloat16,
    modulation_dim: int = 3072,
) -> DualConditionBranch:
    load_device = "cpu" if offload else device

    condition_lq = load_single_condition_branch(params, load_device).to(branch_dtype)
    condition_lq.load_state_dict(lucidflux_weights["condition_lq"], strict=False)
    condition_lq = condition_lq.to(device)

    condition_pre = load_single_condition_branch(params, load_device).to(branch_dtype)
    condition_pre.load_state_dict(lucidflux_weights["condition_ldr"], strict=False)

    modulation_lq = ConditionModulation(dim=modulation_dim).to(branch_dtype)
    modulation_lq.load_state_dict(lucidflux_weights["modulation_lq"], strict=False)

    modulation_pre = ConditionModulation(dim=modulation_dim).to(branch_dtype)
    modulation_pre.load_state_dict(lucidflux_weights["modulation_ldr"], strict=False)

    target_device = "cpu" if offload else device
    return DualConditionBranch(
        condition_lq,
        condition_pre,
        modulation_lq=modulation_lq,
        modulation_pre=modulation_pre,
        depth=params.depth,
    ).to(target_device)


def load_condition_branch_with_redux(
    params: FluxParams,
    lucidflux_weights: dict,
    device: torch.device | str,
    offload: bool = False,
    branch_dtype: torch.dtype = torch.bfloat16,
    connector_dtype: torch.dtype = torch.float32,
    modulation_dim: int = 3072,
    vision_feature_dim: int = 1152,
) -> ConditionBranchWithRedux:
    from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder

    dual_condition_branch = load_dual_condition_branch(
        params=params,
        lucidflux_weights=lucidflux_weights,
        device=device,
        offload=offload,
        branch_dtype=branch_dtype,
        modulation_dim=modulation_dim,
    )
    redux_image_encoder = ReduxImageEncoder(txt_in_features=params.context_in_dim)
    redux_image_encoder.load_state_dict(
        lucidflux_weights["connector"], strict=False
    )
    redux_image_encoder.eval()
    target_device = "cpu" if offload else device
    redux_image_encoder.to(target_device).to(dtype=connector_dtype)
    redux_in_dim = redux_image_encoder.redux_up.in_features
    if vision_feature_dim == redux_in_dim:
        vision_feature_adapter = nn.Identity()
    else:
        vision_feature_adapter = nn.Linear(vision_feature_dim, redux_in_dim)
        adapter_state = lucidflux_weights.get("vision_feature_adapter")
        if adapter_state is not None:
            vision_feature_adapter.load_state_dict(adapter_state, strict=False)
        vision_feature_adapter.to(target_device).to(dtype=connector_dtype)

    return ConditionBranchWithRedux(
        dual_condition_branch,
        redux_image_encoder,
        vision_feature_adapter=vision_feature_adapter,
    )


def build_condition_branch_fresh(
    params: FluxParams,
    device: torch.device | str,
    branch_dtype: torch.dtype = torch.bfloat16,
    connector_dtype: torch.dtype = torch.float32,
    modulation_dim: int = 3072,
    vision_feature_dim: int = 1152,
) -> ConditionBranchWithRedux:
    """Build a fresh (randomly initialized) ConditionBranchWithRedux for training from scratch."""
    from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder

    condition_lq = SingleConditionBranch(params).to(device, dtype=branch_dtype)
    condition_pre = SingleConditionBranch(params).to(device, dtype=branch_dtype)
    modulation_lq = ConditionModulation(dim=modulation_dim).to(device, dtype=branch_dtype)
    modulation_pre = ConditionModulation(dim=modulation_dim).to(device, dtype=branch_dtype)

    dual_branch = DualConditionBranch(
        condition_lq, condition_pre, modulation_lq, modulation_pre, params.depth
    )
    redux_image_encoder = ReduxImageEncoder(txt_in_features=params.context_in_dim).to(device, dtype=connector_dtype)
    redux_in_dim = redux_image_encoder.redux_up.in_features
    if vision_feature_dim == redux_in_dim:
        vision_feature_adapter = nn.Identity()
    else:
        vision_feature_adapter = nn.Linear(vision_feature_dim, redux_in_dim).to(
            device, dtype=connector_dtype
        )

    if params.text_in_dim != params.context_in_dim:
        text_state_adapter = nn.Linear(params.text_in_dim, params.context_in_dim).to(device, dtype=branch_dtype)
    else:
        text_state_adapter = nn.Identity()
    return ConditionBranchWithRedux(
        dual_branch,
        redux_image_encoder,
        vision_feature_adapter=vision_feature_adapter,
        text_state_adapter=text_state_adapter,
    )


def export_lucidflux_state(model: ConditionBranchWithRedux) -> dict:
    dual_condition_branch = model.condition_branch
    state = {
        "condition_lq": dual_condition_branch.lq.state_dict(),
        "condition_ldr": dual_condition_branch.pre.state_dict(),
        "modulation_lq": dual_condition_branch.modulation_lq.state_dict(),
        "modulation_ldr": dual_condition_branch.modulation_pre.state_dict(),
        "connector": model.redux_image_encoder.state_dict(),
    }
    if not isinstance(model.vision_feature_adapter, nn.Identity):
        state["vision_feature_adapter"] = model.vision_feature_adapter.state_dict()
    return state
