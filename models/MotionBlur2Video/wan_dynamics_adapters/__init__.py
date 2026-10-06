"""V-JEPA dynamics-token injection into Wan 2.2's self-attention (PGA-style)."""

from .context import WanDynamicsContext
from .features import (
    JEPAFactoredConnector,
    MultiLayerDynamicsConnector,
    WanBlurCondEmbed,
    broadcast_frame_tokens_to_video_tokens,
    pool_dynamics_to_frame_tokens,
    resample_dynamics_grid,
)
from .wan_pga import WanDynamicsCrossAttnStack, WanPGAAttnProcessor, WanPGAModule
from .wan_wrappers import WanBlockWrapper, install_wan_dynamics_adapters

__all__ = [
    "WanDynamicsContext",
    "WanPGAAttnProcessor",
    "WanPGAModule",
    "WanDynamicsCrossAttnStack",
    "WanBlockWrapper",
    "JEPAFactoredConnector",
    "MultiLayerDynamicsConnector",
    "WanBlurCondEmbed",
    "broadcast_frame_tokens_to_video_tokens",
    "install_wan_dynamics_adapters",
    "pool_dynamics_to_frame_tokens",
    "resample_dynamics_grid",
]
