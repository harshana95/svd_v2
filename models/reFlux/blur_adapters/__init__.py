"""LumaFlux-style blur-map + JEPA adapters for Flux2."""

from .context import BlurContext
from .coupler import BlurResidualCoupler
from .features import BlurFeatures, BlurMapExtractor, map_to_tokens, tokens_to_grid
from .jepa_connector import JEPAPerceptualEncoder
from .modulation import BlockModulation, TimestepLayerModulation
from .pcm import PCMModule
from .pga import Flux2PGAAttnProcessor, Flux2ParallelPGAAttnProcessor, PGAModule
from .wrappers import Flux2DoubleBlockWrapper, Flux2SingleBlockWrapper, install_blur_adapters

__all__ = [
    "BlurContext",
    "BlurFeatures",
    "BlurMapExtractor",
    "BlurResidualCoupler",
    "BlockModulation",
    "Flux2DoubleBlockWrapper",
    "Flux2PGAAttnProcessor",
    "Flux2ParallelPGAAttnProcessor",
    "Flux2SingleBlockWrapper",
    "JEPAPerceptualEncoder",
    "PCMModule",
    "PGAModule",
    "TimestepLayerModulation",
    "install_blur_adapters",
    "map_to_tokens",
    "tokens_to_grid",
]
