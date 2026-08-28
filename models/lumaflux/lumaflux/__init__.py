"""LumaFlux: physically- & perceptually-guided diffusion transformer for SDR->HDR ITM.

Paper-exact re-implementation of arXiv 2604.02787 on a frozen FLUX.1-dev backbone.
"""

from .model import LumaFlux, LumaFluxConfig
from .losses import LumaFluxLoss
from .dataset import PairedHDRDataset, SyntheticPairDataset

__all__ = [
    "LumaFlux",
    "LumaFluxConfig",
    "LumaFluxLoss",
    "PairedHDRDataset",
    "SyntheticPairDataset",
]
