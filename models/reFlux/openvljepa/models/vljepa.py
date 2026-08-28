"""OpenVL-JEPA inference wrapper (no training loss)."""

import torch
import torch.nn as nn

from .encoder import VJEPA2Encoder
from .predictor import Predictor
from .y_encoder import YEncoder


class OpenVLJEPA(nn.Module):
    def __init__(
        self,
        encoder_cfg: dict,
        y_encoder_cfg: dict,
        predictor_cfg: dict,
        torch_dtype=None,
    ):
        super().__init__()

        self.x_encoder = VJEPA2Encoder(
            model_name=encoder_cfg["model_name"],
            output_dim=encoder_cfg.get("output_dim", None),
            frozen=encoder_cfg.get("frozen", True),
        )
        vis_dim = self.x_encoder.output_dim

        self.y_encoder = YEncoder(
            model_name=y_encoder_cfg["model_name"],
            proj_dim=y_encoder_cfg["proj_dim"],
            torch_dtype=torch_dtype,
        )

        self.predictor = Predictor(
            llama_name=predictor_cfg["llama_name"],
            n_keep_layers=predictor_cfg.get("n_keep_layers", 8),
            vis_dim=vis_dim,
            proj_dim=predictor_cfg["proj_dim"],
            disable_causal_mask=predictor_cfg.get("disable_causal_mask", True),
            torch_dtype=torch_dtype,
        )

        assert y_encoder_cfg["proj_dim"] == predictor_cfg["proj_dim"]

    def predict_embedding(
        self,
        pixel_values: torch.Tensor,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted target embedding S_Y: (B, proj_dim)."""
        vis_embeds = self.x_encoder(pixel_values)
        return self.predictor(vis_embeds, query_ids, query_mask)

    def forward(
        self,
        pixel_values: torch.Tensor,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.predict_embedding(pixel_values, query_ids, query_mask)
