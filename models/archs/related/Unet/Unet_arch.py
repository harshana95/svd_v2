
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from diffusers import UNet2DModel
from einops import rearrange

from transformers import PreTrainedModel, PretrainedConfig


class Unet_config(PretrainedConfig):
    model_type = "Unet_arch"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

class Unet_arch(PreTrainedModel):
    config_class = Unet_config

    def __init__(self, config):
        super().__init__(config)
        self.model = UNet2DModel(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            attention_head_dim=config.attention_head_dim,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
            block_out_channels=config.block_out_channels,
        )
        self.init_weights()

    def forward(self, x, *args):
        if len(args) > 0:
            x = torch.cat([x, *args], dim=1)
        return self.model(x, timestep=1).sample
