import einops
import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from diffusers.models.autoencoders.vae import Encoder, DiagonalGaussianDistribution
from diffusers.configuration_utils import ConfigMixin, register_to_config

class Encoder_config(PretrainedConfig):
    model_type = "Encoder_arch"

    def __init__(self,in_channels=3,out_channels=3, down_block_types=[], block_out_channels=[64,], **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_block_types = down_block_types
        self.block_out_channels = block_out_channels
        

    
class Encoder_arch(PreTrainedModel):
    config_class = Encoder_config

    def __init__(self, opt):
        super().__init__(opt)
        self.encoder = Encoder(
                in_channels=opt.in_channels,
                out_channels=opt.out_channels,
                down_block_types=opt.down_block_types,
                block_out_channels=opt.block_out_channels,
                layers_per_block=2,
                act_fn= "silu",
                norm_num_groups=32,
                double_z=True,
                mid_block_add_attention=True,
            )
        self.quant_conv = nn.Conv2d(2 * opt.out_channels, 2 * opt.out_channels, 1)

    def forward(self, x):
        enc = self.encoder(x)
        enc = self.quant_conv(enc)
        posterior = DiagonalGaussianDistribution(enc)
        return posterior.sample()