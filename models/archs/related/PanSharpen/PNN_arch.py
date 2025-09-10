import os
import torch.nn as nn
import torch
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig

class PNN_config(PretrainedConfig):
    model_type = "PNN_arch"
    def __init__(self, sf=1, mul_channel=3, pan_channel=1, **kwargs):
        super().__init__(**kwargs)
        self.sf = sf
        self.mul_channel = mul_channel
        self.pan_channel = pan_channel
        
class PNN_arch(PreTrainedModel):
    config_class = PNN_config
    def __init__(self, config):
        super().__init__(config)
        self.sf = config.sf
        self.conv_1 = nn.Conv2d(in_channels=config.mul_channel+config.pan_channel, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=config.mul_channel, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
    
    def forward(self, x, y):
        if self.sf != 1:
            x = torch.nn.functional.interpolate(x, scale_factor=self.sf, mode='bicubic', align_corners=True)
        in_put = torch.cat([x,y], -3)
        fea = self.relu(self.conv_1(in_put))  
        fea =  self.relu(self.conv_2(fea))
        out = self.conv_3(fea)
        return out

