import logging

import einops
import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from models.archs.related.NAFNet.NAFNet_arch import NAFBlock, NAFNet



class MSArrayIR_config(PretrainedConfig):
    model_type = "MSArrayIR_arch"

    def __init__(self, array_size=[1,1],  **kwargs):
        super().__init__(**kwargs)
        self.array_size = array_size
        


class MSArrayIR_arch(PreTrainedModel):
    config_class = MSArrayIR_config
    def __init__(self, config):
        super().__init__(config)
        self.n = config.array_size[0]*config.array_size[1]
        self.c = config.img_channel
        assert config.width > self.c*self.n, "Width must be greater than the number of channels times the number of elements in the array."
        self.intro = nn.Conv2d(in_channels=self.c*self.n, out_channels=config.width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        chan = config.width
        # self.up = nn.Sequential(
        #             nn.Conv2d(chan, chan * config.downscale_factor[0], 1, bias=False),
        #             nn.PixelShuffle(config.downscale_factor[0])
        #         )
        # chan = chan // config.downscale_factor[0]
        self.decoder = nn.Sequential(*[NAFBlock(chan) for _ in range(5)])
        # self.net = NAFNet(img_channel=chan, width=chan, middle_blk_num=config.middle_blk_num, enc_blk_nums=config.enc_blk_nums, dec_blk_nums=config.dec_blk_nums)
        self.ending = nn.Conv2d(in_channels=chan, out_channels=self.c, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

    def forward(self, x):
        x = einops.rearrange(x, 'b n c h w -> b (n c) h w')
        x = self.intro(x)
        # x = self.up(x)
        x = self.decoder(x)
        x = self.ending(x)
        return x

class MSArrayClassifier_config(PretrainedConfig):
    model_type = "MSArrayClassifier_arch"

    def __init__(self, image_shape=[28,28], n_classes=1, array_size=[1,1], downscale_factor=[1,1], **kwargs):
        super().__init__(**kwargs)
        self.image_shape = image_shape
        self.n_classes = n_classes
        self.array_size = array_size
        self.downscale_factor = downscale_factor


class MSArrayClassifier_arch(PreTrainedModel):
    config_class = MSArrayClassifier_config
    def __init__(self, config):
        super().__init__(config)
        self.image_shape = config.image_shape
        self.n = config.array_size[0]*config.array_size[1]
        self.c = config.img_channel

        chan = config.width
        self.conv1 = nn.Conv2d(in_channels=self.c*self.n, out_channels=chan, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=chan, out_channels=chan*2, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(chan*2*int(self.image_shape[0]/4)*int(self.image_shape[1]/4), 128)
        self.fc2 = nn.Linear(128, config.n_classes)

    def forward(self, x):
        x = einops.rearrange(x, 'b n c h w -> b (n c) h w')
        # print(x.shape)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        # print(x.shape)
        x = self.maxpool1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        # print(x.shape)
        x = self.maxpool2(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.sigmoid(x)
        return x
