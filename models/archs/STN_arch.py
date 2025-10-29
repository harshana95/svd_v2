import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

class STN_config(PretrainedConfig):
    def __init__(self, input_channels=6, **kwargs):
        super().__init__(**kwargs)
        self.input_channels = input_channels

# The main LKPN (Spatially-varying Kernel Prediction Network)
class STN_arch(PreTrainedModel):
    config_class = STN_config
    def __init__(self, config):
        super().__init__(config)
        # Define the localization network. This is an example, it can be
        # any network that regresses the transformation parameters.
        self.localization = nn.Sequential(
            nn.Conv2d(config.input_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 2x3 affine matrix. The input size is
        # now calculated for a 512x512 image:
        # 512 -> conv7 -> 506 -> maxpool2 -> 253 -> conv5 -> 249 -> maxpool2 -> 124
        # So the flattened size is 10 channels * 124 * 124
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 124 * 124, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights with an identity transformation.
        # This helps the model start with no transformation and learn
        # to apply a transformation only when needed.
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # Localisation Network
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 124 * 124)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Grid Generator
        # F.affine_grid generates the sampling grid.
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        
        # Sampler
        # F.grid_sample performs the bilinear interpolation.
        x = F.grid_sample(x, grid, align_corners=True)

        return x
