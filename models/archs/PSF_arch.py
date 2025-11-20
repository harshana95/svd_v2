import math

import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

class PSF_MLP_config(PretrainedConfig):
    def __init__(self, in_features=4, kernel_size=65, hidden_features=64, hidden_layers=3, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.kernel_size = kernel_size
        self.out_features = 3 * (kernel_size ** 2)
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers

class PSF_MLP_arch(PreTrainedModel):
    """All-linear layer. This network suits for low-k intensity/amplitude PSF function prediction."""
    config_class = PSF_MLP_config
    def __init__(self, config):
        super().__init__(config)
        self.kernel_size = config.kernel_size

        layers = [
            nn.Linear(config.in_features, config.hidden_features // 4, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_features // 4, config.hidden_features, bias=True),
            nn.ReLU(inplace=True),
        ]

        for _ in range(config.hidden_layers):
            layers.extend(
                [
                    nn.Linear(config.hidden_features, config.hidden_features, bias=True),
                    nn.ReLU(inplace=True),
                ]
            )

        layers.extend(
            [nn.Linear(config.hidden_features, config.out_features, bias=True), nn.Sigmoid()]
        )
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        # x = x.view(x.shape[0], 3, self.kernel_size*self.kernel_size)
        # x = F.normalize(x, p=1, dim=-1)
        x = x.view(x.shape[0], 3, self.kernel_size, self.kernel_size)
        return x
    
class PSF_MLPConv_config(PretrainedConfig):
    def __init__(self, in_features=4, kernel_size=65, channels=3, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.kernel_size = kernel_size
        self.channels = channels
        self.activation = activation

class PSF_MLPConv_arch(PreTrainedModel):
    config_class = PSF_MLPConv_config
    """MLP encoder + convolutional decoder proposed in "Differentiable Compound Optics and Processing Pipeline Optimization for End-To-end Camera Design". This network suits for high-k intensity/amplitude PSF function prediction.

    Input:
        in_features (int): Input features, shape of [batch_size, in_features].
        ks (int): The size of the output image.
        channels (int): The number of output channels. Defaults to 3.
        activation (str): The activation function. Defaults to 'relu'.

    Output:
        x (Tensor): The output image. Shape of [batch_size, channels, ks, ks].
    """

    def __init__(self, config):
        super().__init__(config)
        in_features = config.in_features
        ks = config.kernel_size
        channels = config.channels
        activation = config.activation

        self.ks_mlp = min(ks, 32)
        if ks > 32:
            assert ks % 32 == 0, "ks must be 32n"
            upsample_times = int(math.log(ks / 32, 2))

        linear_output = channels * self.ks_mlp**2
        self.ks = ks
        self.channels = channels

        # MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, linear_output),
        )

        # Conv decoder
        conv_layers = []
        conv_layers.append(
            nn.ConvTranspose2d(channels, 64, kernel_size=3, stride=1, padding=1)
        )
        conv_layers.append(nn.ReLU())
        for _ in range(upsample_times):
            conv_layers.append(
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Upsample(scale_factor=2))

        conv_layers.append(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        conv_layers.append(nn.ReLU())
        conv_layers.append(
            nn.ConvTranspose2d(64, channels, kernel_size=3, stride=1, padding=1)
        )
        self.decoder = nn.Sequential(*conv_layers)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        # Encode the input using the MLP
        encoded = self.encoder(x)

        # Reshape the output from the MLP to feed to the CNN
        decoded_input = encoded.view(
            -1, self.channels, self.ks_mlp, self.ks_mlp
        )  # reshape to (batch_size, channels, height, width)

        # Decode the output using the CNN
        decoded = self.decoder(decoded_input)
        # decoded = self.activation(decoded)

        # This normalization only works for PSF network
        decoded = F.normalize(decoded, p=1, dim=[-1, -2])

        return decoded
