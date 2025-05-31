import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class Vector2ImageNet_config(PretrainedConfig):
    def __init__(self, input_dim=3, latent_dim=256, output_channels=3, image_size=64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.image_size = image_size

class Vector2ImageNet_arch(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        input_dim = config.input_dim
        latent_dim = config.latent_dim
        output_channels = config.output_channels
        image_size = config.image_size

        self.latent_dim = latent_dim
        self.image_size = image_size

        # Fully connected layers to map 3D vector to latent space
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * (image_size // 8) * (image_size // 8)),
            nn.ReLU()
        )

        # Convolutional decoder to generate the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),  # Upsample x2
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample x2
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),  # Upsample x2
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        # Pass through fully connected layers
        x = self.fc(x)
        # Reshape to (batch_size, latent_dim, height, width)
        x = x.view(x.size(0), self.latent_dim, self.image_size // 8, self.image_size // 8)
        # Pass through convolutional decoder
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    model = Vector2ImageNet_arch(Vector2ImageNet_config(input_dim=3, latent_dim=256, output_channels=3, image_size=64))
    input_vector = torch.randn(8, 3)  # Batch of 8, 3D vectors
    output_image = model(input_vector)  # Output shape: (8, 3, 64, 64)
    print(output_image.shape)