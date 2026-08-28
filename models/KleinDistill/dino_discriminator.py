"""PatchGAN-style discriminator operating on DINOv2 patch-token features
instead of raw pixels.

Complements a pixel-space PatchGAN discriminator: classifying real/fake in a
frozen pretrained feature space (cf. vision-aided/projected GAN discriminators,
e.g. "Projected GANs Converge Faster") tends to give a stronger, more stable
adversarial signal than a from-scratch pixel discriminator alone, since the
features already encode rich semantic/structural content rather than starting
from raw RGB statistics.
"""

import torch.nn as nn


class DinoPatchDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, patch_tokens):
        """patch_tokens: [B, N, D] DINO patch tokens; N must be a perfect square
        (DINOv2 patches tile a square grid over the input image)."""
        b, n, d = patch_tokens.shape
        side = int(round(n ** 0.5))
        assert side * side == n, f"Expected a square patch grid, got N={n}"
        x = patch_tokens.transpose(1, 2).reshape(b, d, side, side)
        return self.net(x)
