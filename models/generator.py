"""Generator architecture for the conditional DCGAN."""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Conditional DCGAN Generator.

    Architecture:
        (nz + nembed) -> Linear -> 256x4x4
        -> ConvTranspose2d 256->128 (8x8)
        -> ConvTranspose2d 128->64  (16x16)
        -> ConvTranspose2d 64->3    (32x32)

    Conditioning: class embedding concatenated to latent vector z.
    """

    def __init__(self, latent_dim=100, embed_dim=50, num_classes=10,
                 image_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # Class embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Project and reshape: (nz + nembed) -> 256 * 4 * 4
        self.project = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256 * 4 * 4, bias=False),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True),
        )

        # Upsampling blocks
        self.conv_blocks = nn.Sequential(
            # Block 1: 256x4x4 -> 128x8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Block 2: 128x8x8 -> 64x16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Block 3 (output): 64x16x16 -> 3x32x32
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        """Generate images from noise and class labels.

        Args:
            z: (B, latent_dim) noise vector.
            labels: (B,) integer class labels.

        Returns:
            (B, 3, 32, 32) generated images in [-1, 1].
        """
        # Embed class labels and concatenate with z
        embedding = self.label_embedding(labels)        # (B, embed_dim)
        x = torch.cat([z, embedding], dim=1)            # (B, latent_dim + embed_dim)

        # Project and reshape to feature map
        x = self.project(x)                             # (B, 256*4*4)
        x = x.view(x.size(0), 256, 4, 4)               # (B, 256, 4, 4)

        # Upsample to image
        x = self.conv_blocks(x)                         # (B, 3, 32, 32)
        return x
