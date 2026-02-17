"""Discriminator architecture for the conditional DCGAN."""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Conditional DCGAN Discriminator with projection conditioning.

    Architecture:
        Conv2d 3->64   (32x32 -> 16x16, stride=2)
        Conv2d 64->128 (16x16 -> 8x8,   stride=2)
        Conv2d 128->256(8x8   -> 4x4,   stride=2)
        Flatten -> 4096
        Linear(4096, 1) + projection(class_embed dot features)

    No sigmoid at output — raw logits (used with BCEWithLogitsLoss).
    """

    def __init__(self, num_classes=10, image_channels=3):
        super().__init__()

        # Convolutional feature extraction
        self.features = nn.Sequential(
            # Block 1: 3x32x32 -> 64x16x16 (no BatchNorm on first layer)
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: 64x16x16 -> 128x8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: 128x8x8 -> 256x4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        feature_dim = 256 * 4 * 4  # 4096

        # Unconditional output head
        self.fc = nn.Linear(feature_dim, 1)

        # Projection discriminator: class embedding in feature space
        self.label_embedding = nn.Embedding(num_classes, feature_dim)

    def forward(self, images, labels):
        """Classify images as real/fake conditioned on class labels.

        Args:
            images: (B, 3, 32, 32) input images.
            labels: (B,) integer class labels.

        Returns:
            (B, 1) raw logits (no sigmoid).
        """
        # Extract features
        x = self.features(images)                       # (B, 256, 4, 4)
        x = x.view(x.size(0), -1)                      # (B, 4096)

        # Unconditional score
        out = self.fc(x)                                # (B, 1)

        # Projection conditioning: dot product of features and class embedding
        embed = self.label_embedding(labels)            # (B, 4096)
        proj = torch.sum(x * embed, dim=1, keepdim=True)  # (B, 1)

        return out + proj                               # (B, 1)
