"""Generator architectures for the conditional DCGAN.

Provides both the baseline DCGAN generator and an improved generator with:
  - Residual connections for better gradient flow
  - Configurable normalization (BatchNorm, LayerNorm)
  - Configurable activation functions
  - Wider channel capacity
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Conditional DCGAN Generator.

    Standard ConvTranspose2d upsampling architecture with BatchNorm + ReLU.
    Supports configurable channel widths for capacity tuning.

    Architecture (default channels [256, 128, 64]):
        (nz + nembed) -> Linear -> ch0 x 4x4
        -> ConvTranspose2d ch0->ch1 (8x8)
        -> ConvTranspose2d ch1->ch2 (16x16)
        -> ConvTranspose2d ch2->3   (32x32)

    Conditioning: class embedding concatenated to latent vector z.
    """

    def __init__(self, latent_dim=100, embed_dim=50, num_classes=10,
                 image_channels=3, channels=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        if channels is None:
            channels = [256, 128, 64]

        ch0, ch1, ch2 = channels

        # Class embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Project and reshape: (nz + nembed) -> ch0 * 4 * 4
        self.project = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, ch0 * 4 * 4, bias=False),
            nn.BatchNorm1d(ch0 * 4 * 4),
            nn.ReLU(True),
        )
        self.ch0 = ch0

        # Upsampling blocks
        self.conv_blocks = nn.Sequential(
            # Block 1: ch0 x 4x4 -> ch1 x 8x8
            nn.ConvTranspose2d(ch0, ch1, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(ch1),
            nn.ReLU(True),

            # Block 2: ch1 x 8x8 -> ch2 x 16x16
            nn.ConvTranspose2d(ch1, ch2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(ch2),
            nn.ReLU(True),

            # Block 3 (output): ch2 x 16x16 -> 3 x 32x32
            nn.ConvTranspose2d(ch2, image_channels, kernel_size=4, stride=2,
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
        x = self.project(x)                             # (B, ch0*4*4)
        x = x.view(x.size(0), self.ch0, 4, 4)          # (B, ch0, 4, 4)

        # Upsample to image
        x = self.conv_blocks(x)                         # (B, 3, 32, 32)
        return x


# ---------------------------------------------------------------------------
# Improved Generator
# ---------------------------------------------------------------------------

class ResidualUpsampleBlock(nn.Module):
    """Upsampling block with a residual (skip) connection.

    Main path:  BN -> Act -> Upsample(2x) -> Conv3x3 -> BN -> Act -> Conv3x3
    Skip path:  Upsample(2x) -> Conv1x1

    Using pre-activation layout (BN before conv) following best practices from
    BigGAN and ResNet-v2.
    """

    def __init__(self, in_channels, out_channels, normalization="batchnorm",
                 activation="relu"):
        super().__init__()

        act_fn = nn.ReLU(True) if activation == "relu" else nn.LeakyReLU(0.2, True)

        # --- Main path ---
        layers = []
        # Pre-activation norm + activation
        layers.append(_make_norm(normalization, in_channels))
        layers.append(act_fn)
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        layers.append(_make_norm(normalization, out_channels))
        layers.append(nn.ReLU(True) if activation == "relu" else nn.LeakyReLU(0.2, True))
        layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        self.main = nn.Sequential(*layers)

        # --- Skip path ---
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)


class ImprovedGenerator(nn.Module):
    """Enhanced conditional generator with residual upsampling blocks.

    Improvements over baseline:
      - Residual connections prevent vanishing gradients in deeper networks
      - Configurable normalization (BatchNorm or LayerNorm)
      - Wider default channels (512->256->128) for higher capacity
      - Pre-activation residual blocks (BN-Act-Conv) for stable training
    """

    def __init__(self, latent_dim=128, embed_dim=64, num_classes=10,
                 image_channels=3, channels=None, normalization="batchnorm",
                 activation="relu"):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        if channels is None:
            channels = [512, 256, 128]

        ch0 = channels[0]  # first feature map channels (e.g., 512)

        # Class embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Project: (z + embed) -> ch0 * 4 * 4
        self.project = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, ch0 * 4 * 4, bias=False),
            nn.BatchNorm1d(ch0 * 4 * 4),
            nn.ReLU(True),
        )
        self.ch0 = ch0

        # Residual upsampling blocks: 4x4 -> 8x8 -> 16x16 -> 32x32
        self.blocks = nn.ModuleList()
        in_ch = ch0
        for out_ch in channels[1:]:
            self.blocks.append(
                ResidualUpsampleBlock(in_ch, out_ch, normalization, activation)
            )
            in_ch = out_ch

        # Final: norm -> act -> upsample -> conv -> tanh
        self.final = nn.Sequential(
            _make_norm(normalization, in_ch),
            nn.ReLU(True) if activation == "relu" else nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, image_channels, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        embedding = self.label_embedding(labels)
        x = torch.cat([z, embedding], dim=1)

        x = self.project(x)
        x = x.view(x.size(0), self.ch0, 4, 4)

        for block in self.blocks:
            x = block(x)

        x = self.final(x)
        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_norm(norm_type, num_features):
    """Create a normalization layer.

    Args:
        norm_type: "batchnorm", "layernorm", or "none".
        num_features: number of channels / features.

    Returns:
        nn.Module normalization layer.
    """
    if norm_type == "batchnorm":
        return nn.BatchNorm2d(num_features)
    elif norm_type == "layernorm":
        # LayerNorm over the channel dimension (GroupNorm with 1 group)
        return nn.GroupNorm(1, num_features)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization: {norm_type}")
