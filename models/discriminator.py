"""Discriminator architectures for the conditional DCGAN.

Provides both the baseline and an improved discriminator with:
  - Spectral normalization for enforcing Lipschitz constraint
  - Minibatch standard deviation for detecting mode collapse
  - Residual downsampling blocks
  - Configurable normalization and activation
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    """Conditional DCGAN Discriminator with projection conditioning.

    Standard strided-conv downsampling architecture. Supports configurable
    channel widths and normalization (batchnorm or spectralnorm).

    Architecture (default channels [64, 128, 256]):
        Conv2d 3->ch0   (32x32 -> 16x16, stride=2)
        Conv2d ch0->ch1  (16x16 -> 8x8,   stride=2)
        Conv2d ch1->ch2  (8x8   -> 4x4,   stride=2)
        Flatten -> ch2*4*4
        Linear(ch2*4*4, 1) + projection(class_embed dot features)

    No sigmoid at output — raw logits/scores.
    """

    def __init__(self, num_classes=10, image_channels=3, channels=None,
                 normalization="batchnorm"):
        super().__init__()

        if channels is None:
            channels = [64, 128, 256]

        ch0, ch1, ch2 = channels
        use_sn = (normalization == "spectralnorm")
        wrap = spectral_norm if use_sn else lambda x: x

        # Convolutional feature extraction
        # Block 1: no normalization on first layer (DCGAN convention)
        layers = [
            wrap(nn.Conv2d(image_channels, ch0, kernel_size=4, stride=2,
                           padding=1, bias=not use_sn)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Block 2
        layers.append(wrap(nn.Conv2d(ch0, ch1, kernel_size=4, stride=2,
                                     padding=1, bias=not use_sn)))
        if not use_sn:
            layers.append(nn.BatchNorm2d(ch1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Block 3
        layers.append(wrap(nn.Conv2d(ch1, ch2, kernel_size=4, stride=2,
                                     padding=1, bias=not use_sn)))
        if not use_sn:
            layers.append(nn.BatchNorm2d(ch2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.features = nn.Sequential(*layers)

        feature_dim = ch2 * 4 * 4

        # Unconditional output head
        self.fc = wrap(nn.Linear(feature_dim, 1))

        # Projection discriminator: class embedding in feature space
        self.label_embedding = nn.Embedding(num_classes, feature_dim)
        if use_sn:
            self.label_embedding = spectral_norm(self.label_embedding)

    def forward(self, images, labels):
        """Classify images as real/fake conditioned on class labels.

        Args:
            images: (B, 3, 32, 32) input images.
            labels: (B,) integer class labels.

        Returns:
            (B, 1) raw logits (no sigmoid).
        """
        # Extract features
        x = self.features(images)
        x = x.view(x.size(0), -1)

        # Unconditional score
        out = self.fc(x)

        # Projection conditioning: dot product of features and class embedding
        embed = self.label_embedding(labels)
        proj = torch.sum(x * embed, dim=1, keepdim=True)

        return out + proj


# ---------------------------------------------------------------------------
# Improved Discriminator (residual architecture)
# ---------------------------------------------------------------------------

class ResidualDownsampleBlock(nn.Module):
    """Downsampling block with residual (skip) connection.

    Main path:  Act -> Conv3x3 -> Act -> Conv3x3 -> AvgPool(2x)
    Skip path:  Conv1x1 -> AvgPool(2x)

    All convolutions wrapped in spectral normalization when requested.
    """

    def __init__(self, in_channels, out_channels, normalization="spectralnorm",
                 activation="leaky_relu", is_first=False):
        super().__init__()

        act_fn = (nn.LeakyReLU(0.2, True) if activation == "leaky_relu"
                  else nn.ReLU(True))
        wrap = _sn_wrap if normalization == "spectralnorm" else lambda x: x

        # --- Main path ---
        layers = []
        if not is_first:
            layers.append(act_fn)
        layers.append(wrap(nn.Conv2d(in_channels, out_channels, 3, 1, 1)))
        layers.append(nn.LeakyReLU(0.2, True) if activation == "leaky_relu"
                      else nn.ReLU(True))
        layers.append(wrap(nn.Conv2d(out_channels, out_channels, 3, 1, 1)))
        layers.append(nn.AvgPool2d(2))
        self.main = nn.Sequential(*layers)

        # --- Skip path ---
        self.skip = nn.Sequential(
            wrap(nn.Conv2d(in_channels, out_channels, 1, 1, 0)),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)


class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation layer (from ProGAN / StyleGAN).

    Appends a constant feature map containing the mean standard deviation
    across the batch. This gives the discriminator a signal about the
    diversity of the current minibatch, helping it detect mode collapse.
    """

    def forward(self, x):
        # x: (B, C, H, W)
        batch_std = x.std(dim=0, keepdim=True).mean()          # scalar
        stddev_map = batch_std.expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, stddev_map], dim=1)                # (B, C+1, H, W)


class ImprovedDiscriminator(nn.Module):
    """Enhanced conditional discriminator with spectral normalization.

    Improvements over baseline:
      - Spectral normalization enforces the Lipschitz constraint required by
        WGAN theory without needing weight clipping, improving training
        stability and reducing mode collapse.
      - Residual downsampling blocks for better gradient flow.
      - Minibatch standard deviation helps detect mode collapse.
      - Average pooling instead of strided conv for smoother downsampling.
      - No BatchNorm (incompatible with spectral norm and WGAN-GP).
    """

    def __init__(self, num_classes=10, image_channels=3, channels=None,
                 normalization="spectralnorm", activation="leaky_relu",
                 use_minibatch_stddev=True):
        super().__init__()

        if channels is None:
            channels = [128, 256, 512]

        self.use_minibatch_stddev = use_minibatch_stddev
        wrap = _sn_wrap if normalization == "spectralnorm" else lambda x: x

        # Residual downsampling blocks: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.blocks = nn.ModuleList()
        in_ch = image_channels
        for i, out_ch in enumerate(channels):
            self.blocks.append(
                ResidualDownsampleBlock(
                    in_ch, out_ch,
                    normalization=normalization,
                    activation=activation,
                    is_first=(i == 0),
                )
            )
            in_ch = out_ch

        # Minibatch stddev adds 1 channel
        extra_ch = 1 if use_minibatch_stddev else 0
        if use_minibatch_stddev:
            self.mbstd = MinibatchStdDev()

        # Activation before final layers
        self.act = (nn.LeakyReLU(0.2, True) if activation == "leaky_relu"
                    else nn.ReLU(True))

        feature_dim = (in_ch + extra_ch) * 4 * 4

        # Unconditional head
        self.fc = wrap(nn.Linear(feature_dim, 1))

        # Projection conditioning
        self.label_embedding = nn.Embedding(num_classes, feature_dim)
        if normalization == "spectralnorm":
            self.label_embedding = spectral_norm(self.label_embedding)

    def forward(self, images, labels):
        x = images
        for block in self.blocks:
            x = block(x)

        x = self.act(x)

        if self.use_minibatch_stddev:
            x = self.mbstd(x)

        x = x.view(x.size(0), -1)

        out = self.fc(x)

        embed = self.label_embedding(labels)
        proj = torch.sum(x * embed, dim=1, keepdim=True)

        return out + proj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sn_wrap(module):
    """Apply spectral normalization to a module."""
    return spectral_norm(module)
