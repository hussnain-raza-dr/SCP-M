"""Loss function implementations for vanilla GAN training."""

import torch
import torch.nn as nn

_bce_logits = nn.BCEWithLogitsLoss()


def discriminator_loss(real_output, fake_output, real_label=1.0,
                       fake_label=0.0):
    """Vanilla GAN discriminator loss (BCEWithLogitsLoss).

    Args:
        real_output: (B, 1) raw logits from D on real images.
        fake_output: (B, 1) raw logits from D on fake images.
        real_label: target value for real images (1.0 or 0.9 for smoothing).
        fake_label: target value for fake images (0.0).

    Returns:
        (total_loss, real_loss, fake_loss) tuple.
    """
    real_targets = torch.full_like(real_output, real_label)
    fake_targets = torch.full_like(fake_output, fake_label)

    real_loss = _bce_logits(real_output, real_targets)
    fake_loss = _bce_logits(fake_output, fake_targets)
    total_loss = real_loss + fake_loss

    return total_loss, real_loss, fake_loss


def generator_loss(fake_output):
    """Vanilla GAN generator loss — G wants D to classify fakes as real.

    Args:
        fake_output: (B, 1) raw logits from D on generated images.

    Returns:
        Scalar loss.
    """
    real_targets = torch.ones_like(fake_output)
    return _bce_logits(fake_output, real_targets)
