"""Loss function implementations for GAN training.

Supports multiple loss formulations:
  - Vanilla GAN (BCEWithLogitsLoss)
  - WGAN-GP (Wasserstein distance + gradient penalty)
  - LSGAN (Least-Squares GAN)
  - Hinge loss (used in SNGAN / BigGAN)
"""

import torch
import torch.nn as nn
import torch.autograd as autograd

_bce_logits = nn.BCEWithLogitsLoss()


# ===================================================================
# Vanilla GAN (baseline)
# ===================================================================

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


# ===================================================================
# WGAN-GP (Wasserstein GAN with Gradient Penalty)
# ===================================================================
# Reference: Gulrajani et al., "Improved Training of Wasserstein GANs", 2017
#
# Why it helps:
#   - The Wasserstein distance provides a meaningful, smooth loss that
#     correlates with sample quality (unlike vanilla GAN's JS divergence).
#   - Gradient penalty enforces the Lipschitz constraint softly, avoiding
#     the training instability of weight clipping.
#   - Eliminates mode collapse by providing consistent gradients even
#     when the discriminator is optimal.
# ===================================================================

def wgan_discriminator_loss(real_output, fake_output):
    """WGAN critic loss: maximize E[D(real)] - E[D(fake)].

    Equivalent to minimizing -(E[D(real)] - E[D(fake)]).

    Returns:
        (wasserstein_loss, real_mean, fake_mean) tuple.
    """
    real_mean = real_output.mean()
    fake_mean = fake_output.mean()
    loss = fake_mean - real_mean  # we minimize this
    return loss, -real_mean, fake_mean


def wgan_generator_loss(fake_output):
    """WGAN generator loss: maximize E[D(fake)] = minimize -E[D(fake)]."""
    return -fake_output.mean()


def gradient_penalty(discriminator, real_images, fake_images, labels, device,
                     lambda_gp=10.0):
    """Compute gradient penalty for WGAN-GP.

    Interpolates between real and fake images, passes through D, and penalizes
    the norm of the gradient w.r.t. the interpolated inputs.

    The gradient penalty encourages ||∇D(x̂)||₂ ≈ 1, which is the necessary
    condition for the optimal WGAN critic.

    Args:
        discriminator: the discriminator / critic model.
        real_images: (B, C, H, W) real image batch.
        fake_images: (B, C, H, W) generated image batch.
        labels: (B,) class labels for conditional GAN.
        device: torch device.
        lambda_gp: penalty coefficient (default 10.0).

    Returns:
        Scalar gradient penalty loss (already scaled by lambda_gp).
    """
    batch_size = real_images.size(0)

    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)

    # Forward pass
    d_interpolated = discriminator(interpolated, labels)

    # Compute gradients
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Flatten and compute norm
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)

    # Penalty: (||∇D|| - 1)²
    penalty = lambda_gp * ((grad_norm - 1.0) ** 2).mean()
    return penalty


# ===================================================================
# R1 Gradient Penalty (real-only regularization)
# ===================================================================
# Reference: Mescheder et al., "Which Training Methods for GANs do
#            actually Converge?", ICML 2018
#
# Why it helps:
#   - Unlike WGAN-GP (penalizes interpolated samples), R1 only penalizes
#     gradient norm on REAL images, directly preventing discriminator
#     collapse on real recognition.
#   - Compatible with spectral normalization and hinge loss.
#   - Small coefficient (gamma=0.1-1.0) is enough to stabilize training
#     without over-constraining the discriminator.
# ===================================================================

def r1_gradient_penalty(discriminator, real_images, labels, device,
                        gamma=1.0):
    """R1 gradient penalty on real images only.

    Penalizes ||grad_D(real)||^2 to keep D well-behaved on the real manifold.
    This prevents D from losing its ability to recognize real images.

    Args:
        discriminator: the discriminator model.
        real_images: (B, C, H, W) real image batch.
        labels: (B,) class labels for conditional GAN.
        device: torch device.
        gamma: penalty coefficient (default 1.0).

    Returns:
        Scalar R1 penalty loss (already scaled by gamma/2).
    """
    real_images = real_images.detach().requires_grad_(True)
    real_output = discriminator(real_images, labels)

    gradients = autograd.grad(
        outputs=real_output.sum(),
        inputs=real_images,
        create_graph=True,
    )[0]

    # R1 = (gamma/2) * E[||grad||^2]
    grad_penalty = (gamma / 2.0) * gradients.pow(2).sum(dim=[1, 2, 3]).mean()
    return grad_penalty


# ===================================================================
# LSGAN (Least Squares GAN)
# ===================================================================
# Reference: Mao et al., "Least Squares GANs", 2017
#
# Why it helps:
#   - Replaces binary cross-entropy with MSE, which penalizes samples
#     far from the decision boundary even if they're on the correct side.
#   - Generates higher-quality images by pushing fake samples closer to
#     the real data manifold instead of just crossing the decision boundary.
#   - More stable gradients compared to vanilla GAN — no vanishing
#     gradient problem when D is confident.
# ===================================================================

def lsgan_discriminator_loss(real_output, fake_output):
    """LSGAN discriminator loss: MSE with targets 1 (real) and 0 (fake).

    L_D = 0.5 * E[(D(real) - 1)²] + 0.5 * E[D(fake)²]

    Returns:
        (total_loss, real_loss, fake_loss) tuple.
    """
    real_loss = 0.5 * ((real_output - 1.0) ** 2).mean()
    fake_loss = 0.5 * (fake_output ** 2).mean()
    total_loss = real_loss + fake_loss
    return total_loss, real_loss, fake_loss


def lsgan_generator_loss(fake_output):
    """LSGAN generator loss: MSE with target 1 (G wants D to output 1).

    L_G = 0.5 * E[(D(fake) - 1)²]
    """
    return 0.5 * ((fake_output - 1.0) ** 2).mean()


# ===================================================================
# Hinge Loss
# ===================================================================
# Reference: Miyato et al., "Spectral Normalization for GANs", 2018
#            Zhang et al., "Self-Attention GAN", 2019
#
# Why it helps:
#   - Used in SNGAN and SAGAN — proven to work well with spectral norm.
#   - Does not push D(real) to infinity; instead, only requires D(real)>1
#     and D(fake)<-1. This prevents D from becoming too confident.
#   - Provides strong gradients to G while keeping D well-calibrated.
# ===================================================================

def hinge_discriminator_loss(real_output, fake_output):
    """Hinge loss for discriminator.

    L_D = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]

    Returns:
        (total_loss, real_loss, fake_loss) tuple.
    """
    real_loss = torch.relu(1.0 - real_output).mean()
    fake_loss = torch.relu(1.0 + fake_output).mean()
    total_loss = real_loss + fake_loss
    return total_loss, real_loss, fake_loss


def hinge_generator_loss(fake_output):
    """Hinge loss for generator: maximize E[D(fake)] = minimize -E[D(fake)]."""
    return -fake_output.mean()


# ===================================================================
# Dispatcher — select loss by name
# ===================================================================

def get_d_loss_fn(loss_type):
    """Return (d_loss_fn, needs_gradient_penalty) for the given loss type.

    Args:
        loss_type: "vanilla", "wgan-gp", "lsgan", or "hinge".

    Returns:
        Discriminator loss function with signature (real_output, fake_output, **kwargs)
        -> (total, real_part, fake_part).
    """
    dispatch = {
        "vanilla": discriminator_loss,
        "wgan-gp": wgan_discriminator_loss,
        "lsgan": lsgan_discriminator_loss,
        "hinge": hinge_discriminator_loss,
    }
    if loss_type not in dispatch:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                         f"Choose from {list(dispatch.keys())}")
    return dispatch[loss_type]


def get_g_loss_fn(loss_type):
    """Return generator loss function for the given loss type."""
    dispatch = {
        "vanilla": generator_loss,
        "wgan-gp": wgan_generator_loss,
        "lsgan": lsgan_generator_loss,
        "hinge": hinge_generator_loss,
    }
    if loss_type not in dispatch:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                         f"Choose from {list(dispatch.keys())}")
    return dispatch[loss_type]
