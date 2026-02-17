"""Plotting and visualization utilities for cGAN."""

import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

# CIFAR-10 class names for labeling
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def denormalize(tensor):
    """Convert images from [-1, 1] back to [0, 1] for display."""
    return (tensor + 1.0) / 2.0


def save_image_grid(images, labels, epoch, save_dir, nrow=10):
    """Save a grid of generated images sorted by class.

    Args:
        images: (N, 3, 32, 32) tensor in [-1, 1].
        labels: (N,) class labels.
        epoch: current epoch number (for filename).
        save_dir: directory to save the image.
        nrow: number of images per row.
    """
    os.makedirs(save_dir, exist_ok=True)

    images = denormalize(images.cpu())
    labels_cpu = labels.cpu()

    # Sort by label to group classes together
    sorted_indices = labels_cpu.argsort()
    images = images[sorted_indices]

    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)

    fig, ax = plt.subplots(1, 1, figsize=(nrow * 1.2, len(images) // nrow * 1.2))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.set_title(f"Generated Samples - Epoch {epoch}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"samples_epoch_{epoch:04d}.png"), dpi=150
    )
    plt.close()


def plot_training_curves(history, save_dir):
    """Plot generator/discriminator loss and accuracy curves.

    Args:
        history: dict with keys g_losses, d_losses, d_real_acc, d_fake_acc.
        save_dir: directory to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["g_losses"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, history["g_losses"], label="Generator")
    ax1.plot(epochs, history["d_losses"], label="Discriminator")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Losses")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, history["d_real_acc"], label="D Accuracy (Real)")
    ax2.plot(epochs, history["d_fake_acc"], label="D Accuracy (Fake)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Discriminator Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


def interpolate_latent_space(generator, z1, z2, label, num_steps=10,
                             device="cpu"):
    """Linear interpolation between two latent vectors with a fixed class.

    Args:
        generator: trained Generator model.
        z1: (latent_dim,) start latent vector.
        z2: (latent_dim,) end latent vector.
        label: integer class label.
        num_steps: number of interpolation steps.
        device: torch device.

    Returns:
        (num_steps, 3, 32, 32) tensor of generated images.
    """
    generator.eval()
    alphas = torch.linspace(0, 1, num_steps)
    z_interp = torch.stack(
        [z1 * (1 - a) + z2 * a for a in alphas]
    ).to(device)
    labels = torch.full((num_steps,), label, dtype=torch.long, device=device)

    with torch.no_grad():
        images = generator(z_interp, labels)
    return images


def visualize_class_variation(generator, z, num_classes=10, device="cpu"):
    """Fixed latent vector, varying class label.

    Args:
        generator: trained Generator model.
        z: (latent_dim,) single latent vector.
        num_classes: number of classes.
        device: torch device.

    Returns:
        (num_classes, 3, 32, 32) tensor of generated images.
    """
    generator.eval()
    z_batch = z.unsqueeze(0).repeat(num_classes, 1).to(device)
    labels = torch.arange(num_classes, device=device)

    with torch.no_grad():
        images = generator(z_batch, labels)
    return images


def create_full_visualization(generator, save_dir, latent_dim=100,
                              num_classes=10, device="cpu"):
    """Generate all visualizations from a trained generator.

    Creates:
    1. Class-conditioned sample grid (10x10)
    2. Latent space interpolation for each class
    3. Class variation with multiple fixed z vectors

    Args:
        generator: trained Generator model.
        save_dir: directory to save visualizations.
        latent_dim: dimensionality of latent space.
        num_classes: number of classes.
        device: torch device.
    """
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()

    # 1. Sample grid: 10 images per class
    z = torch.randn(num_classes * 10, latent_dim, device=device)
    labels = torch.arange(num_classes, device=device).repeat_interleave(10)
    with torch.no_grad():
        samples = generator(z, labels)
    save_image_grid(samples, labels, epoch=0, save_dir=save_dir, nrow=10)

    # 2. Latent space interpolation for a few classes
    z1 = torch.randn(latent_dim, device=device)
    z2 = torch.randn(latent_dim, device=device)

    fig, axes = plt.subplots(num_classes, 10, figsize=(15, 15))
    fig.suptitle("Latent Space Interpolation (rows = classes)", fontsize=14)

    for cls in range(num_classes):
        interp_images = interpolate_latent_space(
            generator, z1, z2, label=cls, num_steps=10, device=device
        )
        interp_images = denormalize(interp_images.cpu())
        for j in range(10):
            axes[cls, j].imshow(interp_images[j].permute(1, 2, 0).numpy())
            axes[cls, j].axis("off")
        axes[cls, 0].set_ylabel(CIFAR10_CLASSES[cls], fontsize=8,
                                rotation=0, labelpad=50)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "interpolation.png"), dpi=150)
    plt.close()

    # 3. Class variation: 5 fixed z vectors, 10 classes each
    fig, axes = plt.subplots(5, num_classes, figsize=(15, 8))
    fig.suptitle("Class Variation (rows = fixed z, cols = class)", fontsize=14)

    for i in range(5):
        z_fixed = torch.randn(latent_dim, device=device)
        var_images = visualize_class_variation(
            generator, z_fixed, num_classes=num_classes, device=device
        )
        var_images = denormalize(var_images.cpu())
        for cls in range(num_classes):
            axes[i, cls].imshow(var_images[cls].permute(1, 2, 0).numpy())
            axes[i, cls].axis("off")
            if i == 0:
                axes[i, cls].set_title(CIFAR10_CLASSES[cls], fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_variation.png"), dpi=150)
    plt.close()

    print(f"Visualizations saved to {save_dir}")
