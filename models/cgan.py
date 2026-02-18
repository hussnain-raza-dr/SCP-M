"""Complete conditional GAN wrapper with weight initialization and utilities.

Supports both baseline and improved architectures via config-driven selection.
"""

import os
import torch
import torch.nn as nn

from models.generator import Generator, ImprovedGenerator
from models.discriminator import Discriminator, ImprovedDiscriminator


def weights_init(m):
    """DCGAN weight initialization.

    Conv/ConvTranspose: Normal(0, 0.02)
    BatchNorm: weight Normal(1, 0.02), bias zeros.
    Linear: Normal(0, 0.02), bias zeros.
    Embedding: Normal(0, 0.02).
    """
    classname = m.__class__.__name__
    if "Conv" in classname and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)
    elif classname == "Linear":
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif classname == "Embedding":
        nn.init.normal_(m.weight.data, 0.0, 0.02)


def weights_init_improved(m):
    """Orthogonal initialization for improved architectures.

    Orthogonal init provides better gradient flow in deep residual networks
    and is the standard choice in BigGAN and SNGAN.

    Spectral-normed layers are skipped (they manage their own scaling).
    """
    classname = m.__class__.__name__
    # Skip spectral-normed modules (they have weight_orig, not weight)
    if hasattr(m, "weight_orig"):
        return
    if "Conv" in classname and hasattr(m, "weight"):
        nn.init.orthogonal_(m.weight.data)
    elif "BatchNorm" in classname or "GroupNorm" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif classname == "Linear":
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif classname == "Embedding":
        nn.init.orthogonal_(m.weight.data)


class ConditionalGAN:
    """Wrapper holding Generator and Discriminator with helper methods.

    Automatically selects baseline or improved architectures based on config.
    """

    def __init__(self, config, device):
        model_cfg = config["model"]
        self.latent_dim = model_cfg["latent_dim"]
        self.num_classes = model_cfg["num_classes"]
        self.device = device

        g_type = model_cfg.get("generator_type", "baseline")
        d_type = model_cfg.get("discriminator_type", "baseline")

        # --- Build Generator ---
        if g_type == "improved":
            self.generator = ImprovedGenerator(
                latent_dim=model_cfg["latent_dim"],
                embed_dim=model_cfg["embed_dim"],
                num_classes=model_cfg["num_classes"],
                image_channels=model_cfg["image_channels"],
                channels=model_cfg.get("generator_channels", [512, 256, 128]),
                normalization=model_cfg.get("g_normalization", "batchnorm"),
                activation=model_cfg.get("g_activation", "relu"),
            )
        else:
            self.generator = Generator(
                latent_dim=model_cfg["latent_dim"],
                embed_dim=model_cfg["embed_dim"],
                num_classes=model_cfg["num_classes"],
                image_channels=model_cfg["image_channels"],
                channels=model_cfg.get("generator_channels", [256, 128, 64]),
            )

        # --- Build Discriminator ---
        if d_type == "improved":
            self.discriminator = ImprovedDiscriminator(
                num_classes=model_cfg["num_classes"],
                image_channels=model_cfg["image_channels"],
                channels=model_cfg.get("discriminator_channels", [128, 256, 512]),
                normalization=model_cfg.get("d_normalization", "spectralnorm"),
                activation=model_cfg.get("d_activation", "leaky_relu"),
                use_minibatch_stddev=model_cfg.get("d_use_minibatch_stddev", True),
            )
        else:
            self.discriminator = Discriminator(
                num_classes=model_cfg["num_classes"],
                image_channels=model_cfg["image_channels"],
                channels=model_cfg.get("discriminator_channels", [64, 128, 256]),
                normalization=model_cfg.get("d_normalization", "batchnorm"),
            )

        # Apply weight initialization
        if g_type == "improved":
            self.generator.apply(weights_init_improved)
        else:
            self.generator.apply(weights_init)

        # Only apply weight init to D if not using spectral norm
        # (spectral norm manages its own weight scaling)
        d_norm = model_cfg.get("d_normalization", "batchnorm")
        if d_type == "improved" and d_norm == "spectralnorm":
            self.discriminator.apply(weights_init_improved)
        else:
            self.discriminator.apply(weights_init)

        # Move to device
        self.generator.to(device)
        self.discriminator.to(device)

    def sample(self, num_images, labels=None):
        """Generate images.

        Args:
            num_images: number of images to generate.
            labels: (num_images,) class labels. If None, random labels are used.

        Returns:
            (images, labels) tuple.
        """
        z = torch.randn(num_images, self.latent_dim, device=self.device)
        if labels is None:
            labels = torch.randint(0, self.num_classes, (num_images,),
                                   device=self.device)
        self.generator.eval()
        with torch.no_grad():
            images = self.generator(z, labels)
        return images, labels

    def save_checkpoint(self, path, epoch, optimizer_g, optimizer_d, history,
                        G_ema=None):
        """Save full training state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "history": history,
        }
        if G_ema is not None:
            state["generator_ema_state_dict"] = G_ema.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path, optimizer_g=None, optimizer_d=None):
        """Load training state from disk.

        Returns:
            epoch number to resume from.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        if optimizer_g is not None:
            optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        if optimizer_d is not None:
            optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        return checkpoint["epoch"], checkpoint.get("history", {})
