"""Complete conditional GAN wrapper with weight initialization and utilities."""

import os
import torch
import torch.nn as nn

from models.generator import Generator
from models.discriminator import Discriminator


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


class ConditionalGAN:
    """Wrapper holding Generator and Discriminator with helper methods."""

    def __init__(self, config, device):
        model_cfg = config["model"]
        self.latent_dim = model_cfg["latent_dim"]
        self.num_classes = model_cfg["num_classes"]
        self.device = device

        self.generator = Generator(
            latent_dim=model_cfg["latent_dim"],
            embed_dim=model_cfg["embed_dim"],
            num_classes=model_cfg["num_classes"],
            image_channels=model_cfg["image_channels"],
        )
        self.discriminator = Discriminator(
            num_classes=model_cfg["num_classes"],
            image_channels=model_cfg["image_channels"],
        )

        # Apply DCGAN weight initialization
        self.generator.apply(weights_init)
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

    def save_checkpoint(self, path, epoch, optimizer_g, optimizer_d, history):
        """Save full training state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "history": history,
        }, path)

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
