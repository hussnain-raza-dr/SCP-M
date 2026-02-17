"""Training script for the conditional DCGAN with CLI interface."""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.optim import Adam
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import get_dataloaders
from models.cgan import ConditionalGAN
from training.losses import discriminator_loss, generator_loss
from evaluation.visualize import save_image_grid, plot_training_curves


def set_seed(seed=42):
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config, resume_path=None):
    """Main training function.

    Args:
        config: dict loaded from YAML config.
        resume_path: optional path to checkpoint to resume from.
    """
    # Setup
    seed = config.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    results_dir = config["paths"]["results_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Data
    train_loader, test_loader = get_dataloaders(config)
    print(f"Training batches: {len(train_loader)}, "
          f"Test batches: {len(test_loader)}")

    # Model
    cgan = ConditionalGAN(config, device)
    G = cgan.generator
    D = cgan.discriminator

    # Print model summaries
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")

    # Optimizers
    train_cfg = config["training"]
    optimizer_g = Adam(
        G.parameters(),
        lr=train_cfg["lr_g"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=train_cfg.get("weight_decay_g", 0.0),
    )
    optimizer_d = Adam(
        D.parameters(),
        lr=train_cfg["lr_d"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=train_cfg.get("weight_decay_d", 0.0),
    )

    # Fixed noise for consistent visualization across epochs
    num_classes = config["model"]["num_classes"]
    latent_dim = config["model"]["latent_dim"]
    num_per_class = config["evaluation"]["num_sample_per_class"]
    num_fixed = num_classes * num_per_class  # 10 * 10 = 100

    fixed_noise = torch.randn(num_fixed, latent_dim, device=device)
    fixed_labels = torch.arange(num_classes, device=device).repeat_interleave(
        num_per_class
    )  # [0,0,...,1,1,...,9,9,...]

    # Training history
    history = {
        "g_losses": [],
        "d_losses": [],
        "d_real_acc": [],
        "d_fake_acc": [],
    }

    # Resume from checkpoint
    start_epoch = 0
    if resume_path is not None:
        start_epoch, saved_history = cgan.load_checkpoint(
            resume_path, optimizer_g, optimizer_d
        )
        if saved_history:
            history = saved_history
        print(f"Resumed from epoch {start_epoch}")

    # Label smoothing
    label_smooth_real = train_cfg.get("label_smooth_real", 1.0)
    label_smooth_fake = train_cfg.get("label_smooth_fake", 0.0)

    # Training loop
    num_epochs = train_cfg["num_epochs"]
    eval_cfg = config["evaluation"]

    for epoch in range(start_epoch, num_epochs):
        G.train()
        D.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_d_real_acc = 0.0
        epoch_d_fake_acc = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for real_images, real_labels in pbar:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)

            # =============================================
            # Train Discriminator
            # =============================================
            optimizer_d.zero_grad()

            # D on real images
            real_output = D(real_images, real_labels)

            # Generate fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (batch_size,),
                                        device=device)
            fake_images = G(z, fake_labels)

            # D on fake images (detached — don't update G)
            fake_output = D(fake_images.detach(), fake_labels)

            # Discriminator loss and update
            d_loss, _, _ = discriminator_loss(
                real_output, fake_output,
                real_label=label_smooth_real,
                fake_label=label_smooth_fake,
            )
            d_loss.backward()
            optimizer_d.step()

            # Discriminator accuracy
            d_real_acc = (torch.sigmoid(real_output) > 0.5).float().mean().item()
            d_fake_acc = (torch.sigmoid(fake_output) < 0.5).float().mean().item()

            # =============================================
            # Train Generator
            # =============================================
            optimizer_g.zero_grad()

            # Generate fresh fakes
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (batch_size,),
                                        device=device)
            fake_images = G(z, fake_labels)

            # D on fakes (no detach — gradients flow into G)
            fake_output = D(fake_images, fake_labels)

            # Generator loss and update
            g_loss = generator_loss(fake_output)
            g_loss.backward()
            optimizer_g.step()

            # =============================================
            # Logging
            # =============================================
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_d_real_acc += d_real_acc
            epoch_d_fake_acc += d_fake_acc
            num_batches += 1

            if num_batches % eval_cfg["log_every"] == 0:
                pbar.set_postfix({
                    "D": f"{d_loss.item():.4f}",
                    "G": f"{g_loss.item():.4f}",
                    "D_r": f"{d_real_acc:.2f}",
                    "D_f": f"{d_fake_acc:.2f}",
                })

        # End-of-epoch averages
        avg_g = epoch_g_loss / num_batches
        avg_d = epoch_d_loss / num_batches
        avg_d_real = epoch_d_real_acc / num_batches
        avg_d_fake = epoch_d_fake_acc / num_batches

        history["g_losses"].append(avg_g)
        history["d_losses"].append(avg_d)
        history["d_real_acc"].append(avg_d_real)
        history["d_fake_acc"].append(avg_d_fake)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"D_loss: {avg_d:.4f}  G_loss: {avg_g:.4f}  "
            f"D_acc(real): {avg_d_real:.2f}  D_acc(fake): {avg_d_fake:.2f}"
        )

        # Generate sample grid
        if (epoch + 1) % eval_cfg["sample_every"] == 0:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise, fixed_labels)
            save_image_grid(samples, fixed_labels, epoch + 1, results_dir,
                            nrow=num_per_class)

        # Save checkpoint
        if (epoch + 1) % eval_cfg["save_every"] == 0:
            ckpt_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"
            )
            cgan.save_checkpoint(ckpt_path, epoch + 1, optimizer_g,
                                 optimizer_d, history)
            print(f"  Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
    cgan.save_checkpoint(final_path, num_epochs, optimizer_g, optimizer_d,
                         history)
    print(f"Final checkpoint saved: {final_path}")

    # Save training curves
    plot_training_curves(history, results_dir)

    # Save history as JSON
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Conditional DCGAN")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
