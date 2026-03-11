"""Training script for the conditional GAN with CLI interface.

Supports both baseline (vanilla GAN) and improved configurations:
  - Multiple loss functions: vanilla, wgan-gp, lsgan, hinge
  - Configurable D:G training ratio
  - Gradient penalty (WGAN-GP)
  - Exponential moving average (EMA) of generator weights
  - Gradient clipping
  - Multi-GPU distributed training via MPI + PyTorch DDP
"""

import argparse
import copy
import json
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import get_dataloaders
from models.cgan import ConditionalGAN
from training.losses import (
    get_d_loss_fn,
    get_g_loss_fn,
    gradient_penalty,
    r1_gradient_penalty,
)
from evaluation.visualize import save_image_grid, plot_training_curves


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize distributed process group using environment variables.

    Works with both torchrun and mpirun launchers. Returns (rank, world_size)
    or (0, 1) when running without distributed.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Launched via torchrun / torch.distributed.launch
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        # Launched via mpirun (OpenMPI)
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
    else:
        # Single-process, non-distributed
        return 0, 1

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size


def cleanup_distributed():
    """Destroy the distributed process group if active."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Return True if this is rank 0 (or non-distributed)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


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


def create_ema(model):
    """Create an exponential moving average copy of a model.

    EMA keeps a smoothed version of the generator weights. At inference time,
    the EMA model typically produces higher-quality and more stable outputs
    because it averages out the noise from individual SGD steps.
    """
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


def update_ema(ema_model, model, decay=0.999):
    """Update EMA model parameters: ema = decay * ema + (1 - decay) * model."""
    with torch.no_grad():
        for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(model_p.data, alpha=1.0 - decay)


def train(config, resume_path=None):
    """Main training function.

    Args:
        config: dict loaded from YAML config.
        resume_path: optional path to checkpoint to resume from.
    """
    # Distributed setup
    rank, world_size = setup_distributed()
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Setup
    seed = config.get("seed", 42)
    set_seed(seed + rank)  # different seed per rank for data diversity
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if is_main_process():
        print(f"Using device: {device}")
        if distributed:
            print(f"Distributed training: {world_size} processes")

    # Paths
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    results_dir = config["paths"]["results_dir"]
    if is_main_process():
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
    if distributed:
        dist.barrier()

    # Data (pass distributed info for DistributedSampler)
    train_loader, test_loader = get_dataloaders(
        config, distributed=distributed, rank=rank, world_size=world_size,
    )
    if is_main_process():
        print(f"Training batches: {len(train_loader)}, "
              f"Test batches: {len(test_loader)}")

    # Model
    cgan = ConditionalGAN(config, device)
    G = cgan.generator
    D = cgan.discriminator

    # Wrap models in DDP
    if distributed:
        G = DDP(G, device_ids=[local_rank] if torch.cuda.is_available() else None)
        D = DDP(D, device_ids=[local_rank] if torch.cuda.is_available() else None)
    # Keep references to the underlying module for checkpointing / EMA
    G_module = G.module if distributed else G
    D_module = D.module if distributed else D

    # Print model summaries
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    if is_main_process():
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

    # Learning rate scheduler: linear decay over the last decay_fraction of training.
    # Set lr_decay_start >= 1.0 to disable decay (constant LR throughout).
    decay_start = train_cfg.get("lr_decay_start", 0.5)  # fraction of training
    num_epochs = train_cfg["num_epochs"]
    decay_start_epoch = int(num_epochs * decay_start)

    def lr_lambda(epoch):
        if decay_start_epoch >= num_epochs or epoch < decay_start_epoch:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start_epoch) / (num_epochs - decay_start_epoch))

    scheduler_g = LambdaLR(optimizer_g, lr_lambda)
    scheduler_d = LambdaLR(optimizer_d, lr_lambda)

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

    # --- Configuration for improved training ---
    loss_type = train_cfg.get("loss_type", "vanilla")
    d_loss_fn = get_d_loss_fn(loss_type)
    g_loss_fn = get_g_loss_fn(loss_type)
    d_steps = train_cfg.get("d_steps", 1)
    g_steps = train_cfg.get("g_steps", 1)
    gp_lambda = train_cfg.get("gp_lambda", 10.0)
    use_gp = (loss_type == "wgan-gp")
    r1_gamma = train_cfg.get("r1_gamma", 0.0)
    use_r1 = (r1_gamma > 0)
    grad_clip_g = train_cfg.get("grad_clip_g", 0.0)
    grad_clip_d = train_cfg.get("grad_clip_d", 0.0)

    # Label smoothing (only applicable to vanilla loss)
    label_smooth_real = train_cfg.get("label_smooth_real", 1.0)
    label_smooth_fake = train_cfg.get("label_smooth_fake", 0.0)

    # EMA (on the unwrapped module)
    use_ema = train_cfg.get("use_ema", False)
    ema_decay = train_cfg.get("ema_decay", 0.999)
    G_ema = None
    if use_ema:
        G_ema = create_ema(G_module)
        if is_main_process():
            print(f"EMA enabled (decay={ema_decay})")

    # Resume from checkpoint (must come after G_ema is created)
    start_epoch = 0
    if resume_path is not None:
        start_epoch, saved_history = cgan.load_checkpoint(
            resume_path, optimizer_g, optimizer_d, G_ema=G_ema
        )
        if saved_history:
            history = saved_history
        if is_main_process():
            print(f"Resumed from epoch {start_epoch}")

    if is_main_process():
        print(f"Loss type: {loss_type}")
        print(f"D steps: {d_steps}, G steps: {g_steps}")
        if use_gp:
            print(f"Gradient penalty lambda: {gp_lambda}")
        if use_r1:
            print(f"R1 gradient penalty gamma: {r1_gamma}")

    # Training loop
    eval_cfg = config["evaluation"]

    for epoch in range(start_epoch, num_epochs):
        # Set epoch on distributed sampler for proper shuffling
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        G.train()
        D.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_d_real_acc = 0.0
        epoch_d_fake_acc = 0.0
        num_batches = 0
        g_updates = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}",
                    disable=not is_main_process())
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

            # Discriminator loss
            if loss_type == "vanilla":
                d_loss, _, _ = d_loss_fn(
                    real_output, fake_output,
                    real_label=label_smooth_real,
                    fake_label=label_smooth_fake,
                )
            else:
                d_loss, _, _ = d_loss_fn(real_output, fake_output)

            # Gradient penalty for WGAN-GP
            if use_gp:
                gp = gradient_penalty(
                    D, real_images, fake_images.detach(), real_labels,
                    device, lambda_gp=gp_lambda,
                )
                d_loss = d_loss + gp

            # R1 gradient penalty on real images (prevents D collapse on reals)
            if use_r1:
                r1 = r1_gradient_penalty(D, real_images, real_labels, device,
                                         gamma=r1_gamma)
                d_loss = d_loss + r1

            d_loss.backward()
            if grad_clip_d > 0:
                torch.nn.utils.clip_grad_norm_(D.parameters(), grad_clip_d)
            optimizer_d.step()

            # Discriminator accuracy
            # For vanilla/lsgan: sigmoid-based (output is logit for BCE)
            # For hinge/wgan-gp: sign-based (output is unbounded critic score)
            with torch.no_grad():
                if loss_type in ("hinge", "wgan-gp"):
                    # Critic is correct when D(real) > 0 and D(fake) < 0
                    d_real_acc = (real_output > 0).float().mean().item()
                    d_fake_acc = (fake_output < 0).float().mean().item()
                else:
                    d_real_acc = (torch.sigmoid(real_output) > 0.5).float().mean().item()
                    d_fake_acc = (torch.sigmoid(fake_output) < 0.5).float().mean().item()

            epoch_d_loss += d_loss.item()
            epoch_d_real_acc += d_real_acc
            epoch_d_fake_acc += d_fake_acc
            num_batches += 1

            # =============================================
            # Train Generator (every d_steps batches)
            # =============================================
            if num_batches % d_steps == 0:
                for _ in range(g_steps):
                    optimizer_g.zero_grad()

                    # Generate fresh fakes
                    z = torch.randn(batch_size, latent_dim, device=device)
                    fake_labels = torch.randint(0, num_classes, (batch_size,),
                                                device=device)
                    fake_images = G(z, fake_labels)

                    # D on fakes (no detach — gradients flow into G)
                    fake_output = D(fake_images, fake_labels)

                    # Generator loss
                    g_loss = g_loss_fn(fake_output)
                    g_loss.backward()
                    if grad_clip_g > 0:
                        torch.nn.utils.clip_grad_norm_(G.parameters(), grad_clip_g)
                    optimizer_g.step()

                    # Update EMA (on unwrapped module)
                    if use_ema:
                        update_ema(G_ema, G_module, ema_decay)

                epoch_g_loss += g_loss.item()
                g_updates += 1

            # Logging
            if num_batches % eval_cfg["log_every"] == 0:
                pbar.set_postfix({
                    "D": f"{d_loss.item():.4f}",
                    "G": f"{g_loss.item():.4f}" if g_updates > 0 else "N/A",
                    "D_r": f"{d_real_acc:.2f}",
                    "D_f": f"{d_fake_acc:.2f}",
                })

        # End-of-epoch averages
        avg_d = epoch_d_loss / num_batches
        avg_d_real = epoch_d_real_acc / num_batches
        avg_d_fake = epoch_d_fake_acc / num_batches
        avg_g = epoch_g_loss / max(g_updates, 1)

        history["g_losses"].append(avg_g)
        history["d_losses"].append(avg_d)
        history["d_real_acc"].append(avg_d_real)
        history["d_fake_acc"].append(avg_d_fake)

        current_lr_g = scheduler_g.get_last_lr()[0]
        current_lr_d = scheduler_d.get_last_lr()[0]
        if is_main_process():
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"D_loss: {avg_d:.4f}  G_loss: {avg_g:.4f}  "
                f"D_acc(real): {avg_d_real:.2f}  D_acc(fake): {avg_d_fake:.2f}  "
                f"lr_g: {current_lr_g:.6f}  lr_d: {current_lr_d:.6f}"
            )

        # Step learning rate schedulers
        scheduler_g.step()
        scheduler_d.step()

        # Only rank 0 does visualization and checkpointing
        if is_main_process():
            # Generate sample grid (use EMA generator if available)
            if (epoch + 1) % eval_cfg["sample_every"] == 0:
                gen_for_viz = G_ema if use_ema else G_module
                gen_for_viz.eval()
                with torch.no_grad():
                    samples = gen_for_viz(fixed_noise, fixed_labels)
                save_image_grid(samples, fixed_labels, epoch + 1, results_dir,
                                nrow=num_per_class)

            # Save checkpoint (using unwrapped modules via cgan)
            if (epoch + 1) % eval_cfg["save_every"] == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"
                )
                cgan.save_checkpoint(ckpt_path, epoch + 1, optimizer_g,
                                     optimizer_d, history, G_ema=G_ema)
                print(f"  Checkpoint saved: {ckpt_path}")

        if distributed:
            dist.barrier()

    # Final checkpoint and outputs (rank 0 only)
    if is_main_process():
        final_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        cgan.save_checkpoint(final_path, num_epochs, optimizer_g, optimizer_d,
                             history, G_ema=G_ema)
        print(f"Final checkpoint saved: {final_path}")

        # Save training curves
        plot_training_curves(history, results_dir)

        # Save history as JSON
        history_path = os.path.join(results_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved: {history_path}")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Train Conditional GAN")
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
