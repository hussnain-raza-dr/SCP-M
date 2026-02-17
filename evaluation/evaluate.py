"""Evaluation script: generate samples, compute metrics, plot diagnostics."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import get_dataloaders
from models.cgan import ConditionalGAN
from evaluation.visualize import create_full_visualization

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def evaluate_discriminator(discriminator, generator, dataloader, config,
                           device):
    """Compute discriminator accuracy (overall and per-class).

    Returns:
        dict with overall and per-class accuracy, plus raw confidence lists.
    """
    discriminator.eval()
    generator.eval()

    latent_dim = config["model"]["latent_dim"]
    num_classes = config["model"]["num_classes"]

    # Per-class counters
    real_correct = np.zeros(num_classes)
    real_total = np.zeros(num_classes)
    fake_correct = np.zeros(num_classes)
    fake_total = np.zeros(num_classes)

    # Confidence scores for histograms
    real_confidences = []
    fake_confidences = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # Real images
            real_out = discriminator(images, labels)
            real_probs = torch.sigmoid(real_out).squeeze()
            real_confidences.append(real_probs.cpu().numpy())

            real_preds = (real_probs > 0.5).float()
            for c in range(num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    real_correct[c] += real_preds[mask].sum().item()
                    real_total[c] += mask.sum().item()

            # Fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (batch_size,),
                                        device=device)
            fake_images = generator(z, fake_labels)
            fake_out = discriminator(fake_images, fake_labels)
            fake_probs = torch.sigmoid(fake_out).squeeze()
            fake_confidences.append(fake_probs.cpu().numpy())

            fake_preds = (fake_probs < 0.5).float()
            for c in range(num_classes):
                mask = (fake_labels == c)
                if mask.sum() > 0:
                    fake_correct[c] += fake_preds[mask].sum().item()
                    fake_total[c] += mask.sum().item()

    # Aggregate
    real_acc_per_class = np.where(real_total > 0, real_correct / real_total, 0)
    fake_acc_per_class = np.where(fake_total > 0, fake_correct / fake_total, 0)

    overall_real = real_correct.sum() / real_total.sum()
    overall_fake = fake_correct.sum() / fake_total.sum()
    overall_total = (real_correct.sum() + fake_correct.sum()) / (
        real_total.sum() + fake_total.sum()
    )

    return {
        "real_accuracy": float(overall_real),
        "fake_accuracy": float(overall_fake),
        "total_accuracy": float(overall_total),
        "real_acc_per_class": real_acc_per_class.tolist(),
        "fake_acc_per_class": fake_acc_per_class.tolist(),
        "real_confidences": np.concatenate(real_confidences),
        "fake_confidences": np.concatenate(fake_confidences),
    }


def compute_generation_quality(generator, discriminator, config, device,
                               num_samples_per_class=500):
    """Compute per-class mean D score for generated images.

    Lower D score (closer to 0) means G is better at fooling D for that class.
    """
    generator.eval()
    discriminator.eval()

    latent_dim = config["model"]["latent_dim"]
    num_classes = config["model"]["num_classes"]

    mean_scores = []
    with torch.no_grad():
        for c in range(num_classes):
            z = torch.randn(num_samples_per_class, latent_dim, device=device)
            labels = torch.full((num_samples_per_class,), c, dtype=torch.long,
                                device=device)
            fake_images = generator(z, labels)
            scores = torch.sigmoid(discriminator(fake_images, labels)).squeeze()
            mean_scores.append(scores.mean().item())

    return mean_scores


def compute_output_stats(generator, config, device, num_samples=1000):
    """Compute pixel-level statistics of generated images."""
    generator.eval()
    latent_dim = config["model"]["latent_dim"]
    num_classes = config["model"]["num_classes"]

    z = torch.randn(num_samples, latent_dim, device=device)
    labels = torch.randint(0, num_classes, (num_samples,), device=device)

    with torch.no_grad():
        images = generator(z, labels)

    return {
        "pixel_mean": images.mean().item(),
        "pixel_std": images.std().item(),
        "pixel_min": images.min().item(),
        "pixel_max": images.max().item(),
    }


def plot_confusion_matrix(real_acc, fake_acc, class_names, save_dir):
    """Plot discriminator accuracy per class as a heatmap.

    Rows: classes. Columns: [Real Acc, Fake Acc].
    """
    data = np.array([real_acc, fake_acc]).T  # (num_classes, 2)

    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Real Acc", "Fake Acc"])
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_title("Discriminator Accuracy per Class")

    # Annotate cells
    for i in range(len(class_names)):
        for j in range(2):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=10)

    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "discriminator_confusion_matrix.png"),
                dpi=150)
    plt.close()


def plot_confidence_histogram(real_confidences, fake_confidences, save_dir):
    """Plot overlapping histograms of D's confidence on real vs fake images."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(real_confidences, bins=50, alpha=0.6, label="Real", color="green",
            density=True)
    ax.hist(fake_confidences, bins=50, alpha=0.6, label="Fake", color="red",
            density=True)

    ax.set_xlabel("D(x) confidence (sigmoid output)")
    ax.set_ylabel("Density")
    ax.set_title("Discriminator Confidence Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confidence_histogram.png"), dpi=150)
    plt.close()


def plot_per_class_generation_quality(mean_scores, class_names, save_dir):
    """Bar chart of mean D score per class for generated images."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(class_names))
    bars = ax.bar(x, mean_scores, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Mean D(G(z)) score")
    ax.set_title("Per-class Generation Quality (lower = G fools D better)")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, score in zip(bars, mean_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{score:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "generation_quality_per_class.png"),
                dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained cGAN")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output", type=str, default="./results/evaluation",
        help="Directory to save evaluation outputs",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    cgan = ConditionalGAN(config, device)
    epoch, _ = cgan.load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint from epoch {epoch}")

    os.makedirs(args.output, exist_ok=True)

    # 1. Discriminator accuracy (overall + per-class)
    _, test_loader = get_dataloaders(config)
    results = evaluate_discriminator(
        cgan.discriminator, cgan.generator, test_loader, config, device
    )

    print(f"\nDiscriminator accuracy:")
    print(f"  Real:  {results['real_accuracy']:.4f}")
    print(f"  Fake:  {results['fake_accuracy']:.4f}")
    print(f"  Total: {results['total_accuracy']:.4f}")

    num_classes = config["model"]["num_classes"]
    class_names = CIFAR10_CLASSES[:num_classes]

    print(f"\nPer-class accuracy:")
    for i, name in enumerate(class_names):
        print(f"  {name:12s}  real={results['real_acc_per_class'][i]:.3f}  "
              f"fake={results['fake_acc_per_class'][i]:.3f}")

    # 2. Confusion matrix plot
    plot_confusion_matrix(
        results["real_acc_per_class"], results["fake_acc_per_class"],
        class_names, args.output,
    )
    print(f"\nSaved: discriminator_confusion_matrix.png")

    # 3. Confidence histogram
    plot_confidence_histogram(
        results["real_confidences"], results["fake_confidences"], args.output,
    )
    print(f"Saved: confidence_histogram.png")

    # 4. Per-class generation quality
    gen_scores = compute_generation_quality(
        cgan.generator, cgan.discriminator, config, device,
    )
    plot_per_class_generation_quality(gen_scores, class_names, args.output)
    print(f"Saved: generation_quality_per_class.png")

    print(f"\nPer-class mean D(G(z)) score:")
    for name, score in zip(class_names, gen_scores):
        print(f"  {name:12s}  {score:.4f}")

    # 5. Output distribution stats
    stats = compute_output_stats(cgan.generator, config, device)
    print(f"\nGenerated image pixel stats:")
    print(f"  Mean: {stats['pixel_mean']:.4f}")
    print(f"  Std:  {stats['pixel_std']:.4f}")
    print(f"  Min:  {stats['pixel_min']:.4f}")
    print(f"  Max:  {stats['pixel_max']:.4f}")

    # 6. Full visualizations (grids, interpolation, class variation)
    create_full_visualization(
        cgan.generator,
        save_dir=args.output,
        latent_dim=config["model"]["latent_dim"],
        num_classes=config["model"]["num_classes"],
        device=device,
    )

    # Save all metrics as JSON
    metrics = {
        "real_accuracy": results["real_accuracy"],
        "fake_accuracy": results["fake_accuracy"],
        "total_accuracy": results["total_accuracy"],
        "real_acc_per_class": dict(zip(class_names,
                                       results["real_acc_per_class"])),
        "fake_acc_per_class": dict(zip(class_names,
                                       results["fake_acc_per_class"])),
        "generation_quality_per_class": dict(zip(class_names, gen_scores)),
        "output_stats": stats,
    }
    with open(os.path.join(args.output, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nAll metrics saved to {args.output}/evaluation_metrics.json")


if __name__ == "__main__":
    main()
