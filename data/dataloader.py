"""Dataset loading and preprocessing for CIFAR-10."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TIN_MEAN = (0.5, 0.5, 0.5)
TIN_STD = (0.5, 0.5, 0.5)


def get_train_transform(config):
    """Training transform: config-driven augmentation + normalization to [-1, 1].

    Args:
        config: dict loaded from YAML config file.
    """
    aug_cfg = config["data"]["augmentation"]
    image_size = config["model"]["image_size"]
    transform_list = []

    # RandomResizedCrop (must come before ToTensor)
    if aug_cfg.get("random_resized_crop", False):
        scale_min = aug_cfg.get("random_resized_crop_scale_min", 0.8)
        scale_max = aug_cfg.get("random_resized_crop_scale_max", 1.0)
        transform_list.append(
            transforms.RandomResizedCrop(
                image_size, scale=(scale_min, scale_max)
            )
        )

    # RandomHorizontalFlip
    if aug_cfg.get("horizontal_flip", False):
        p = aug_cfg.get("horizontal_flip_p", 0.5)
        transform_list.append(transforms.RandomHorizontalFlip(p=p))

    # RandomRotation
    if aug_cfg.get("random_rotation", False):
        degrees = aug_cfg.get("random_rotation_degrees", 10)
        transform_list.append(transforms.RandomRotation(degrees=degrees))

    # ColorJitter
    if aug_cfg.get("color_jitter", False):
        transform_list.append(transforms.ColorJitter(
            brightness=aug_cfg.get("color_jitter_brightness", 0.2),
            contrast=aug_cfg.get("color_jitter_contrast", 0.2),
            saturation=aug_cfg.get("color_jitter_saturation", 0.2),
            hue=aug_cfg.get("color_jitter_hue", 0.1),
        ))

    # ToTensor + Normalize (always applied)
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=TIN_MEAN, std=TIN_STD))

    return transforms.Compose(transform_list)


def get_test_transform():
    """Test transform: normalization only (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=TIN_MEAN, std=TIN_STD),
    ])


def get_dataloaders(config):
    """Create train and test DataLoaders for CIFAR-10.

    Args:
        config: dict loaded from YAML config file.

    Returns:
        (train_loader, test_loader) tuple of DataLoader objects.
    """
    data_cfg = config["data"]
    data_root = data_cfg["data_root"]
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True)
    batch_size = config["training"]["batch_size"]

    train_transform = get_train_transform(config)
    test_transform = get_test_transform()

    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
