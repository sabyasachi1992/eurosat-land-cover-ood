"""Normalization statistics and transform pipelines for EuroSAT images."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision import transforms

from src.config import Config
from src.data.dataset import EuroSATDataset
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_norm_stats(
    dataset: EuroSATDataset,
) -> tuple[list[float], list[float]]:
    """Compute per-channel mean and std over all images in *dataset*.

    The dataset should be the **training** split with no augmentation
    (only ``ToTensor`` so pixel values are in [0, 1]).

    Args:
        dataset: An ``EuroSATDataset`` whose transform converts images
            to tensors (e.g. ``transforms.ToTensor()``).

    Returns:
        ``([mean_r, mean_g, mean_b], [std_r, std_g, std_b])``
    """
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for idx in range(len(dataset)):
        img_tensor, _ = dataset[idx]  # (C, H, W)
        img = img_tensor.to(torch.float64)
        channel_sum += img.sum(dim=(1, 2))
        channel_sq_sum += (img ** 2).sum(dim=(1, 2))
        pixel_count += img.shape[1] * img.shape[2]

    mean = (channel_sum / pixel_count).tolist()
    std = ((channel_sq_sum / pixel_count - torch.tensor(mean, dtype=torch.float64) ** 2).sqrt()).tolist()

    logger.info("Computed norm stats — mean: %s, std: %s", mean, std)
    return mean, std


def save_norm_stats(mean: list[float], std: list[float], path: str) -> None:
    """Persist normalization statistics to a JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"mean": mean, "std": std}, f)
    logger.info("Saved norm stats to %s", path)


def load_norm_stats(path: str) -> tuple[list[float], list[float]]:
    """Load normalization statistics from a JSON file.

    Returns:
        ``(mean, std)`` each as a list of 3 floats.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data["mean"], data["std"]


def get_train_transform(
    config: Config, mean: list[float], std: list[float]
) -> transforms.Compose:
    """Build the training transform pipeline.

    Order: ``ToTensor`` → config-toggled augmentations → ``Normalize``.

    Each augmentation is included only if enabled in ``config.augmentation``.
    """
    ops: list[transforms.transforms.Transform] = [transforms.ToTensor()]

    aug = config.augmentation

    if aug.get("horizontal_flip"):
        ops.append(transforms.RandomHorizontalFlip(p=0.5))

    if aug.get("vertical_flip"):
        ops.append(transforms.RandomVerticalFlip(p=0.5))

    rotation = aug.get("random_rotation")
    if rotation:
        ops.append(transforms.RandomRotation(degrees=rotation))

    jitter = aug.get("color_jitter")
    if jitter:
        ops.append(
            transforms.ColorJitter(
                brightness=jitter.get("brightness", 0),
                contrast=jitter.get("contrast", 0),
                saturation=jitter.get("saturation", 0),
                hue=jitter.get("hue", 0),
            )
        )

    ops.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(ops)


def get_eval_transform(
    mean: list[float], std: list[float]
) -> transforms.Compose:
    """Build the evaluation transform pipeline (ToTensor + Normalize only)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
