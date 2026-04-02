"""EuroSAT dataset loading, image discovery, and stratified splitting."""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.utils.logging import get_logger

logger = get_logger(__name__)


class EuroSATDataset(Dataset):
    """PyTorch Dataset for EuroSAT RGB patches.

    Args:
        file_paths: List of absolute or relative paths to .jpg images.
        labels: Integer class labels corresponding to each file path.
        transform: Optional callable applied to each PIL image.
    """

    def __init__(
        self,
        file_paths: list[str],
        labels: list[int],
        transform: Optional[Callable] = None,
    ) -> None:
        if len(file_paths) != len(labels):
            raise ValueError(
                f"file_paths length ({len(file_paths)}) != labels length ({len(labels)})"
            )
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load image at *idx*, apply transform, return (tensor, label).

        If the image is corrupt or unreadable, log a warning and try the
        next valid index (wrapping around).  Raises ``RuntimeError`` only
        if *every* image in the dataset is unreadable.
        """
        for offset in range(len(self)):
            real_idx = (idx + offset) % len(self)
            path = self.file_paths[real_idx]
            try:
                img = Image.open(path).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                return img, self.labels[real_idx]
            except Exception as exc:
                logger.warning("Skipping corrupt/unreadable image %s: %s", path, exc)
        raise RuntimeError("All images in the dataset are unreadable")


def discover_images(
    root: str, class_names: list[str]
) -> tuple[list[str], list[int]]:
    """Walk class directories under *root* and collect .jpg file paths.

    Args:
        root: Path to the dataset root (e.g. ``EuroSAT/2750``).
        class_names: Ordered list of class directory names.  The position
            in the list determines the integer label (0-indexed).

    Returns:
        Tuple of ``(file_paths, labels)`` where each entry corresponds
        to a discovered .jpg image.

    Raises:
        FileNotFoundError: If a class directory does not exist.
        ValueError: If a class directory exists but contains no valid images.
    """
    file_paths: list[str] = []
    labels: list[int] = []

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        class_files: list[str] = []
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(".jpg"):
                fpath = os.path.join(class_dir, fname)
                # Quick readability check
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                    class_files.append(fpath)
                except Exception as exc:
                    logger.warning("Skipping corrupt/unreadable file %s: %s", fpath, exc)

        if not class_files:
            raise ValueError(
                f"No valid .jpg images found in class directory: {class_dir}"
            )

        file_paths.extend(class_files)
        labels.extend([label] * len(class_files))

    return file_paths, labels


def stratified_split(
    file_paths: list[str],
    labels: list[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str], list[int], list[int], list[int]]:
    """Two-step stratified split into train / val / test sets.

    Step 1: Split off the test portion.
    Step 2: Split the remainder into train and val.

    Args:
        file_paths: Image file paths.
        labels: Corresponding integer labels.
        train_ratio: Fraction for training (e.g. 0.70).
        val_ratio: Fraction for validation (e.g. 0.15).
        test_ratio: Fraction for testing (e.g. 0.15).
        seed: Random seed for reproducibility.

    Returns:
        ``(train_paths, val_paths, test_paths,
          train_labels, val_labels, test_labels)``
    """
    # Step 1: split off test
    remaining_paths, test_paths, remaining_labels, test_labels = train_test_split(
        file_paths,
        labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )

    # Step 2: split remainder into train / val
    # val_ratio relative to the remaining data
    val_relative = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        remaining_paths,
        remaining_labels,
        test_size=val_relative,
        random_state=seed,
        stratify=remaining_labels,
    )

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels
