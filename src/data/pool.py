"""Unlabeled pool construction for OOD evaluation."""

from __future__ import annotations

import random

from src.data.dataset import discover_images
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_unlabeled_pool(
    known_test_paths: list[str],
    known_test_labels: list[int],
    ghost_root: str,
    ghost_classes: list[str],
    n_known_samples: int,
    seed: int,
) -> tuple[list[str], list[int]]:
    """Build a shuffled unlabeled pool mixing known and ghost images.

    Args:
        known_test_paths: File paths from the known-class test split.
        known_test_labels: Corresponding labels (unused beyond sampling).
        ghost_root: Root directory containing ghost-class subdirectories.
        ghost_classes: Ordered list of ghost-class directory names.
        n_known_samples: Number of known-class samples to include.
        seed: Random seed for reproducibility.

    Returns:
        ``(shuffled_paths, ground_truth_labels)`` where ground-truth
        labels are ``0`` for in-distribution (known) and ``1`` for OOD
        (ghost).  These labels are kept separate and used only for
        final evaluation.
    """
    # Sample known test paths
    rng = random.Random(seed)
    sampled_known = rng.sample(known_test_paths, min(n_known_samples, len(known_test_paths)))

    # Discover all ghost-class images
    ghost_paths, _ = discover_images(ghost_root, ghost_classes)

    # Combine
    pool_paths = list(sampled_known) + list(ghost_paths)
    ground_truth = [0] * len(sampled_known) + [1] * len(ghost_paths)

    # Shuffle together deterministically
    combined = list(zip(pool_paths, ground_truth))
    rng2 = random.Random(seed)
    rng2.shuffle(combined)

    shuffled_paths = [p for p, _ in combined]
    shuffled_labels = [l for _, l in combined]

    logger.info(
        "Built unlabeled pool: %d known + %d ghost = %d total",
        len(sampled_known),
        len(ghost_paths),
        len(shuffled_paths),
    )

    return shuffled_paths, shuffled_labels
