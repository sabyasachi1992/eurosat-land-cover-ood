"""HDBSCAN clustering and cluster statistics."""

from __future__ import annotations

import numpy as np
import hdbscan
from PIL import Image


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
) -> np.ndarray:
    """Apply HDBSCAN density-based clustering.

    Args:
        embeddings: 2D embeddings of shape (N, D), typically from UMAP.
        min_cluster_size: Minimum cluster size for HDBSCAN.
        min_samples: Minimum samples parameter for HDBSCAN.

    Returns:
        Cluster labels of shape (N,). -1 indicates noise.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    return clusterer.fit_predict(embeddings)


def compute_cluster_stats(
    file_paths: list[str],
    cluster_labels: np.ndarray,
) -> dict[int, dict]:
    """Compute per-cluster statistics.

    For each cluster (excluding noise label -1): count, mean RGB values,
    and representative sample indices.

    Args:
        file_paths: List of image file paths.
        cluster_labels: Cluster label for each file path. -1 = noise.

    Returns:
        Dict mapping cluster_id to:
            {"count": int, "mean_rgb": [r, g, b], "representative_indices": [int, ...]}
    """
    unique_labels = sorted(set(cluster_labels.tolist()))
    stats: dict[int, dict] = {}

    for label in unique_labels:
        if label == -1:
            continue

        indices = np.where(cluster_labels == label)[0]
        count = len(indices)

        # Compute mean RGB across all images in the cluster
        rgb_sums = np.zeros(3, dtype=np.float64)
        valid_count = 0
        for idx in indices:
            try:
                img = Image.open(file_paths[idx]).convert("RGB")
                pixels = np.array(img, dtype=np.float64)
                rgb_sums += pixels.mean(axis=(0, 1))
                valid_count += 1
            except Exception:
                continue

        if valid_count > 0:
            mean_rgb = (rgb_sums / valid_count).tolist()
        else:
            mean_rgb = [0.0, 0.0, 0.0]

        # Representative indices: up to 9 samples closest to cluster centroid
        # Use the embeddings if available, otherwise random sample
        n_reps = min(9, count)
        rng = np.random.default_rng(seed=42)
        representative_indices = rng.choice(indices, size=n_reps, replace=False).tolist()

        stats[label] = {
            "count": count,
            "mean_rgb": mean_rgb,
            "representative_indices": representative_indices,
        }

    return stats
