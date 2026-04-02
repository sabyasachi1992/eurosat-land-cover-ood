"""UMAP dimensionality reduction for feature embeddings."""

from __future__ import annotations

import numpy as np
import umap


def reduce_umap(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Apply UMAP dimensionality reduction.

    Args:
        features: Feature matrix of shape (N, D).
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance parameter for UMAP.
        n_components: Target dimensionality.
        random_state: Random seed for reproducibility.

    Returns:
        Reduced embeddings of shape (N, n_components).
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    return reducer.fit_transform(features)
