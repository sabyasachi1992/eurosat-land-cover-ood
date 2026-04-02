"""Mahalanobis distance-based OOD scoring."""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from src.ood.features import FeatureExtractor


def fit_class_gaussians(
    model: nn.Module,
    train_loader: DataLoader,
    feature_layer: str,
    device: str,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Fit class-conditional Gaussians on training features.

    Extracts features from the training data, computes per-class mean
    vectors and a shared (tied) covariance matrix across all classes.

    Args:
        model: Trained classifier.
        train_loader: DataLoader for training data yielding (images, labels).
        feature_layer: Name of the intermediate layer to extract features from.
        device: Device string.

    Returns:
        Tuple of (class_means, shared_covariance) where:
            class_means: dict mapping class label (int) to mean feature vector.
            shared_covariance: shared covariance matrix with ε·I regularization.
    """
    extractor = FeatureExtractor(model, feature_layer)
    features = extractor.extract(train_loader, device)  # (N, D)

    # Collect labels from the dataloader
    all_labels: list[int] = []
    for batch in train_loader:
        labels = batch[1]
        all_labels.extend(labels.numpy().tolist())
    labels_arr = np.array(all_labels)

    unique_classes = sorted(set(labels_arr.tolist()))
    D = features.shape[1]

    # Per-class means
    class_means: dict[int, np.ndarray] = {}
    for c in unique_classes:
        mask = labels_arr == c
        class_means[c] = features[mask].mean(axis=0)

    # Shared (tied) covariance: pool centered features from all classes
    centered = np.zeros_like(features)
    for c in unique_classes:
        mask = labels_arr == c
        centered[mask] = features[mask] - class_means[c]

    shared_cov = (centered.T @ centered) / len(features)

    # Regularization: add ε·I
    eps = 1e-6
    shared_cov += eps * np.eye(D)

    return class_means, shared_cov


def compute_mahalanobis_scores(
    model: nn.Module,
    dataloader: DataLoader,
    feature_layer: str,
    class_means: dict[int, np.ndarray],
    shared_cov: np.ndarray,
    device: str,
) -> np.ndarray:
    """Compute Mahalanobis distance scores for OOD detection.

    For each sample, computes the minimum Mahalanobis distance to any
    class mean. Higher distance = more OOD.

    Args:
        model: Trained classifier.
        dataloader: DataLoader for evaluation data.
        feature_layer: Name of the intermediate layer to extract features from.
        class_means: Per-class mean vectors from fit_class_gaussians.
        shared_cov: Shared covariance matrix from fit_class_gaussians.
        device: Device string.

    Returns:
        Array of Mahalanobis scores with shape (N,).
    """
    extractor = FeatureExtractor(model, feature_layer)
    features = extractor.extract(dataloader, device)  # (N, D)

    # Invert the shared covariance
    cov_inv = np.linalg.inv(shared_cov)

    N = features.shape[0]
    scores = np.full(N, np.inf)

    for c, mean_vec in class_means.items():
        diff = features - mean_vec  # (N, D)
        # Mahalanobis distance: sqrt((x - mu)^T Sigma^{-1} (x - mu))
        left = diff @ cov_inv  # (N, D)
        dists = np.sqrt(np.sum(left * diff, axis=1))  # (N,)
        scores = np.minimum(scores, dists)

    return scores
