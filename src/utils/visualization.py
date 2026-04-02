"""Plotting helpers for training curves, confusion matrices, OOD histograms, etc."""

from __future__ import annotations

import math
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend by default
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


def plot_training_curves(
    history: dict,
    save_path: Optional[str] = None,
) -> None:
    """Plot train/val loss and accuracy curves.

    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: If provided, save figure to this path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_lr_curve(
    history: dict,
    save_path: Optional[str] = None,
) -> None:
    """Plot learning rate vs epoch.

    Args:
        history: Dict with key 'lr' (list of learning rates per epoch).
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(history["lr"]) + 1)
    ax.plot(epochs, history["lr"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix as a seaborn heatmap.

    Args:
        cm: Confusion matrix of shape (K, K).
        class_names: List of class names for axis labels.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_ood_histograms(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    method_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot overlapping histograms of ID vs OOD scores.

    Args:
        id_scores: OOD scores for in-distribution samples.
        ood_scores: OOD scores for out-of-distribution samples.
        method_name: Name of the OOD method (for title).
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(id_scores, bins=50, alpha=0.6, label="In-Distribution", density=True)
    ax.hist(ood_scores, bins=50, alpha=0.6, label="OOD", density=True)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title(f"OOD Score Distribution — {method_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_umap_clusters(
    embeddings_2d: np.ndarray,
    cluster_labels: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Scatter plot of 2D UMAP embeddings colored by cluster label.

    Args:
        embeddings_2d: Array of shape (N, 2).
        cluster_labels: Cluster labels of shape (N,). -1 = noise.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = sorted(set(cluster_labels.tolist()))
    cmap = plt.cm.get_cmap("tab10", max(len(unique_labels), 1))

    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        color = "lightgray" if label == -1 else cmap(i)
        name = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=name,
            s=5,
            alpha=0.6,
        )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP Clusters")
    ax.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_patch_grid(
    file_paths: list[str],
    title: str = "",
    n_cols: int = 5,
    save_path: Optional[str] = None,
) -> None:
    """Display a grid of image patches.

    Args:
        file_paths: Paths to image files to display.
        title: Title for the figure.
        n_cols: Number of columns in the grid.
        save_path: If provided, save figure to this path.
    """
    n = len(file_paths)
    if n == 0:
        return
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax = axes[i, j]
            if idx < n:
                try:
                    img = Image.open(file_paths[idx]).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Error", ha="center", va="center")
            ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_misclassified(
    file_paths: list[str],
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
    n_samples: int = 9,
    save_path: Optional[str] = None,
) -> None:
    """Display a grid of misclassified examples with true/predicted labels.

    Args:
        file_paths: Paths to image files.
        true_labels: Ground-truth labels.
        pred_labels: Predicted labels.
        class_names: Human-readable class names.
        n_samples: Maximum number of misclassified samples to show.
        save_path: If provided, save figure to this path.
    """
    # Find misclassified indices
    misclassified = [
        i for i in range(len(true_labels)) if true_labels[i] != pred_labels[i]
    ]

    if not misclassified:
        print("No misclassified samples found.")
        return

    # Limit to n_samples
    show = misclassified[:n_samples]
    n = len(show)
    n_cols = min(3, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(n_rows):
        for j in range(n_cols):
            idx_in_show = i * n_cols + j
            ax = axes[i, j]
            if idx_in_show < n:
                sample_idx = show[idx_in_show]
                try:
                    img = Image.open(file_paths[sample_idx]).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Error", ha="center", va="center")
                true_name = class_names[true_labels[sample_idx]]
                pred_name = class_names[pred_labels[sample_idx]]
                ax.set_title(f"T: {true_name}\nP: {pred_name}", fontsize=9)
            ax.axis("off")

    fig.suptitle("Misclassified Examples", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
