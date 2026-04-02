"""Classification evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def compute_classification_report(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> dict:
    """Compute per-class precision, recall, F1 and macro averages.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: Human-readable class names (order matches label indices).

    Returns:
        Dictionary with per-class metrics and macro averages
        (sklearn classification_report with output_dict=True).
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def compute_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> np.ndarray:
    """Compute confusion matrix as a numpy array.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: Human-readable class names (used to determine label count).

    Returns:
        Confusion matrix of shape (K, K) where K = len(class_names).
    """
    labels = list(range(len(class_names)))
    return confusion_matrix(y_true, y_pred, labels=labels)
