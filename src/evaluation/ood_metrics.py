"""OOD detection evaluation metrics (AUROC, FPR@TPR)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Area Under the ROC Curve for OOD detection.

    Args:
        scores: OOD scores of shape (N,). Higher score = more OOD.
        labels: Binary labels of shape (N,). 0 = in-distribution, 1 = OOD.

    Returns:
        AUROC value in [0, 1].
    """
    return float(roc_auc_score(labels, scores))


def compute_fpr_at_tpr(
    scores: np.ndarray,
    labels: np.ndarray,
    tpr_target: float = 0.95,
) -> float:
    """Compute False Positive Rate at a given True Positive Rate threshold.

    Args:
        scores: OOD scores of shape (N,). Higher score = more OOD.
        labels: Binary labels of shape (N,). 0 = in-distribution, 1 = OOD.
        tpr_target: Target TPR (default 0.95).

    Returns:
        FPR value at the point where TPR first reaches tpr_target.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    # Find the first index where TPR >= tpr_target
    idx = np.searchsorted(tpr, tpr_target, side="left")
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])
