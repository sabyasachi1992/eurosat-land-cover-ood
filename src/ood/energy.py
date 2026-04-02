"""Energy-based OOD scoring."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_energy_scores(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> np.ndarray:
    """Compute energy scores for all samples in the dataloader.

    Energy = -log(sum(exp(logits)))  (negative logsumexp).
    Lower energy = in-distribution, higher energy = OOD.

    Args:
        model: Trained classifier producing logits.
        dataloader: DataLoader yielding (images, labels) or (images,) batches.
        device: Device string (e.g. "cpu", "cuda").

    Returns:
        Array of energy scores with shape (N,).
    """
    model.eval()
    model.to(device)
    scores: list[torch.Tensor] = []

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        images = images.to(device)
        logits = model(images)
        # Energy = -logsumexp(logits, dim=1)
        energy = -torch.logsumexp(logits, dim=1)
        scores.append(energy.cpu())

    return torch.cat(scores, dim=0).numpy()
