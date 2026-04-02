"""Loss function factory."""

from __future__ import annotations

import torch.nn as nn

from src.config import Config


def create_loss(config: Config) -> nn.Module:
    """Return a loss module based on ``config.loss_function``.

    Args:
        config: Experiment configuration.

    Returns:
        An ``nn.Module`` loss function.

    Raises:
        ValueError: If the loss function name is not recognised.
    """
    name = config.loss_function

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    else:
        raise ValueError(
            f"Unknown loss function '{name}'. "
            f"Valid options: 'cross_entropy', 'label_smoothing'."
        )
