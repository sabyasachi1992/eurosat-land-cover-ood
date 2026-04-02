"""Model factory — create model by config architecture name."""

from __future__ import annotations

import torch.nn as nn

from src.config import Config
from src.models.simple_cnn import SimpleCNN
from src.models.resnet_small import ResNetSmall
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_model(config: Config) -> nn.Module:
    """Instantiate a model based on ``config.architecture``.

    Args:
        config: Experiment configuration.

    Returns:
        An ``nn.Module`` ready for training.

    Raises:
        ValueError: If the architecture name is not recognised.
    """
    name = config.architecture

    if name == "simple_cnn":
        model = SimpleCNN(num_classes=len(config.known_classes))
    elif name == "resnet_small":
        model = ResNetSmall(num_classes=len(config.known_classes))
    else:
        raise ValueError(
            f"Unknown architecture '{name}'. "
            f"Valid options: 'simple_cnn', 'resnet_small'."
        )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Created %s with %d parameters", name, total_params)

    return model
