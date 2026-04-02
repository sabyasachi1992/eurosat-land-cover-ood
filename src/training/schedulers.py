"""Learning rate scheduler factory."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, _LRScheduler

from src.config import Config


def create_scheduler(optimizer: Optimizer, config: Config) -> _LRScheduler:
    """Return an LR scheduler based on ``config.scheduler``.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        config: Experiment configuration.

    Returns:
        An ``_LRScheduler`` instance.

    Raises:
        ValueError: If the scheduler name is not recognised.
    """
    name = config.scheduler

    if name == "step_lr":
        return StepLR(
            optimizer,
            step_size=config.scheduler_params["step_size"],
            gamma=config.scheduler_params["gamma"],
        )
    elif name == "cosine_annealing":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler_params.get("T_max", config.epochs),
        )
    else:
        raise ValueError(
            f"Unknown scheduler '{name}'. "
            f"Valid options: 'step_lr', 'cosine_annealing'."
        )
