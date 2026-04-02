"""Global seed setting for reproducibility."""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set seeds for Python random, NumPy, and PyTorch (CPU + CUDA).

    Also sets torch.backends.cudnn.deterministic = True for full
    reproducibility on CUDA devices.

    Args:
        seed: Non-negative integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
