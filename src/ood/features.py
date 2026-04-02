"""Hook-based feature extraction from intermediate model layers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FeatureExtractor:
    """Extract intermediate-layer activations using PyTorch forward hooks.

    Args:
        model: The neural network model.
        layer_name: Name of the layer to hook (e.g. "layer3", "block2").

    Raises:
        ValueError: If layer_name is not found in the model.
    """

    def __init__(self, model: nn.Module, layer_name: str) -> None:
        self.model = model
        self.layer_name = layer_name
        self._features: list[torch.Tensor] = []
        self._hook_handle = None

        # Validate that the layer exists
        found = False
        for name, _ in self.model.named_modules():
            if name == layer_name:
                found = True
                break
        if not found:
            available = [n for n, _ in self.model.named_modules() if n]
            raise ValueError(
                f"Layer '{layer_name}' not found in model. "
                f"Available layers: {available}"
            )

    def _register_hook(self) -> None:
        """Register a forward hook on the target layer."""
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                self._hook_handle = module.register_forward_hook(self._hook_fn)
                return

    def _hook_fn(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        """Hook callback that stores the layer output."""
        self._features.append(output.detach().cpu())

    def _remove_hook(self) -> None:
        """Remove the registered forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def extract(self, dataloader: DataLoader, device: str) -> np.ndarray:
        """Extract features from all samples in the dataloader.

        Args:
            dataloader: DataLoader yielding (images, labels) or (images,) batches.
            device: Device string (e.g. "cpu", "cuda").

        Returns:
            Feature array of shape (N, D) where D is the feature dimension.
        """
        self.model.eval()
        self.model.to(device)
        self._features = []
        self._register_hook()

        try:
            for batch in dataloader:
                # Support both (images, labels) and (images,) batches
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(device)
                self.model(images)
        finally:
            self._remove_hook()

        # Concatenate all captured features
        all_features = torch.cat(self._features, dim=0)

        # Global average pooling if features are 4D (B, C, H, W) -> (B, C)
        if all_features.ndim == 4:
            all_features = all_features.mean(dim=[2, 3])

        return all_features.numpy()
