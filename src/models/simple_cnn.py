"""SimpleCNN — 3-block CNN baseline for 64x64 RGB classification."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class SimpleCNN(nn.Module):
    """Shallow 3-block CNN (~50K-100K params).

    Each block: Conv2d → BatchNorm2d → ReLU → MaxPool2d.
    Followed by adaptive average pooling, flatten, and two linear layers
    with ReLU + Dropout.

    Input:  (B, 3, 64, 64)
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()

        # Block 1: 3 → 32 channels, spatial 64 → 32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 32 → 64 channels, spatial 32 → 16
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 64 → 96 channels, spatial 16 → 8
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global average pooling reduces 96x8x8 → 96
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier: 96 → 64 → num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning logits."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def get_feature_layer_names(self) -> list[str]:
        """Return list of hookable layer names for feature extraction."""
        return ["block1", "block2", "block3"]
