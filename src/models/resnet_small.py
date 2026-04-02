"""ResNetSmall — custom ResNet-style network for 64x64 RGB classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):
    """Basic residual block: Conv→BN→ReLU→Conv→BN + skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection with 1x1 conv if dimensions change
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNetSmall(nn.Module):
    """Custom ResNet with 4 stages (~500K-2M params).

    Architecture:
        Initial conv: 3→32, 3x3, stride 1, padding 1, BN, ReLU
        layer1: 2 ResidualBlocks, 32→32, stride 1
        layer2: 2 ResidualBlocks, 32→64, stride 2
        layer3: 2 ResidualBlocks, 64→128, stride 2
        layer4: 2 ResidualBlocks, 128→256, stride 2
        Global average pooling → flatten → Linear(256, num_classes)

    Input:  (B, 3, 64, 64)
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(32, 32, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 192, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, num_classes)

        # Kaiming initialization for all conv layers
        self._initialize_weights()

    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, num_blocks: int, stride: int,
    ) -> nn.Sequential:
        """Build a stage of residual blocks."""
        layers: list[nn.Module] = []
        # First block may change dimensions
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        # Remaining blocks keep dimensions
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Apply Kaiming initialization to all conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning logits."""
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_feature_layer_names(self) -> list[str]:
        """Return list of hookable layer names for feature extraction."""
        return ["layer1", "layer2", "layer3", "layer4"]
