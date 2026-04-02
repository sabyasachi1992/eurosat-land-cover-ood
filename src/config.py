"""Configuration dataclass with YAML loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


_VALID_ARCHITECTURES = {"simple_cnn", "resnet_small"}
_VALID_LOSS_FUNCTIONS = {"cross_entropy", "label_smoothing"}
_VALID_SCHEDULERS = {"step_lr", "cosine_annealing"}


@dataclass
class Config:
    """Single source of truth for all experiment hyperparameters."""

    # Paths
    dataset_root: str = ""
    output_dir: str = ""
    weights_path: str = ""
    norm_stats_path: str = ""

    # Seed
    seed: int = 42

    # Classes
    known_classes: list[str] = field(default_factory=list)
    ghost_classes: list[str] = field(default_factory=list)

    # Data splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    unlabeled_known_count: int = 2000

    # Augmentation
    augmentation: dict = field(default_factory=dict)

    # Model
    architecture: str = "resnet_small"

    # Training
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 100
    loss_function: str = "cross_entropy"
    label_smoothing: float = 0.1
    scheduler: str = "cosine_annealing"
    scheduler_params: dict = field(default_factory=dict)
    early_stopping_patience: int = 10

    # OOD
    ood_feature_layer: str = "layer3"
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    hdbscan_min_cluster_size: int = 50
    hdbscan_min_samples: int = 10

    @staticmethod
    def load(path: str) -> Config:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Populated and validated Config instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If validation fails.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("Config file must contain a YAML mapping")

        config = Config(**data)
        config.validate()
        return config

    def validate(self) -> None:
        """Validate all fields and raise descriptive errors."""
        errors: list[str] = []

        # Path fields must be non-empty strings
        for field_name in ("dataset_root", "output_dir", "weights_path", "norm_stats_path"):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"'{field_name}' must be a non-empty string, got {val!r}")

        # Seed must be a non-negative integer
        if not isinstance(self.seed, int) or self.seed < 0:
            errors.append(f"'seed' must be a non-negative integer, got {self.seed!r}")

        # Split ratios must sum to 1.0
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            errors.append(
                f"Split ratios must sum to 1.0, got "
                f"{self.train_ratio} + {self.val_ratio} + {self.test_ratio} = {ratio_sum}"
            )

        # Architecture must be valid
        if self.architecture not in _VALID_ARCHITECTURES:
            errors.append(
                f"'architecture' must be one of {_VALID_ARCHITECTURES}, got {self.architecture!r}"
            )

        # Loss function must be valid
        if self.loss_function not in _VALID_LOSS_FUNCTIONS:
            errors.append(
                f"'loss_function' must be one of {_VALID_LOSS_FUNCTIONS}, got {self.loss_function!r}"
            )

        # Scheduler must be valid
        if self.scheduler not in _VALID_SCHEDULERS:
            errors.append(
                f"'scheduler' must be one of {_VALID_SCHEDULERS}, got {self.scheduler!r}"
            )

        # Positive integers
        for field_name in ("epochs", "batch_size", "early_stopping_patience"):
            val = getattr(self, field_name)
            if not isinstance(val, int) or val <= 0:
                errors.append(f"'{field_name}' must be a positive integer, got {val!r}")

        # unlabeled_known_count must be a positive integer
        if not isinstance(self.unlabeled_known_count, int) or self.unlabeled_known_count <= 0:
            errors.append(
                f"'unlabeled_known_count' must be a positive integer, got {self.unlabeled_known_count!r}"
            )

        # OOD positive integers
        for field_name in ("umap_n_neighbors", "umap_n_components",
                           "hdbscan_min_cluster_size", "hdbscan_min_samples"):
            val = getattr(self, field_name)
            if not isinstance(val, int) or val <= 0:
                errors.append(f"'{field_name}' must be a positive integer, got {val!r}")

        # Non-empty class lists
        if not self.known_classes:
            errors.append("'known_classes' must be a non-empty list")
        if not self.ghost_classes:
            errors.append("'ghost_classes' must be a non-empty list")

        # ood_feature_layer must be non-empty
        if not isinstance(self.ood_feature_layer, str) or not self.ood_feature_layer.strip():
            errors.append("'ood_feature_layer' must be a non-empty string")

        if errors:
            raise ValueError("Config validation failed:\n  - " + "\n  - ".join(errors))
