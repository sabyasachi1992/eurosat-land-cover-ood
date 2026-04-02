"""Training loop with early stopping and checkpointing."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.training.losses import create_loss
from src.training.schedulers import create_scheduler
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
    """

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self._best_score: float = float("inf")
        self._counter: int = 0
        self.best_weights: OrderedDict | None = None

    def step(self, val_loss: float) -> bool:
        """Record a validation loss and decide whether to stop.

        Args:
            val_loss: Current epoch's validation loss.

        Returns:
            ``True`` if training should stop (no improvement for
            ``patience`` consecutive epochs).
        """
        if val_loss < self._best_score:
            self._best_score = val_loss
            self._counter = 0
            return False
        else:
            self._counter += 1
            return self._counter >= self.patience

    @property
    def best_score(self) -> float:
        """Return the minimum validation loss observed so far."""
        return self._best_score


class Trainer:
    """Full training loop with validation, early stopping, and checkpointing.

    Args:
        model: The neural network to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Experiment configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
    ) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = create_loss(config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = create_scheduler(self.optimizer, config)
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)

        logger.info("Trainer initialised on %s", self.device)

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            History dict with keys: train_loss, val_loss, train_acc,
            val_acc, lr (lists per epoch), best_epoch, stopped_epoch.
        """
        history: dict = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
            "best_epoch": 0,
            "stopped_epoch": 0,
        }

        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate_epoch()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f train_acc=%.4f "
                "val_loss=%.4f val_acc=%.4f lr=%.6f",
                epoch, self.config.epochs,
                train_loss, train_acc, val_loss, val_acc, current_lr,
            )

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self.early_stopping.best_weights = copy.deepcopy(
                    self.model.state_dict()
                )
                self._save_checkpoint(best_val_loss, best_epoch)

            # Early stopping check
            should_stop = self.early_stopping.step(val_loss)
            if should_stop:
                logger.info(
                    "Early stopping triggered at epoch %d (best epoch: %d)",
                    epoch, best_epoch,
                )
                history["stopped_epoch"] = epoch
                history["best_epoch"] = best_epoch
                break
        else:
            # Completed all epochs without early stopping
            history["stopped_epoch"] = self.config.epochs
            history["best_epoch"] = best_epoch

        # Restore best weights
        if self.early_stopping.best_weights is not None:
            self.model.load_state_dict(self.early_stopping.best_weights)

        return history

    def _train_epoch(self) -> tuple[float, float]:
        """Run one training epoch.

        Returns:
            (average_loss, accuracy) for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(
            self.train_loader, desc="Training", leave=False,
        ):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = running_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def _validate_epoch(self) -> tuple[float, float]:
        """Run one validation epoch.

        Returns:
            (average_loss, accuracy) for the epoch.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(
                self.val_loader, desc="Validation", leave=False,
            ):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = running_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def _save_checkpoint(self, best_val_loss: float, best_epoch: int) -> None:
        """Save model checkpoint to ``config.weights_path``."""
        path = Path(self.config.weights_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "architecture": self.config.architecture,
            "state_dict": copy.deepcopy(self.model.state_dict()),
            "config": asdict(self.config),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }
        torch.save(checkpoint, path)
        logger.info("Saved checkpoint to %s (epoch %d, val_loss=%.4f)",
                     path, best_epoch, best_val_loss)
