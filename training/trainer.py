"""
Training infrastructure for model training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
import time
from .losses import CombinedLoss, calculate_iou, calculate_dice, calculate_pixel_accuracy


class Trainer:
    """Handle model training with validation and metrics tracking."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            device: Device to train on
            criterion: Loss function
            optimizer: Optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion or CombinedLoss()
        self.optimizer = optimizer

        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with loss and metrics
        """
        self.model.eval()
        val_loss = 0.0
        iou_scores = []
        dice_scores = []
        accuracy_scores = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)

                iou = calculate_iou(probs, masks)
                dice = calculate_dice(probs, masks)
                accuracy = calculate_pixel_accuracy(probs, masks)

                iou_scores.append(iou)
                dice_scores.append(dice)
                accuracy_scores.append(accuracy)

        metrics = {
            'loss': val_loss / len(val_loader),
            'iou': sum(iou_scores) / len(iou_scores),
            'dice': sum(dice_scores) / len(dice_scores),
            'accuracy': sum(accuracy_scores) / len(accuracy_scores)
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: int = 5,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, List]:
        """
        Train model with validation and early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            progress_callback: Callback for progress updates

        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            epoch_time = time.time() - start_time

            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)

            if progress_callback:
                progress_callback(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    val_metrics=val_metrics,
                    epoch_time=epoch_time
                )

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
        }

        return history

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Make prediction on single image.

        Args:
            image: Input image tensor

        Returns:
            Predicted mask (probabilities)
        """
        self.model.eval()

        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)

            image = image.to(self.device)
            output = self.model(image)
            probs = torch.sigmoid(output)

        return probs.squeeze()
