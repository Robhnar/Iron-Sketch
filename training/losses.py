"""
Loss functions for binary segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        """
        Initialize Dice loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.

        Args:
            predictions: Model predictions (logits)
            targets: Ground truth masks

        Returns:
            Dice loss value
        """
        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Dice + BCE loss for binary segmentation."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        """
        Initialize combined loss.

        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.

        Args:
            predictions: Model predictions (logits)
            targets: Ground truth masks

        Returns:
            Combined loss value
        """
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)

        return self.dice_weight * dice + self.bce_weight * bce


def calculate_iou(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union (IoU) metric.

    Args:
        predictions: Model predictions (probabilities)
        targets: Ground truth masks
        threshold: Threshold for binarization

    Returns:
        IoU score
    """
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()

    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou.item()


def calculate_dice(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Dice coefficient.

    Args:
        predictions: Model predictions (probabilities)
        targets: Ground truth masks
        threshold: Threshold for binarization

    Returns:
        Dice score
    """
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()

    intersection = (predictions * targets).sum()
    dice = (2.0 * intersection) / (predictions.sum() + targets.sum() + 1e-8)

    return dice.item()


def calculate_pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate pixel-wise accuracy.

    Args:
        predictions: Model predictions (probabilities)
        targets: Ground truth masks
        threshold: Threshold for binarization

    Returns:
        Pixel accuracy
    """
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()

    correct = (predictions == targets).sum()
    total = targets.numel()

    accuracy = correct / total
    return accuracy.item()
