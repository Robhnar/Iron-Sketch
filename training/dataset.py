"""
PyTorch Dataset for loading welding path images and masks.
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from typing import List, Dict, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import requests


class WeldingDataset(Dataset):
    """Dataset for welding path segmentation."""

    def __init__(
        self,
        image_records: List[Dict],
        transform: Optional[Callable] = None,
        augmentation: Optional[A.Compose] = None
    ):
        """
        Initialize dataset.

        Args:
            image_records: List of dicts with 'input_image_url' and 'target_mask_url'
            transform: Transform to apply (normalization, etc)
            augmentation: Augmentation pipeline
        """
        self.image_records = image_records
        self.augmentation = augmentation
        self.transform = transform or self._default_transform()

    def __len__(self) -> int:
        return len(self.image_records)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get image and mask pair.

        Args:
            idx: Index

        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        record = self.image_records[idx]

        image = self._load_image_from_url(record['input_image_url'])
        mask = self._load_image_from_url(record['target_mask_url'])

        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = (mask > 127).astype(np.float32)

        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        mask = mask.unsqueeze(0)

        return image, mask

    @staticmethod
    def _load_image_from_url(url: str) -> np.ndarray:
        """Load image from URL."""
        response = requests.get(url)
        response.raise_for_status()

        nparr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to load image from {url}")

        return image

    @staticmethod
    def _default_transform() -> A.Compose:
        """Default transform for normalization."""
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    @staticmethod
    def create_augmentation_pipeline(
        rotation_limit: int = 15,
        brightness_limit: float = 0.2,
        enable_flip: bool = True
    ) -> A.Compose:
        """
        Create augmentation pipeline.

        Args:
            rotation_limit: Maximum rotation degrees
            brightness_limit: Brightness adjustment range
            enable_flip: Whether to enable flipping

        Returns:
            Albumentations Compose object
        """
        transforms = [
            A.Rotate(limit=rotation_limit, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=brightness_limit,
                p=0.5
            )
        ]

        if enable_flip:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3)
            ])

        return A.Compose(transforms)
