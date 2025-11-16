"""
Image processing utilities for AI welding path generation.
Handles resizing, normalization, and augmentation.
"""

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from typing import Tuple, Optional


class ImageProcessor:
    """Handles image preprocessing for model input."""

    TARGET_HEIGHT = 256
    TARGET_WIDTH = 384
    TARGET_RATIO = 3 / 2

    @staticmethod
    def resize_with_aspect_ratio(
        image: np.ndarray,
        target_height: int = TARGET_HEIGHT,
        target_width: int = TARGET_WIDTH
    ) -> np.ndarray:
        """
        Resize image to target dimensions while maintaining aspect ratio.
        Uses padding or cropping as needed.

        Args:
            image: Input image (H, W, C) or (H, W)
            target_height: Target height (default 256)
            target_width: Target width (default 384)

        Returns:
            Resized image with target dimensions
        """
        h, w = image.shape[:2]
        target_ratio = target_width / target_height
        current_ratio = w / h

        if current_ratio > target_ratio:
            new_w = target_width
            new_h = int(target_width / current_ratio)
        else:
            new_h = target_height
            new_w = int(target_height * current_ratio)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if len(image.shape) == 3:
            output = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
        else:
            output = np.zeros((target_height, target_width), dtype=image.dtype)

        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2

        output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return output

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range for model input.

        Args:
            image: Input image (H, W, C) in uint8 format

        Returns:
            Normalized image as float32
        """
        return image.astype(np.float32) / 255.0

    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """
        Convert normalized image back to uint8.

        Args:
            image: Normalized image (H, W, C) in [0, 1] range

        Returns:
            Image in uint8 format
        """
        return (image * 255).astype(np.uint8)

    @staticmethod
    def binary_threshold(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert grayscale or probability mask to binary.

        Args:
            mask: Input mask
            threshold: Threshold value (default 0.5)

        Returns:
            Binary mask (0 or 255)
        """
        if mask.max() <= 1.0:
            mask = mask * 255

        binary = np.where(mask > threshold * 255, 255, 0).astype(np.uint8)
        return binary

    @staticmethod
    def post_process_mask(
        mask: np.ndarray,
        close_kernel_size: int = 3,
        min_area: int = 100
    ) -> np.ndarray:
        """
        Post-process binary mask with morphological operations.

        Args:
            mask: Binary mask
            close_kernel_size: Kernel size for morphological closing
            min_area: Minimum contour area to keep

        Returns:
            Cleaned binary mask
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel_size, close_kernel_size)
        )

        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        output = np.zeros_like(mask)

        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(output, [contour], -1, 255, -1)

        return output

    @staticmethod
    def skeletonize(mask: np.ndarray) -> np.ndarray:
        """
        Convert mask to skeleton (single-pixel width paths).

        Args:
            mask: Binary mask

        Returns:
            Skeletonized mask
        """
        from skimage.morphology import skeletonize as sk_skeletonize

        if mask.max() > 1:
            mask = mask / 255

        skeleton = sk_skeletonize(mask.astype(bool))
        return (skeleton * 255).astype(np.uint8)

    @staticmethod
    def create_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create overlay of mask on image.

        Args:
            image: Original RGB image
            mask: Binary mask
            color: Overlay color (BGR)
            alpha: Transparency (0-1)

        Returns:
            Image with overlay
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        overlay = image.copy()

        mask_bool = mask > 127
        overlay[mask_bool] = color

        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        return result

    @staticmethod
    def get_augmentation_pipeline(
        rotation_limit: int = 15,
        brightness_limit: float = 0.2,
        enable_flip: bool = True
    ) -> A.Compose:
        """
        Create augmentation pipeline for training.

        Args:
            rotation_limit: Maximum rotation angle
            brightness_limit: Brightness adjustment range
            enable_flip: Whether to enable flipping

        Returns:
            Albumentations composition
        """
        transforms = [
            A.Rotate(limit=rotation_limit, p=0.5),
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

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """
        Load image from bytes.

        Args:
            image_bytes: Image data as bytes

        Returns:
            NumPy array (H, W, C) in BGR format
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        """Convert RGB to BGR."""
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
