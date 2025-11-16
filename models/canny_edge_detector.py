"""
Canny Edge Detection Model Wrapper
Treats Canny edge detection as a "model" with tunable hyperparameters.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque


class CannyEdgeDetector(nn.Module):
    """
    Canny edge detection wrapped as a PyTorch model for consistency.

    This allows Canny to be used in the same pipeline as neural networks,
    with "training" being parameter tuning rather than weight updates.
    """

    def __init__(
        self,
        low_threshold: float = 0.1,
        high_threshold: float = 0.3,
        kernel_size: int = 5,
        sigma: float = 1.4,
        threshold_mode: str = "normalized",
        resize_dim: Optional[int] = None,
        use_l2_gradient: bool = True
    ):
        """
        Initialize Canny edge detector with parameters.

        Args:
            low_threshold: Low threshold for edge detection (0-1 for normalized, absolute for absolute mode)
            high_threshold: High threshold for edge detection
            kernel_size: Gaussian kernel size (must be odd)
            sigma: Gaussian blur sigma
            threshold_mode: 'normalized' (0-1) or 'absolute' (0-255)
            resize_dim: Optional resize dimension for smaller side
            use_l2_gradient: Whether to use L2 gradient magnitude
        """
        super().__init__()

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
        self.threshold_mode = threshold_mode
        self.resize_dim = resize_dim
        self.use_l2_gradient = use_l2_gradient

        # Constants for hysteresis
        self.STRONG_VAL = 255
        self.WEAK_VAL = 75

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process batch of images through Canny edge detection.

        Args:
            x: Input tensor (B, C, H, W) in range [0, 1]

        Returns:
            Edge maps tensor (B, 1, H, W) in range [0, 1]
        """
        batch_size = x.shape[0]
        device = x.device

        edge_maps = []

        for i in range(batch_size):
            # Convert to numpy
            img = x[i].cpu().numpy()
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            img = (img * 255).astype(np.uint8)

            # Process single image
            edges = self._process_single_image(img)

            # Convert back to tensor
            edges_tensor = torch.from_numpy(edges).float() / 255.0
            edge_maps.append(edges_tensor.unsqueeze(0))

        result = torch.stack(edge_maps, dim=0).to(device)
        return result

    def _process_single_image(self, img: np.ndarray) -> np.ndarray:
        """Process single image with Canny edge detection."""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Optional resize
        if self.resize_dim:
            h, w = gray.shape
            scale = self.resize_dim / min(h, w)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(
            gray,
            (self.kernel_size, self.kernel_size),
            sigmaX=self.sigma,
            sigmaY=self.sigma
        )

        # Calculate thresholds based on mode
        if self.threshold_mode == "normalized":
            low_cv = int(round(self.low_threshold * 255))
            high_cv = int(round(self.high_threshold * 255))
        else:  # absolute
            low_cv = int(round(self.low_threshold))
            high_cv = int(round(self.high_threshold))

        # Apply Canny edge detection
        edges = cv2.Canny(
            blurred,
            low_cv,
            high_cv,
            L2gradient=self.use_l2_gradient
        )

        return edges

    def process_with_intermediates(
        self,
        img: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Process image and return all intermediate steps for visualization.

        Args:
            img: Input image (H, W, C) uint8

        Returns:
            Dictionary with all intermediate results
        """
        results = {}

        # Original
        results['original'] = img.copy()

        # Grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        results['grayscale'] = gray

        # Optional resize
        if self.resize_dim:
            h, w = gray.shape
            scale = self.resize_dim / min(h, w)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            results['resized'] = gray

        # Gaussian blur
        blurred = cv2.GaussianBlur(
            gray,
            (self.kernel_size, self.kernel_size),
            sigmaX=self.sigma,
            sigmaY=self.sigma
        )
        results['blurred'] = blurred

        # Gradients
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        results['gradient_x'] = gx
        results['gradient_y'] = gy

        # Magnitude and direction
        magnitude = np.hypot(gx, gy)
        magnitude_norm = magnitude / (magnitude.max() + 1e-12)
        direction = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
        results['magnitude'] = magnitude_norm
        results['direction'] = direction

        # Non-maximum suppression
        nms = self._non_maximum_suppression(magnitude_norm, direction)
        results['nms'] = nms

        # Double threshold
        if self.threshold_mode == "normalized":
            low_thresh = self.low_threshold
            high_thresh = self.high_threshold
        else:
            low_thresh = self.low_threshold / 255.0
            high_thresh = self.high_threshold / 255.0

        double_thresh = self._double_threshold(nms, low_thresh, high_thresh)
        results['double_threshold'] = double_thresh

        # Final edges with hysteresis
        edges = self._hysteresis(double_thresh)
        results['edges'] = edges

        # Also get OpenCV Canny for comparison
        if self.threshold_mode == "normalized":
            low_cv = int(round(self.low_threshold * 255))
            high_cv = int(round(self.high_threshold * 255))
        else:
            low_cv = int(round(self.low_threshold))
            high_cv = int(round(self.high_threshold))

        edges_cv = cv2.Canny(blurred, low_cv, high_cv, L2gradient=self.use_l2_gradient)
        results['edges_opencv'] = edges_cv

        return results

    def _non_maximum_suppression(
        self,
        magnitude: np.ndarray,
        angle_deg: np.ndarray
    ) -> np.ndarray:
        """Non-maximum suppression for Canny edge detection."""
        H, W = magnitude.shape
        suppressed = np.zeros((H, W), dtype=np.float32)
        angle = angle_deg.copy()
        angle[angle < 0] += 180

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                q, r = 0.0, 0.0
                a = angle[i, j]

                if (0 <= a < 22.5) or (157.5 <= a < 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif 22.5 <= a < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                elif 67.5 <= a < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                elif 112.5 <= a < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    suppressed[i, j] = magnitude[i, j]

        return suppressed

    def _double_threshold(
        self,
        img: np.ndarray,
        low: float,
        high: float
    ) -> np.ndarray:
        """Apply double threshold to edge magnitude."""
        strong = (img >= high)
        weak = (img >= low) & (img < high)
        result = np.zeros_like(img, dtype=np.uint8)
        result[strong] = self.STRONG_VAL
        result[weak] = self.WEAK_VAL
        return result

    def _hysteresis(self, edge_map: np.ndarray) -> np.ndarray:
        """Edge tracking by hysteresis."""
        H, W = edge_map.shape
        result = edge_map.copy()
        queue: deque = deque(zip(*np.where(result == self.STRONG_VAL)))

        while queue:
            i, j = queue.popleft()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    y, x = i + di, j + dj
                    if 0 <= y < H and 0 <= x < W and result[y, x] == self.WEAK_VAL:
                        result[y, x] = self.STRONG_VAL
                        queue.append((y, x))

        result[result != self.STRONG_VAL] = 0
        return result

    def get_parameters_dict(self) -> Dict:
        """Get current parameters as dictionary."""
        return {
            'low_threshold': self.low_threshold,
            'high_threshold': self.high_threshold,
            'kernel_size': self.kernel_size,
            'sigma': self.sigma,
            'threshold_mode': self.threshold_mode,
            'resize_dim': self.resize_dim,
            'use_l2_gradient': self.use_l2_gradient
        }

    def update_parameters(self, params: Dict):
        """Update parameters from dictionary."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Ensure kernel size is odd
        if hasattr(self, 'kernel_size'):
            self.kernel_size = self.kernel_size if self.kernel_size % 2 == 1 else self.kernel_size + 1


def create_canny_model(
    low_threshold: float = 0.1,
    high_threshold: float = 0.3,
    kernel_size: int = 5,
    sigma: float = 1.4,
    **kwargs
) -> CannyEdgeDetector:
    """
    Factory function to create Canny edge detector.

    Args:
        low_threshold: Low threshold for edge detection
        high_threshold: High threshold for edge detection
        kernel_size: Gaussian kernel size
        sigma: Gaussian blur sigma
        **kwargs: Additional parameters

    Returns:
        CannyEdgeDetector instance
    """
    return CannyEdgeDetector(
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        kernel_size=kernel_size,
        sigma=sigma,
        **kwargs
    )
