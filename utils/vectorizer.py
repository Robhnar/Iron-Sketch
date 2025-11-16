"""
Vectorization utilities for converting binary masks to path coordinates.
"""

import cv2
import numpy as np
from typing import List, Tuple


class Vectorizer:
    """Convert binary masks to vector paths."""

    @staticmethod
    def vectorize_mask(
        mask: np.ndarray,
        simplify_epsilon: float = 1.0,
        min_contour_points: int = 2
    ) -> List[List[Tuple[int, int]]]:
        """
        Convert binary mask to vector paths using contour detection.

        Args:
            mask: Binary mask (255 = path, 0 = background)
            simplify_epsilon: Douglas-Peucker simplification parameter
            min_contour_points: Minimum points required to keep a contour

        Returns:
            List of paths, each path is a list of (x, y) coordinates
        """
        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        paths = []

        for contour in contours:
            if len(contour) < min_contour_points:
                continue

            epsilon = simplify_epsilon
            simplified = cv2.approxPolyDP(contour, epsilon, closed=False)

            points = simplified.reshape(-1, 2).tolist()

            if len(points) >= min_contour_points:
                paths.append([(int(x), int(y)) for x, y in points])

        return paths

    @staticmethod
    def optimize_path_order(paths: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        """
        Optimize path order using nearest-neighbor heuristic to minimize travel distance.

        Args:
            paths: List of paths

        Returns:
            Reordered paths
        """
        if not paths:
            return paths

        optimized = [paths[0]]
        remaining = paths[1:]

        while remaining:
            last_point = optimized[-1][-1]

            min_dist = float('inf')
            nearest_idx = 0

            for i, path in enumerate(remaining):
                start_point = path[0]
                dist = Vectorizer._distance(last_point, start_point)

                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i

            optimized.append(remaining.pop(nearest_idx))

        return optimized

    @staticmethod
    def _distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    @staticmethod
    def draw_paths_on_image(
        image: np.ndarray,
        paths: List[List[Tuple[int, int]]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw vector paths on image.

        Args:
            image: Input image
            paths: List of paths
            color: Line color (BGR)
            thickness: Line thickness

        Returns:
            Image with drawn paths
        """
        output = image.copy()

        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        for path in paths:
            for i in range(len(path) - 1):
                pt1 = tuple(path[i])
                pt2 = tuple(path[i + 1])
                cv2.line(output, pt1, pt2, color, thickness)

            for point in path:
                cv2.circle(output, tuple(point), 3, (255, 0, 0), -1)

        return output

    @staticmethod
    def get_path_statistics(paths: List[List[Tuple[int, int]]]) -> dict:
        """
        Calculate statistics about paths.

        Args:
            paths: List of paths

        Returns:
            Dictionary with statistics
        """
        total_points = sum(len(path) for path in paths)
        total_distance = 0.0

        for path in paths:
            for i in range(len(path) - 1):
                total_distance += Vectorizer._distance(path[i], path[i + 1])

        return {
            'num_paths': len(paths),
            'total_points': total_points,
            'average_points_per_path': total_points / len(paths) if paths else 0,
            'total_distance_pixels': total_distance
        }

    @staticmethod
    def transform_coordinates(
        paths: List[List[Tuple[int, int]]],
        scale_mm_per_px: float,
        origin_x_mm: float,
        origin_y_mm: float,
        invert_y: bool = True,
        image_height: int = 256
    ) -> List[List[Tuple[float, float]]]:
        """
        Transform pixel coordinates to robot workspace coordinates.

        Args:
            paths: List of paths in pixel coordinates
            scale_mm_per_px: Millimeters per pixel
            origin_x_mm: X origin offset in mm
            origin_y_mm: Y origin offset in mm
            invert_y: Whether to invert Y axis (image down = robot up)
            image_height: Image height for Y inversion

        Returns:
            Paths in robot coordinates (mm)
        """
        transformed_paths = []

        for path in paths:
            transformed_path = []

            for x_px, y_px in path:
                if invert_y:
                    y_px = image_height - y_px

                x_mm = x_px * scale_mm_per_px + origin_x_mm
                y_mm = y_px * scale_mm_per_px + origin_y_mm

                transformed_path.append((x_mm, y_mm))

            transformed_paths.append(transformed_path)

        return transformed_paths
