#!/usr/bin/env python3
"""
Create sample dataset with synthetic welding patterns for testing.
This script generates simple input images and corresponding binary masks.
"""

import cv2
import numpy as np
from pathlib import Path


def create_sample_dataset(output_dir: str = "sample_dataset", num_samples: int = 10):
    """
    Create sample dataset with synthetic welding patterns.

    Args:
        output_dir: Output directory for dataset
        num_samples: Number of samples to generate
    """
    output_path = Path(output_dir)
    input_dir = output_path / "input"
    target_dir = output_path / "target"

    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {num_samples} sample images in {output_dir}/")

    for idx in range(num_samples):
        input_img = np.ones((256, 384, 3), dtype=np.uint8) * 200
        mask = np.zeros((256, 384), dtype=np.uint8)

        pattern_type = idx % 5

        if pattern_type == 0:
            for i in range(3):
                y = 50 + i * 70
                cv2.line(input_img, (50, y), (334, y), (50, 50, 50), 3)
                cv2.line(mask, (50, y), (334, y), 255, 8)

        elif pattern_type == 1:
            for i in range(4):
                x = 60 + i * 90
                cv2.line(input_img, (x, 30), (x, 226), (50, 50, 50), 3)
                cv2.line(mask, (x, 30), (x, 226), 255, 8)

        elif pattern_type == 2:
            cv2.rectangle(input_img, (80, 60), (304, 196), (50, 50, 50), 3)
            cv2.rectangle(mask, (80, 60), (304, 196), 255, 8)

        elif pattern_type == 3:
            center = (192, 128)
            for radius in [40, 70, 100]:
                cv2.circle(input_img, center, radius, (50, 50, 50), 3)
                cv2.circle(mask, center, radius, 255, 8)

        elif pattern_type == 4:
            points = np.array([
                [50, 128],
                [120, 60],
                [192, 128],
                [264, 60],
                [334, 128]
            ], dtype=np.int32)
            cv2.polylines(input_img, [points], False, (50, 50, 50), 3)
            cv2.polylines(mask, [points], False, 255, 8)

        noise = np.random.randint(-20, 20, input_img.shape, dtype=np.int16)
        input_img = np.clip(input_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        input_filename = input_dir / f"sample_{idx:03d}.png"
        target_filename = target_dir / f"sample_{idx:03d}.png"

        cv2.imwrite(str(input_filename), input_img)
        cv2.imwrite(str(target_filename), mask)

        print(f"  Created sample {idx + 1}/{num_samples}: {input_filename.name}")

    print(f"\n✓ Sample dataset created successfully!")
    print(f"  Input images: {input_dir}")
    print(f"  Target masks: {target_dir}")
    print(f"\nTo use this dataset:")
    print(f"  1. Go to the 'Dataset Builder' tab in the application")
    print(f"  2. Upload all files from the 'input' folder as input images")
    print(f"  3. Upload all files from the 'target' folder as target masks")
    print(f"  4. Click 'Create Dataset'")


if __name__ == "__main__":
    create_sample_dataset()
