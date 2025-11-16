#!/usr/bin/env python3
"""
Generate robot welding script from Canny edge detection.

Combines:
  1. canny_edge_detection_pipeline.py - for edge detection
  2. vectorizer.py - for path extraction
  3. robot_script.py - for script generation

Usage:
    python generate_script_from_canny.py --image input.jpg --output welding_paths.script
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Dict

from canny_edge_detection_pipeline import canny_pipeline
from vectorizer import Vectorizer
from robot_script import RobotScriptGenerator


def generate_welding_script_from_image(
    image_path: Path,
    output_script: Path,
    canny_params: Dict = None,
    robot_params: Dict = None,
    save_intermediates: bool = True
) -> Dict:
    """
    Complete pipeline: image -> Canny edges -> vectorized paths -> robot script.

    Args:
        image_path: Input image path
        output_script: Output script path
        canny_params: Canny edge detection parameters
        robot_params: Robot configuration parameters
        save_intermediates: Whether to save intermediate images

    Returns:
        Dictionary with processing results and statistics
    """
    # Default parameters
    if canny_params is None:
        canny_params = {
            'low_threshold': 0.1,
            'high_threshold': 0.3,
            'kernel_size': 5,
            'sigma': 1.4,
            'resize_dim': 512,
            'threshold_mode': 'normalized'
        }

    if robot_params is None:
        robot_params = {
            'x0': -825.0,
            'y0': -115.0,
            'z0': -363.7,
            'dz': 10.0,
            'move_speed': 200,
            'draw_speed': 50,
            'acceleration': 1000,
            'rx': 180.0,
            'ry': 0.0,
            'rz': 90.0
        }

    print(f"Processing image: {image_path}")

    # Step 1: Apply Canny edge detection
    print("Step 1: Applying Canny edge detection...")
    output_dir = image_path.parent / f"{image_path.stem}_canny_output"
    output_dir.mkdir(exist_ok=True)

    canny_pipeline(
        image_path=image_path,
        out_root=output_dir.parent,
        low_thresh=canny_params['low_threshold'],
        high_thresh=canny_params['high_threshold'],
        ksize=canny_params['kernel_size'],
        sigma=canny_params['sigma'],
        resize_dim=canny_params.get('resize_dim'),
        threshold_mode=canny_params.get('threshold_mode', 'normalized')
    )

    # Load edge detection result
    edges_path = output_dir / '09_edges_custom.png'
    if not edges_path.exists():
        edges_path = output_dir / '10_edges_opencv.png'

    edges = cv2.imread(str(edges_path), cv2.IMREAD_GRAYSCALE)
    if edges is None:
        raise FileNotFoundError(f"Could not load edges from {edges_path}")

    print(f"Edges loaded from: {edges_path}")

    # Step 2: Vectorize edges to paths
    print("Step 2: Vectorizing edges to paths...")
    vectorizer = Vectorizer()
    paths = vectorizer.vectorize_mask(edges, simplify_epsilon=1.0, min_contour_points=3)

    print(f"Found {len(paths)} paths")

    # Optimize path order
    paths = vectorizer.optimize_path_order(paths)

    # Get statistics
    stats = vectorizer.get_path_statistics(paths)
    print(f"Total points: {stats['total_points']}")
    print(f"Average points per path: {stats['average_points_per_path']:.1f}")

    # Step 3: Transform to robot coordinates
    print("Step 3: Transforming to robot coordinates...")
    image_height = edges.shape[0]
    mm_per_pixel = 0.5  # Default scale

    robot_paths = vectorizer.transform_coordinates(
        paths,
        scale_mm_per_px=mm_per_pixel,
        origin_x_mm=robot_params['x0'],
        origin_y_mm=robot_params['y0'],
        invert_y=True,
        image_height=image_height
    )

    # Step 4: Generate robot script
    print("Step 4: Generating robot script...")
    script_gen = RobotScriptGenerator()

    script = script_gen.generate_welding_script(
        paths=robot_paths,
        z_height=robot_params['z0'],
        z_offset=robot_params['dz'],
        move_speed=robot_params['move_speed'],
        draw_speed=robot_params['draw_speed'],
        acceleration=robot_params['acceleration'],
        orientation={
            'rx': robot_params['rx'],
            'ry': robot_params['ry'],
            'rz': robot_params['rz']
        }
    )

    # Save script
    output_script.parent.mkdir(parents=True, exist_ok=True)
    output_script.write_text(script, encoding='utf-8')
    print(f"Script saved to: {output_script}")

    # Save visualization
    if save_intermediates:
        overlay = vectorizer.draw_paths_on_image(edges, paths, color=(0, 255, 0), thickness=2)
        overlay_path = output_dir / 'vector_overlay.png'
        cv2.imwrite(str(overlay_path), overlay)
        print(f"Visualization saved to: {overlay_path}")

        # Save paths as JSON
        paths_json = output_dir / 'paths.json'
        paths_json.write_text(json.dumps(paths, indent=2))

    # Return results
    return {
        'success': True,
        'image_path': str(image_path),
        'output_script': str(output_script),
        'output_dir': str(output_dir),
        'num_paths': len(paths),
        'total_points': stats['total_points'],
        'image_size': edges.shape,
        'canny_params': canny_params,
        'robot_params': robot_params
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate robot welding script from image using Canny edge detection'
    )
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output script path')
    parser.add_argument('--low', type=float, default=0.1, help='Canny low threshold')
    parser.add_argument('--high', type=float, default=0.3, help='Canny high threshold')
    parser.add_argument('--ksize', type=int, default=5, help='Gaussian kernel size')
    parser.add_argument('--sigma', type=float, default=1.4, help='Gaussian sigma')
    parser.add_argument('--resize', type=int, default=512, help='Resize dimension')
    parser.add_argument('--x0', type=float, default=-825.0, help='Robot X origin')
    parser.add_argument('--y0', type=float, default=-115.0, help='Robot Y origin')
    parser.add_argument('--z0', type=float, default=-363.7, help='Robot Z height')
    parser.add_argument('--dz', type=float, default=10.0, help='Z lift height')
    parser.add_argument('--move-speed', type=int, default=200, help='Move speed (mm/s)')
    parser.add_argument('--draw-speed', type=int, default=50, help='Draw speed (mm/s)')

    args = parser.parse_args()

    canny_params = {
        'low_threshold': args.low,
        'high_threshold': args.high,
        'kernel_size': args.ksize,
        'sigma': args.sigma,
        'resize_dim': args.resize,
        'threshold_mode': 'normalized'
    }

    robot_params = {
        'x0': args.x0,
        'y0': args.y0,
        'z0': args.z0,
        'dz': args.dz,
        'move_speed': args.move_speed,
        'draw_speed': args.draw_speed,
        'acceleration': 1000,
        'rx': 180.0,
        'ry': 0.0,
        'rz': 90.0
    }

    try:
        result = generate_welding_script_from_image(
            Path(args.image),
            Path(args.output),
            canny_params=canny_params,
            robot_params=robot_params
        )

        print("\n" + "="*60)
        print("SUCCESS! Script generation completed")
        print("="*60)
        print(f"Number of paths: {result['num_paths']}")
        print(f"Total points: {result['total_points']}")
        print(f"Output script: {result['output_script']}")
        print(f"Output directory: {result['output_dir']}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
