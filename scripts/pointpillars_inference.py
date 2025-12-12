#!/usr/bin/env python3
# =============================================================================
# Standalone PointPillars Inference Script
# Author: Andrés Islas Bravo
# Description: Run inference on PCD/BIN files without ROS2
# =============================================================================

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import time

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcloud_detection_3d.pointpillars_trt import PointPillarsTRT, Detection3D

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available, visualization disabled")


def load_pointcloud(file_path: str) -> np.ndarray:
    """
    Load point cloud from various formats.
    
    Supported formats:
        - .bin (KITTI format): N x 4 float32 binary
        - .pcd (Point Cloud Data)
        - .ply (Polygon File Format)
        - .npy (NumPy array)
        
    Args:
        file_path: Path to point cloud file
        
    Returns:
        points: (N, 4) array [x, y, z, intensity]
    """
    ext = Path(file_path).suffix.lower()
    
    if ext == '.bin':
        # KITTI binary format
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        
    elif ext == '.npy':
        # NumPy format
        points = np.load(file_path)
        if points.shape[1] == 3:
            # Add intensity if missing
            intensity = np.ones((len(points), 1), dtype=np.float32)
            points = np.hstack([points, intensity])
            
    elif ext in ['.pcd', '.ply']:
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D required for PCD/PLY files")
        
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # Add intensity
        if pcd.colors:
            # Use grayscale from colors as intensity
            colors = np.asarray(pcd.colors)
            intensity = np.mean(colors, axis=1, keepdims=True).astype(np.float32)
        else:
            intensity = np.ones((len(points), 1), dtype=np.float32)
        
        points = np.hstack([points, intensity])
        
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return points.astype(np.float32)


def visualize_detections(
    points: np.ndarray,
    detections: list,
    window_name: str = "3D Detection Results"
):
    """
    Visualize point cloud with 3D bounding boxes using Open3D.
    
    Args:
        points: (N, 4) point cloud
        detections: List of Detection3D objects
        window_name: Visualization window name
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available for visualization")
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Color by intensity
    if points.shape[1] >= 4:
        intensities = points[:, 3]
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-6)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = intensities  # Red channel
        colors[:, 1] = intensities  # Green channel
        colors[:, 2] = intensities  # Blue channel
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create bounding boxes
    geometries = [pcd]
    
    # Class colors
    CLASS_COLORS = {
        0: [0, 1, 0],      # Worker: Green
        1: [1, 0.5, 0],    # Forklift: Orange
        2: [0, 0.5, 1],    # Cyclist: Blue
    }
    
    for det in detections:
        # Create oriented bounding box
        center = [det.x, det.y, det.z]
        extent = [det.length, det.width, det.height]
        
        # Rotation matrix from yaw
        R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0, 0, det.yaw])
        
        bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
        bbox.color = CLASS_COLORS.get(det.class_id, [1, 1, 1])
        
        geometries.append(bbox)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    geometries.append(coord_frame)
    
    # Visualize
    print("\nVisualization Controls:")
    print("  Mouse: Rotate view")
    print("  Scroll: Zoom")
    print("  Q: Quit")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1280,
        height=720,
        point_show_normal=False
    )


def print_detections(detections: list):
    """Print detection results in a table"""
    print("\n" + "="*80)
    print(f"{'ID':>4} {'Class':>12} {'Conf':>8} {'X':>8} {'Y':>8} {'Z':>8} {'L':>6} {'W':>6} {'H':>6}")
    print("-"*80)
    
    for i, det in enumerate(detections):
        print(f"{i:>4} {det.class_name:>12} {det.confidence:>8.2f} "
              f"{det.x:>8.2f} {det.y:>8.2f} {det.z:>8.2f} "
              f"{det.length:>6.2f} {det.width:>6.2f} {det.height:>6.2f}")
    
    print("="*80)
    print(f"Total: {len(detections)} detections")


def main():
    parser = argparse.ArgumentParser(
        description='Run PointPillars inference on point cloud files'
    )
    parser.add_argument(
        'input',
        help='Input point cloud file (.bin, .pcd, .ply, .npy)'
    )
    parser.add_argument(
        '--model', '-m',
        default='/opt/models/pointpillars.engine',
        help='Path to TensorRT engine file'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.3,
        help='Detection confidence threshold'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Visualize results with Open3D'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output file for detections (JSON)'
    )
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print("="*60)
    print("PointPillars 3D Object Detection")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print()
    
    # Load point cloud
    print("Loading point cloud...")
    points = load_pointcloud(args.input)
    print(f"  Loaded {len(points):,} points")
    print(f"  Range X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Range Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Range Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = PointPillarsTRT(
        engine_path=args.model,
        score_threshold=args.threshold
    )
    
    # Run inference
    print("\nRunning inference...")
    start = time.perf_counter()
    detections = detector.detect(points)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"  Inference time: {elapsed:.2f} ms")
    print(f"  Detections: {len(detections)}")
    
    # Print results
    if detections:
        print_detections(detections)
    else:
        print("\nNo detections found.")
    
    # Save to JSON
    if args.output:
        import json
        output_data = {
            'input_file': args.input,
            'num_points': len(points),
            'inference_ms': elapsed,
            'detections': [
                {
                    'class_id': d.class_id,
                    'class_name': d.class_name,
                    'confidence': d.confidence,
                    'x': d.x, 'y': d.y, 'z': d.z,
                    'length': d.length, 'width': d.width, 'height': d.height,
                    'yaw': d.yaw
                }
                for d in detections
            ]
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Visualize
    if args.visualize:
        visualize_detections(points, detections)


if __name__ == '__main__':
    main()
