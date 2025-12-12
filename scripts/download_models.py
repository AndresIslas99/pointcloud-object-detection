#!/usr/bin/env python3
# =============================================================================
# Model Download and TensorRT Conversion Script
# Author: Andrés Islas Bravo
# Description: Downloads pre-trained PointPillars and converts to TensorRT
# =============================================================================

import os
import sys
import argparse
import subprocess
import urllib.request
import hashlib
from pathlib import Path


# Pre-trained model URLs
MODELS = {
    'pointpillars_kitti': {
        'url': 'https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars/releases/download/v1.0/pointpillars_kitti.onnx',
        'sha256': None,  # Add checksum if available
        'description': 'PointPillars trained on KITTI dataset (Car, Pedestrian, Cyclist)',
    },
    'pointpillars_nuscenes': {
        'url': 'https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars/releases/download/v1.0/pointpillars_nuscenes.onnx',
        'sha256': None,
        'description': 'PointPillars trained on nuScenes dataset',
    },
}

# Alternative: OpenPCDet pre-trained models
OPENPCDET_MODELS = {
    'pointpillar_kitti': {
        'url': 'https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view',
        'config': 'pointpillar.yaml',
        'description': 'OpenPCDet PointPillars (KITTI)',
    },
    'second_kitti': {
        'url': 'https://drive.google.com/file/d/1-01zsPOsqanZQqIIyy7FpNXStL3y4jdR/view',
        'config': 'second.yaml',
        'description': 'OpenPCDet SECOND (KITTI)',
    },
}


def download_file(url: str, output_path: str, sha256: str = None) -> bool:
    """
    Download a file with progress indication.
    
    Args:
        url: Download URL
        output_path: Local file path
        sha256: Expected SHA256 checksum (optional)
        
    Returns:
        True if successful
    """
    print(f"Downloading: {url}")
    print(f"To: {output_path}")
    
    try:
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download with progress
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rProgress: {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print("\nDownload complete!")
        
        # Verify checksum
        if sha256:
            print("Verifying checksum...")
            with open(output_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            if file_hash != sha256:
                print(f"ERROR: Checksum mismatch!")
                print(f"  Expected: {sha256}")
                print(f"  Got: {file_hash}")
                return False
            print("Checksum OK!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        return False


def convert_to_tensorrt(
    onnx_path: str,
    engine_path: str,
    precision: str = 'fp16',
    workspace_mb: int = 4096
) -> bool:
    """
    Convert ONNX model to TensorRT engine.
    
    Args:
        onnx_path: Input ONNX model path
        engine_path: Output TensorRT engine path
        precision: fp16, fp32, or int8
        workspace_mb: GPU workspace size in MB
        
    Returns:
        True if successful
    """
    print(f"\nConverting to TensorRT ({precision})...")
    print(f"Input: {onnx_path}")
    print(f"Output: {engine_path}")
    
    # Build trtexec command
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
        f'--workspace={workspace_mb}',
    ]
    
    if precision == 'fp16':
        cmd.append('--fp16')
    elif precision == 'int8':
        cmd.append('--int8')
    
    # Add dynamic shape support for point clouds
    cmd.extend([
        '--minShapes=points:1x10000x4',
        '--optShapes=points:1x50000x4',
        '--maxShapes=points:1x204800x4',
    ])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR: TensorRT conversion failed!")
            print(result.stderr)
            return False
        
        print("TensorRT conversion successful!")
        
        # Print engine info
        engine_size = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"Engine size: {engine_size:.2f} MB")
        
        return True
        
    except FileNotFoundError:
        print("ERROR: trtexec not found. Is TensorRT installed?")
        return False
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        return False


def setup_cuda_pointpillars(output_dir: str) -> bool:
    """
    Clone and build NVIDIA CUDA-PointPillars.
    
    Args:
        output_dir: Directory for the cloned repository
        
    Returns:
        True if successful
    """
    print("\n" + "="*60)
    print("Setting up NVIDIA CUDA-PointPillars")
    print("="*60)
    
    repo_dir = os.path.join(output_dir, 'CUDA-PointPillars')
    
    # Clone if not exists
    if not os.path.exists(repo_dir):
        print("Cloning repository...")
        cmd = [
            'git', 'clone',
            'https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars.git',
            repo_dir
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("ERROR: Failed to clone repository")
            return False
    else:
        print(f"Repository already exists at {repo_dir}")
    
    # Build
    print("\nBuilding CUDA-PointPillars...")
    build_dir = os.path.join(repo_dir, 'build')
    os.makedirs(build_dir, exist_ok=True)
    
    # CMake
    cmake_cmd = ['cmake', '..', '-DCMAKE_BUILD_TYPE=Release']
    result = subprocess.run(cmake_cmd, cwd=build_dir)
    if result.returncode != 0:
        print("ERROR: CMake configuration failed")
        return False
    
    # Make
    make_cmd = ['make', '-j4']
    result = subprocess.run(make_cmd, cwd=build_dir)
    if result.returncode != 0:
        print("ERROR: Build failed")
        return False
    
    print("Build successful!")
    
    # Download pre-trained model
    print("\nDownloading pre-trained model...")
    model_script = os.path.join(repo_dir, 'tool', 'download_model.sh')
    if os.path.exists(model_script):
        result = subprocess.run(['bash', model_script], cwd=repo_dir)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download and setup PointPillars models for Jetson AGX Orin'
    )
    parser.add_argument(
        '--output', '-o',
        default='/opt/models',
        help='Output directory for models'
    )
    parser.add_argument(
        '--model', '-m',
        choices=['pointpillars_kitti', 'pointpillars_nuscenes', 'all'],
        default='pointpillars_kitti',
        help='Model to download'
    )
    parser.add_argument(
        '--precision', '-p',
        choices=['fp16', 'fp32', 'int8'],
        default='fp16',
        help='TensorRT precision'
    )
    parser.add_argument(
        '--workspace', '-w',
        type=int,
        default=4096,
        help='TensorRT workspace size (MB)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download, only convert existing ONNX'
    )
    parser.add_argument(
        '--setup-cuda-pointpillars',
        action='store_true',
        help='Clone and build NVIDIA CUDA-PointPillars'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("="*60)
    print("PointPillars Model Setup for Jetson AGX Orin")
    print("="*60)
    print(f"Output directory: {args.output}")
    print(f"Precision: {args.precision}")
    print()
    
    # Option to setup CUDA-PointPillars
    if args.setup_cuda_pointpillars:
        if not setup_cuda_pointpillars(args.output):
            sys.exit(1)
        return
    
    # Download models
    models_to_download = [args.model] if args.model != 'all' else list(MODELS.keys())
    
    for model_name in models_to_download:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}")
            continue
        
        model_info = MODELS[model_name]
        onnx_path = os.path.join(args.output, f'{model_name}.onnx')
        engine_path = os.path.join(args.output, f'{model_name}.engine')
        
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Description: {model_info['description']}")
        print('='*60)
        
        # Download
        if not args.skip_download:
            if not download_file(model_info['url'], onnx_path, model_info.get('sha256')):
                print(f"Failed to download {model_name}")
                continue
        
        # Convert to TensorRT
        if os.path.exists(onnx_path):
            if not convert_to_tensorrt(
                onnx_path, engine_path,
                precision=args.precision,
                workspace_mb=args.workspace
            ):
                print(f"Failed to convert {model_name}")
                continue
        else:
            print(f"ONNX file not found: {onnx_path}")
            continue
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)
    print(f"\nModels are stored in: {args.output}")
    print("\nTo use with the detection node:")
    print(f"  ros2 launch pointcloud_detection_3d detection.launch.py \\")
    print(f"      engine_path:={args.output}/pointpillars_kitti.engine")


if __name__ == '__main__':
    main()
