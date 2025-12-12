#!/usr/bin/env python3
# =============================================================================
# PointPillars TensorRT Inference Engine
# Author: Andrés Islas Bravo
# Description: TensorRT FP16 inference for real-time 3D object detection
# =============================================================================

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time
import os

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("WARNING: TensorRT not available. Using mock inference for testing.")


@dataclass
class Detection3D:
    """3D detection result"""
    class_id: int
    class_name: str
    confidence: float
    x: float       # Center X
    y: float       # Center Y
    z: float       # Center Z
    length: float  # Dimension along X
    width: float   # Dimension along Y
    height: float  # Dimension along Z
    yaw: float     # Rotation around Z


class PointPillarsTRT:
    """
    TensorRT inference engine for PointPillars 3D object detection.
    
    Optimized for Jetson AGX Orin with FP16 precision.
    Expected latency: ~7ms (146 FPS) on Orin MAXN mode.
    
    Attributes:
        engine_path: Path to TensorRT engine file (.engine)
        class_names: List of class names for detection
        score_threshold: Minimum confidence score for detections
        nms_threshold: IoU threshold for NMS
    """
    
    # Default class mapping (KITTI to Industrial)
    DEFAULT_CLASS_NAMES = {
        0: "Worker",      # Pedestrian -> Worker
        1: "Forklift",    # Car -> Forklift
        2: "Cyclist",     # Keep as is for now
    }
    
    # Point cloud range for voxelization (KITTI default)
    POINT_CLOUD_RANGE = [0, -39.68, -3, 69.12, 39.68, 1]
    VOXEL_SIZE = [0.16, 0.16, 4]
    MAX_POINTS_PER_VOXEL = 32
    MAX_VOXELS = 40000
    
    def __init__(
        self,
        engine_path: str,
        class_names: Optional[Dict[int, str]] = None,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        device_id: int = 0
    ):
        """
        Initialize PointPillars TensorRT inference engine.
        
        Args:
            engine_path: Path to serialized TensorRT engine
            class_names: Dict mapping class IDs to names
            score_threshold: Minimum confidence for detection
            nms_threshold: IoU threshold for NMS
            device_id: CUDA device ID
        """
        self.engine_path = engine_path
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.device_id = device_id
        
        # Performance tracking
        self.inference_times: List[float] = []
        
        if TRT_AVAILABLE and os.path.exists(engine_path):
            self._load_engine()
        else:
            print(f"Engine not found at {engine_path}. Using mock inference.")
            self.engine = None
            self.context = None
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self._allocate_buffers()
        
        print(f"Loaded TensorRT engine: {self.engine_path}")
        print(f"  - Num bindings: {self.engine.num_bindings}")
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = self.engine.get_binding_dtype(i)
            print(f"  - Binding {i}: {name}, shape={shape}, dtype={dtype}")
    
    def _allocate_buffers(self):
        """Allocate GPU memory for input/output tensors"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape)
            
            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(i):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
    
    def preprocess(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess point cloud for PointPillars.
        
        Converts raw points to voxel representation (pillars).
        
        Args:
            points: (N, 4) array of [x, y, z, intensity]
            
        Returns:
            voxels: (max_voxels, max_points, 4) voxel features
            coords: (max_voxels, 3) voxel coordinates [z, y, x]
            num_points: (max_voxels,) number of points per voxel
        """
        # Filter points within range
        mask = (
            (points[:, 0] >= self.POINT_CLOUD_RANGE[0]) &
            (points[:, 0] < self.POINT_CLOUD_RANGE[3]) &
            (points[:, 1] >= self.POINT_CLOUD_RANGE[1]) &
            (points[:, 1] < self.POINT_CLOUD_RANGE[4]) &
            (points[:, 2] >= self.POINT_CLOUD_RANGE[2]) &
            (points[:, 2] < self.POINT_CLOUD_RANGE[5])
        )
        points = points[mask]
        
        # Voxelization
        voxel_size = np.array(self.VOXEL_SIZE)
        point_cloud_range = np.array(self.POINT_CLOUD_RANGE)
        grid_size = np.round((point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size).astype(np.int32)
        
        # Compute voxel indices
        voxel_indices = np.floor((points[:, :3] - point_cloud_range[:3]) / voxel_size).astype(np.int32)
        
        # Create voxel hash for grouping
        voxel_hash = voxel_indices[:, 0] + voxel_indices[:, 1] * grid_size[0] + voxel_indices[:, 2] * grid_size[0] * grid_size[1]
        
        # Get unique voxels
        unique_hashes, inverse_indices, voxel_counts = np.unique(
            voxel_hash, return_inverse=True, return_counts=True
        )
        
        num_voxels = min(len(unique_hashes), self.MAX_VOXELS)
        
        # Initialize output arrays
        voxels = np.zeros((self.MAX_VOXELS, self.MAX_POINTS_PER_VOXEL, 4), dtype=np.float32)
        coords = np.zeros((self.MAX_VOXELS, 3), dtype=np.int32)
        num_points = np.zeros((self.MAX_VOXELS,), dtype=np.int32)
        
        # Fill voxels
        for i in range(num_voxels):
            voxel_mask = inverse_indices == i
            voxel_points = points[voxel_mask][:self.MAX_POINTS_PER_VOXEL]
            
            num_pts = len(voxel_points)
            voxels[i, :num_pts] = voxel_points
            num_points[i] = num_pts
            
            # Get voxel coordinate
            first_point_idx = np.where(voxel_mask)[0][0]
            coords[i] = voxel_indices[first_point_idx][[2, 1, 0]]  # z, y, x order
        
        return voxels, coords, num_points
    
    def detect(self, points: np.ndarray) -> List[Detection3D]:
        """
        Run 3D object detection on point cloud.
        
        Args:
            points: (N, 4) array of [x, y, z, intensity]
            
        Returns:
            List of Detection3D objects
        """
        start_time = time.perf_counter()
        
        if self.engine is None:
            # Mock inference for testing without TensorRT
            return self._mock_detect(points)
        
        # Preprocess
        voxels, coords, num_points = self.preprocess(points)
        
        # Copy inputs to GPU
        np.copyto(self.inputs[0]['host'], voxels.ravel())
        np.copyto(self.inputs[1]['host'], coords.ravel())
        np.copyto(self.inputs[2]['host'], num_points.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        
        # Copy outputs to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        self.stream.synchronize()
        
        # Parse detections
        detections = self._parse_outputs()
        
        # Apply NMS
        detections = self._nms(detections)
        
        # Track inference time
        elapsed = time.perf_counter() - start_time
        self.inference_times.append(elapsed)
        
        return detections
    
    def _parse_outputs(self) -> List[Detection3D]:
        """Parse network outputs into Detection3D objects"""
        detections = []
        
        # Output format: [batch, num_boxes, 9] where 9 = [x, y, z, l, w, h, yaw, class, score]
        boxes = self.outputs[0]['host'].reshape(-1, 9)
        
        for box in boxes:
            score = box[8]
            if score < self.score_threshold:
                continue
            
            class_id = int(box[7])
            detection = Detection3D(
                class_id=class_id,
                class_name=self.class_names.get(class_id, f"class_{class_id}"),
                confidence=float(score),
                x=float(box[0]),
                y=float(box[1]),
                z=float(box[2]),
                length=float(box[3]),
                width=float(box[4]),
                height=float(box[5]),
                yaw=float(box[6])
            )
            detections.append(detection)
        
        return detections
    
    def _nms(self, detections: List[Detection3D]) -> List[Detection3D]:
        """Apply 3D Non-Maximum Suppression"""
        if len(detections) == 0:
            return detections
        
        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        # Simple IoU-based NMS (for full 3D IoU, use specialized library)
        keep = []
        suppressed = set()
        
        for i, det_i in enumerate(detections):
            if i in suppressed:
                continue
            keep.append(det_i)
            
            for j, det_j in enumerate(detections[i+1:], i+1):
                if j in suppressed:
                    continue
                if det_i.class_id != det_j.class_id:
                    continue
                
                # Simple 2D IoU (BEV)
                iou = self._compute_bev_iou(det_i, det_j)
                if iou > self.nms_threshold:
                    suppressed.add(j)
        
        return keep
    
    def _compute_bev_iou(self, det1: Detection3D, det2: Detection3D) -> float:
        """Compute Bird's Eye View IoU between two detections"""
        # Simplified axis-aligned box IoU
        x1_min, x1_max = det1.x - det1.length/2, det1.x + det1.length/2
        y1_min, y1_max = det1.y - det1.width/2, det1.y + det1.width/2
        
        x2_min, x2_max = det2.x - det2.length/2, det2.x + det2.length/2
        y2_min, y2_max = det2.y - det2.width/2, det2.y + det2.width/2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        xi_max = min(x1_max, x2_max)
        yi_min = max(y1_min, y2_min)
        yi_max = min(y1_max, y2_max)
        
        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        
        area1 = det1.length * det1.width
        area2 = det2.length * det2.width
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _mock_detect(self, points: np.ndarray) -> List[Detection3D]:
        """
        Mock detection for testing without TensorRT.
        Generates random detections for visualization testing.
        """
        import random
        
        detections = []
        num_detections = random.randint(0, 5)
        
        for _ in range(num_detections):
            class_id = random.choice([0, 1])
            detection = Detection3D(
                class_id=class_id,
                class_name=self.class_names.get(class_id, "Unknown"),
                confidence=random.uniform(0.5, 0.99),
                x=random.uniform(5, 30),
                y=random.uniform(-10, 10),
                z=random.uniform(-0.5, 1.0),
                length=random.uniform(1.5, 4.5) if class_id == 1 else random.uniform(0.4, 0.8),
                width=random.uniform(1.5, 2.0) if class_id == 1 else random.uniform(0.4, 0.8),
                height=random.uniform(1.5, 2.5) if class_id == 1 else random.uniform(1.5, 1.9),
                yaw=random.uniform(-np.pi, np.pi)
            )
            detections.append(detection)
        
        # Simulate inference time
        time.sleep(0.007)  # ~7ms mock latency
        
        return detections
    
    def get_stats(self) -> Dict:
        """Get inference statistics"""
        if len(self.inference_times) == 0:
            return {"avg_ms": 0, "min_ms": 0, "max_ms": 0, "fps": 0}
        
        times_ms = np.array(self.inference_times) * 1000
        return {
            "avg_ms": float(np.mean(times_ms)),
            "min_ms": float(np.min(times_ms)),
            "max_ms": float(np.max(times_ms)),
            "std_ms": float(np.std(times_ms)),
            "fps": float(1000 / np.mean(times_ms)),
            "num_samples": len(self.inference_times)
        }
    
    def reset_stats(self):
        """Reset inference statistics"""
        self.inference_times = []
