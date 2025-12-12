#!/usr/bin/env python3
# =============================================================================
# Improved 3D Object Detector with Clustering and Temporal Filtering
# Author: Andrés Islas Bravo
# Description: Robust detection with fallback clustering when TensorRT unavailable
# =============================================================================

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import deque
import time
import os

# Try to import TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

# Try to import Open3D for clustering
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


@dataclass
class Detection3D:
    """3D detection result"""
    class_id: int
    class_name: str
    confidence: float
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float
    track_id: int = -1  # For temporal tracking


@dataclass
class TrackedObject:
    """Object being tracked over time"""
    detection: Detection3D
    last_seen: float
    hit_count: int = 1
    miss_count: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))


class PointPillarsDetector:
    """
    Improved 3D Object Detector with multiple backends:
    1. TensorRT PointPillars (when available)
    2. Clustering-based detection (fallback)
    
    Features:
    - Temporal filtering to reduce false positives
    - Size-based classification
    - Velocity estimation
    - Configurable detection zones
    """
    
    # Class definitions
    CLASS_NAMES = {
        0: "Worker",      # Small objects (pedestrian-sized)
        1: "Forklift",    # Large objects (vehicle-sized)
    }
    
    # Size templates for classification [length, width, height]
    SIZE_TEMPLATES = {
        "Worker": {
            "min": [0.3, 0.3, 1.2],
            "max": [0.8, 0.8, 2.0],
            "typical": [0.5, 0.5, 1.7]
        },
        "Forklift": {
            "min": [1.5, 1.0, 1.5],
            "max": [4.0, 2.5, 2.5],
            "typical": [2.5, 1.5, 2.0]
        }
    }
    
    def __init__(
        self,
        engine_path: str = None,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        # Clustering parameters
        cluster_eps: float = 0.5,        # DBSCAN epsilon
        cluster_min_points: int = 10,     # Minimum points per cluster
        # Temporal filtering
        min_hits: int = 3,                # Hits before confirming detection
        max_age: float = 0.5,             # Seconds before dropping track
        # Detection zone
        min_range: float = 1.0,
        max_range: float = 50.0,
        min_height: float = -0.5,
        max_height: float = 3.0,
        # Ground removal
        ground_threshold: float = 0.15,
    ):
        self.engine_path = engine_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        
        # Clustering params
        self.cluster_eps = cluster_eps
        self.cluster_min_points = cluster_min_points
        
        # Temporal filtering
        self.min_hits = min_hits
        self.max_age = max_age
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_track_id = 0
        
        # Detection zone
        self.min_range = min_range
        self.max_range = max_range
        self.min_height = min_height
        self.max_height = max_height
        self.ground_threshold = ground_threshold
        
        # Statistics
        self.inference_times: List[float] = []
        self.detection_mode = "unknown"
        
        # Try to load TensorRT engine
        self.trt_engine = None
        if engine_path and os.path.exists(engine_path) and TRT_AVAILABLE:
            self._load_trt_engine(engine_path)
            self.detection_mode = "tensorrt"
        elif O3D_AVAILABLE:
            self.detection_mode = "clustering"
            print("Using Open3D clustering for detection")
        else:
            self.detection_mode = "basic_clustering"
            print("Using basic numpy clustering for detection")
    
    def _load_trt_engine(self, engine_path: str):
        """Load TensorRT engine"""
        try:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            runtime = trt.Runtime(TRT_LOGGER)
            self.trt_engine = runtime.deserialize_cuda_engine(engine_data)
            print(f"Loaded TensorRT engine: {engine_path}")
        except Exception as e:
            print(f"Failed to load TensorRT engine: {e}")
            self.trt_engine = None
    
    def detect(self, points: np.ndarray) -> List[Detection3D]:
        """
        Run detection on point cloud.
        
        Args:
            points: (N, 4) array [x, y, z, intensity]
            
        Returns:
            List of confirmed Detection3D objects
        """
        start_time = time.perf_counter()
        
        # Preprocess: filter points
        points = self._filter_points(points)
        
        if len(points) < self.cluster_min_points:
            return []
        
        # Run detection based on available backend
        if self.trt_engine is not None:
            raw_detections = self._detect_tensorrt(points)
        elif O3D_AVAILABLE:
            raw_detections = self._detect_clustering_o3d(points)
        else:
            raw_detections = self._detect_clustering_numpy(points)
        
        # Apply NMS
        raw_detections = self._nms(raw_detections)
        
        # Temporal filtering
        confirmed_detections = self._temporal_filter(raw_detections)
        
        # Track inference time
        elapsed = time.perf_counter() - start_time
        self.inference_times.append(elapsed)
        
        return confirmed_detections
    
    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        """Filter points by range, height, and remove ground"""
        if len(points) == 0:
            return points
        
        # Range filter
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        range_mask = (distances >= self.min_range) & (distances <= self.max_range)
        
        # Height filter
        height_mask = (points[:, 2] >= self.min_height) & (points[:, 2] <= self.max_height)
        
        # Combined mask
        mask = range_mask & height_mask
        filtered = points[mask]
        
        # Ground removal using RANSAC-like approach
        if len(filtered) > 100:
            filtered = self._remove_ground(filtered)
        
        return filtered
    
    def _remove_ground(self, points: np.ndarray) -> np.ndarray:
        """Remove ground points using height-based filtering"""
        # Simple approach: remove points below threshold
        # More sophisticated: use RANSAC plane fitting
        
        # Estimate ground height from lowest points
        z_values = points[:, 2]
        z_percentile = np.percentile(z_values, 5)  # 5th percentile as ground estimate
        
        # Keep points above ground + threshold
        above_ground = points[:, 2] > (z_percentile + self.ground_threshold)
        
        return points[above_ground]
    
    def _detect_clustering_o3d(self, points: np.ndarray) -> List[Detection3D]:
        """Detect objects using Open3D DBSCAN clustering"""
        detections = []
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # DBSCAN clustering
        labels = np.array(pcd.cluster_dbscan(
            eps=self.cluster_eps,
            min_points=self.cluster_min_points,
            print_progress=False
        ))
        
        # Process each cluster
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = points[cluster_mask]
            
            detection = self._cluster_to_detection(cluster_points)
            if detection is not None:
                detections.append(detection)
        
        return detections
    
    def _detect_clustering_numpy(self, points: np.ndarray) -> List[Detection3D]:
        """Simple grid-based clustering when Open3D not available"""
        detections = []
        
        # Voxel grid clustering
        voxel_size = self.cluster_eps * 2
        
        # Compute voxel indices
        voxel_indices = np.floor(points[:, :3] / voxel_size).astype(int)
        
        # Hash voxels
        voxel_hash = (voxel_indices[:, 0] * 73856093 ^ 
                      voxel_indices[:, 1] * 19349663 ^ 
                      voxel_indices[:, 2] * 83492791)
        
        # Group by voxel
        unique_hashes, inverse = np.unique(voxel_hash, return_inverse=True)
        
        # Find connected components (simplified)
        visited = set()
        clusters = []
        
        for i, h in enumerate(unique_hashes):
            if h in visited:
                continue
            
            # Get all points in this voxel
            mask = inverse == i
            if mask.sum() >= self.cluster_min_points:
                cluster_points = points[mask]
                clusters.append(cluster_points)
                visited.add(h)
        
        # Convert clusters to detections
        for cluster_points in clusters:
            if len(cluster_points) >= self.cluster_min_points:
                detection = self._cluster_to_detection(cluster_points)
                if detection is not None:
                    detections.append(detection)
        
        return detections
    
    def _detect_tensorrt(self, points: np.ndarray) -> List[Detection3D]:
        """Run TensorRT inference (placeholder - implement based on your model)"""
        # This would use the actual TensorRT engine
        # For now, fall back to clustering
        return self._detect_clustering_o3d(points) if O3D_AVAILABLE else self._detect_clustering_numpy(points)
    
    def _cluster_to_detection(self, cluster_points: np.ndarray) -> Optional[Detection3D]:
        """Convert a point cluster to a Detection3D object"""
        if len(cluster_points) < self.cluster_min_points:
            return None
        
        # Compute bounding box
        min_pt = cluster_points[:, :3].min(axis=0)
        max_pt = cluster_points[:, :3].max(axis=0)
        
        center = (min_pt + max_pt) / 2
        dimensions = max_pt - min_pt
        
        length, width, height = dimensions[0], dimensions[1], dimensions[2]
        
        # Filter by size - reject too small or too large
        if length < 0.2 or width < 0.2 or height < 0.3:
            return None  # Too small
        if length > 10 or width > 10 or height > 5:
            return None  # Too large
        
        # Classify based on size
        class_id, class_name, confidence = self._classify_by_size(length, width, height)
        
        # Reject low confidence
        if confidence < self.score_threshold:
            return None
        
        # Estimate yaw from point distribution (PCA-like)
        yaw = self._estimate_yaw(cluster_points[:, :2] - center[:2])
        
        return Detection3D(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            x=float(center[0]),
            y=float(center[1]),
            z=float(center[2]),
            length=float(max(length, width)),  # Ensure length >= width
            width=float(min(length, width)),
            height=float(height),
            yaw=float(yaw)
        )
    
    def _classify_by_size(self, length: float, width: float, height: float) -> Tuple[int, str, float]:
        """Classify object based on dimensions"""
        dims = np.array([length, width, height])
        
        best_class = 1  # Default to Forklift
        best_name = "Forklift"
        best_score = 0.0
        
        for class_name, templates in self.SIZE_TEMPLATES.items():
            min_dims = np.array(templates["min"])
            max_dims = np.array(templates["max"])
            typical_dims = np.array(templates["typical"])
            
            # Check if within bounds
            if np.all(dims >= min_dims * 0.8) and np.all(dims <= max_dims * 1.2):
                # Score based on similarity to typical size
                diff = np.abs(dims - typical_dims) / typical_dims
                score = float(1.0 - np.mean(diff))
                score = max(0.3, min(0.95, score))  # Clamp
                
                if score > best_score:
                    best_score = score
                    best_name = class_name
                    best_class = 0 if class_name == "Worker" else 1
        
        return best_class, best_name, best_score
    
    def _estimate_yaw(self, points_2d: np.ndarray) -> float:
        """Estimate orientation from 2D point distribution"""
        if len(points_2d) < 3:
            return 0.0
        
        # Simple PCA
        centered = points_2d - points_2d.mean(axis=0)
        cov = np.cov(centered.T)
        
        if cov.shape == (2, 2):
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Principal direction
            principal = eigenvectors[:, np.argmax(eigenvalues)]
            yaw = np.arctan2(principal[1], principal[0])
        else:
            yaw = 0.0
        
        return yaw
    
    def _nms(self, detections: List[Detection3D]) -> List[Detection3D]:
        """Non-Maximum Suppression"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        keep = []
        suppressed = set()
        
        for i, det_i in enumerate(detections):
            if i in suppressed:
                continue
            keep.append(det_i)
            
            for j in range(i + 1, len(detections)):
                if j in suppressed:
                    continue
                
                det_j = detections[j]
                
                # Check 3D distance
                dist = np.sqrt(
                    (det_i.x - det_j.x)**2 +
                    (det_i.y - det_j.y)**2 +
                    (det_i.z - det_j.z)**2
                )
                
                # Suppress if too close
                min_size = min(det_i.length, det_j.length)
                if dist < min_size * 0.5:
                    suppressed.add(j)
        
        return keep
    
    def _temporal_filter(self, detections: List[Detection3D]) -> List[Detection3D]:
        """
        Temporal filtering using simple tracking.
        Objects must be seen multiple times before being confirmed.
        """
        current_time = time.time()
        confirmed = []
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for det_idx, det in enumerate(detections):
            best_track_id = None
            best_distance = float('inf')
            
            for track_id, track in self.tracked_objects.items():
                if track_id in matched_tracks:
                    continue
                
                # Distance between detection and track
                dist = np.sqrt(
                    (det.x - track.detection.x)**2 +
                    (det.y - track.detection.y)**2
                )
                
                # Match threshold based on expected movement
                threshold = 2.0  # meters
                
                if dist < threshold and dist < best_distance:
                    best_distance = dist
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                track = self.tracked_objects[best_track_id]
                
                # Update velocity estimate
                dt = current_time - track.last_seen
                if dt > 0:
                    track.velocity = np.array([
                        (det.x - track.detection.x) / dt,
                        (det.y - track.detection.y) / dt,
                        (det.z - track.detection.z) / dt
                    ])
                
                # Update detection
                track.detection = det
                track.detection.track_id = best_track_id
                track.last_seen = current_time
                track.hit_count += 1
                track.miss_count = 0
                
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)
                
                # Confirm if enough hits
                if track.hit_count >= self.min_hits:
                    # Boost confidence based on track history
                    boosted_conf = min(0.95, det.confidence + 0.1 * (track.hit_count - self.min_hits))
                    det.confidence = boosted_conf
                    confirmed.append(det)
            else:
                # Create new track
                new_track = TrackedObject(
                    detection=det,
                    last_seen=current_time,
                    hit_count=1
                )
                det.track_id = self.next_track_id
                self.tracked_objects[self.next_track_id] = new_track
                self.next_track_id += 1
                matched_detections.add(det_idx)
        
        # Update unmatched tracks
        for track_id in list(self.tracked_objects.keys()):
            if track_id not in matched_tracks:
                track = self.tracked_objects[track_id]
                track.miss_count += 1
                
                # Remove old tracks
                if current_time - track.last_seen > self.max_age:
                    del self.tracked_objects[track_id]
                elif track.miss_count > 5:
                    del self.tracked_objects[track_id]
        
        return confirmed
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        if len(self.inference_times) == 0:
            return {"avg_ms": 0, "fps": 0, "mode": self.detection_mode}
        
        times_ms = np.array(self.inference_times) * 1000
        return {
            "avg_ms": float(np.mean(times_ms)),
            "min_ms": float(np.min(times_ms)),
            "max_ms": float(np.max(times_ms)),
            "fps": float(1000 / np.mean(times_ms)),
            "mode": self.detection_mode,
            "active_tracks": len(self.tracked_objects)
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.inference_times = []
    
    def reset_tracks(self):
        """Reset all tracks"""
        self.tracked_objects = {}
        self.next_track_id = 0
