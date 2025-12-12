#!/usr/bin/env python3
# =============================================================================
# Detection Utilities for ROS2
# Author: Andrés Islas Bravo
# Description: Utilities for point cloud conversion, message creation, and visualization
# =============================================================================

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import struct

# ROS2 Messages
from sensor_msgs.msg import PointCloud2, PointField
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Vector3, Quaternion
from std_msgs.msg import Header, ColorRGBA

# Use transforms3d for quaternion math (more portable than tf_transformations)
try:
    from transforms3d.euler import euler2quat
    def quaternion_from_euler(roll, pitch, yaw):
        # transforms3d uses w,x,y,z order, ROS uses x,y,z,w
        w, x, y, z = euler2quat(roll, pitch, yaw)
        return [x, y, z, w]
except ImportError:
    # Fallback implementation
    import math
    def quaternion_from_euler(roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [x, y, z, w]


# Class configuration
CLASS_NAMES = {
    0: "Worker",      # Pedestrian
    1: "Forklift",    # Car
    2: "Cyclist",     # Cyclist
}

CLASS_COLORS = {
    0: (0.0, 1.0, 0.0, 0.8),    # Worker: Green
    1: (1.0, 0.5, 0.0, 0.8),    # Forklift: Orange
    2: (0.0, 0.5, 1.0, 0.8),    # Cyclist: Blue
}


def pointcloud2_to_array(msg: PointCloud2) -> np.ndarray:
    """
    Convert ROS2 PointCloud2 message to numpy array.
    
    Args:
        msg: PointCloud2 message
        
    Returns:
        points: (N, 4) array of [x, y, z, intensity]
    """
    # Get field offsets
    field_names = [f.name for f in msg.fields]
    
    # Find offsets for x, y, z, intensity
    x_offset = next((f.offset for f in msg.fields if f.name == 'x'), None)
    y_offset = next((f.offset for f in msg.fields if f.name == 'y'), None)
    z_offset = next((f.offset for f in msg.fields if f.name == 'z'), None)
    intensity_offset = next((f.offset for f in msg.fields if f.name in ['intensity', 'i']), None)
    
    if x_offset is None:
        raise ValueError("PointCloud2 message must have 'x' field")
    
    # Parse points
    point_step = msg.point_step
    num_points = msg.width * msg.height
    
    points = np.zeros((num_points, 4), dtype=np.float32)
    data = np.frombuffer(msg.data, dtype=np.uint8)
    
    for i in range(num_points):
        offset = i * point_step
        
        points[i, 0] = struct.unpack('f', data[offset + x_offset:offset + x_offset + 4].tobytes())[0]
        points[i, 1] = struct.unpack('f', data[offset + y_offset:offset + y_offset + 4].tobytes())[0]
        points[i, 2] = struct.unpack('f', data[offset + z_offset:offset + z_offset + 4].tobytes())[0]
        
        if intensity_offset is not None:
            points[i, 3] = struct.unpack('f', data[offset + intensity_offset:offset + intensity_offset + 4].tobytes())[0]
        else:
            points[i, 3] = 1.0  # Default intensity
    
    # Filter invalid points (NaN, Inf)
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    
    return points


def pointcloud2_to_array_fast(msg: PointCloud2) -> np.ndarray:
    """
    Fast conversion of PointCloud2 to numpy array.
    Assumes standard XYZI format with float32 fields.
    
    Args:
        msg: PointCloud2 message
        
    Returns:
        points: (N, 4) array of [x, y, z, intensity]
    """
    # Convert data buffer directly
    dtype_list = []
    for field in msg.fields:
        if field.datatype == PointField.FLOAT32:
            dtype_list.append((field.name, np.float32))
        elif field.datatype == PointField.FLOAT64:
            dtype_list.append((field.name, np.float64))
        elif field.datatype == PointField.INT32:
            dtype_list.append((field.name, np.int32))
        elif field.datatype == PointField.UINT32:
            dtype_list.append((field.name, np.uint32))
        elif field.datatype == PointField.UINT8:
            dtype_list.append((field.name, np.uint8))
    
    # Create structured array
    cloud_arr = np.frombuffer(msg.data, dtype=dtype_list)
    
    # Extract XYZI
    field_names = [f.name for f in msg.fields]
    
    x = cloud_arr['x'].astype(np.float32) if 'x' in field_names else np.zeros(len(cloud_arr), dtype=np.float32)
    y = cloud_arr['y'].astype(np.float32) if 'y' in field_names else np.zeros(len(cloud_arr), dtype=np.float32)
    z = cloud_arr['z'].astype(np.float32) if 'z' in field_names else np.zeros(len(cloud_arr), dtype=np.float32)
    
    if 'intensity' in field_names:
        intensity = cloud_arr['intensity'].astype(np.float32)
    elif 'i' in field_names:
        intensity = cloud_arr['i'].astype(np.float32)
    else:
        intensity = np.ones(len(cloud_arr), dtype=np.float32)
    
    points = np.column_stack([x, y, z, intensity])
    
    # Filter invalid points
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    
    return points


def create_detection3d_msg(
    detection,  # Detection3D dataclass from pointpillars_trt
    header: Header
) -> Detection3D:
    """
    Create vision_msgs/Detection3D message from detection result.
    
    Args:
        detection: Detection3D dataclass
        header: Message header with timestamp and frame_id
        
    Returns:
        Detection3D message
    """
    msg = Detection3D()
    msg.header = header
    
    # Object hypothesis
    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = str(detection.class_id)
    hypothesis.hypothesis.score = detection.confidence
    
    # Pose (center of bounding box)
    hypothesis.pose.pose.position.x = detection.x
    hypothesis.pose.pose.position.y = detection.y
    hypothesis.pose.pose.position.z = detection.z
    
    # Orientation from yaw angle
    q = quaternion_from_euler(0, 0, detection.yaw)
    hypothesis.pose.pose.orientation.x = q[0]
    hypothesis.pose.pose.orientation.y = q[1]
    hypothesis.pose.pose.orientation.z = q[2]
    hypothesis.pose.pose.orientation.w = q[3]
    
    msg.results.append(hypothesis)
    
    # Bounding box
    msg.bbox.center.position.x = detection.x
    msg.bbox.center.position.y = detection.y
    msg.bbox.center.position.z = detection.z
    msg.bbox.center.orientation = hypothesis.pose.pose.orientation
    msg.bbox.size.x = detection.length
    msg.bbox.size.y = detection.width
    msg.bbox.size.z = detection.height
    
    return msg


def create_detection3d_array_msg(
    detections: List,  # List of Detection3D dataclass
    header: Header
) -> Detection3DArray:
    """
    Create vision_msgs/Detection3DArray message from detection results.
    
    Args:
        detections: List of Detection3D dataclass objects
        header: Message header
        
    Returns:
        Detection3DArray message
    """
    msg = Detection3DArray()
    msg.header = header
    
    for det in detections:
        det_msg = create_detection3d_msg(det, header)
        msg.detections.append(det_msg)
    
    return msg


def create_bbox_marker(
    detection,  # Detection3D dataclass
    marker_id: int,
    header: Header,
    namespace: str = "detections"
) -> Marker:
    """
    Create visualization_msgs/Marker for 3D bounding box.
    
    Args:
        detection: Detection3D dataclass
        marker_id: Unique marker ID
        header: Message header
        namespace: Marker namespace
        
    Returns:
        Marker message (LINE_LIST type for wireframe box)
    """
    marker = Marker()
    marker.header = header
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    
    # Line width
    marker.scale.x = 0.05
    
    # Color based on class
    color = CLASS_COLORS.get(detection.class_id, (1.0, 1.0, 1.0, 0.8))
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    
    # Lifetime
    marker.lifetime.sec = 0
    marker.lifetime.nanosec = 100000000  # 100ms
    
    # Get 8 corners of the 3D bounding box
    corners = _get_bbox_corners(
        detection.x, detection.y, detection.z,
        detection.length, detection.width, detection.height,
        detection.yaw
    )
    
    # Create 12 lines for wireframe box
    # Bottom face
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical
    ]
    
    for i, j in edges:
        p1 = Point()
        p1.x, p1.y, p1.z = corners[i]
        p2 = Point()
        p2.x, p2.y, p2.z = corners[j]
        marker.points.append(p1)
        marker.points.append(p2)
    
    return marker


def create_label_marker(
    detection,  # Detection3D dataclass
    marker_id: int,
    header: Header,
    namespace: str = "labels"
) -> Marker:
    """
    Create text label marker for detection.
    
    Args:
        detection: Detection3D dataclass
        marker_id: Unique marker ID
        header: Message header
        namespace: Marker namespace
        
    Returns:
        Marker message (TEXT_VIEW_FACING type)
    """
    marker = Marker()
    marker.header = header
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    
    # Position above the bounding box
    marker.pose.position.x = detection.x
    marker.pose.position.y = detection.y
    marker.pose.position.z = detection.z + detection.height / 2 + 0.5
    marker.pose.orientation.w = 1.0
    
    # Text scale
    marker.scale.z = 0.4
    
    # Color (white text)
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 1.0
    
    # Text content
    marker.text = f"{detection.class_name}: {detection.confidence:.2f}"
    
    # Lifetime
    marker.lifetime.sec = 0
    marker.lifetime.nanosec = 100000000  # 100ms
    
    return marker


def create_marker_array(
    detections: List,  # List of Detection3D dataclass
    header: Header
) -> MarkerArray:
    """
    Create MarkerArray for all detections (boxes + labels).
    
    Args:
        detections: List of Detection3D dataclass objects
        header: Message header
        
    Returns:
        MarkerArray message
    """
    marker_array = MarkerArray()
    
    # Clear old markers first
    clear_marker = Marker()
    clear_marker.header = header
    clear_marker.action = Marker.DELETEALL
    marker_array.markers.append(clear_marker)
    
    for i, det in enumerate(detections):
        # Bounding box
        bbox_marker = create_bbox_marker(det, i * 2, header, "detection_boxes")
        marker_array.markers.append(bbox_marker)
        
        # Label
        label_marker = create_label_marker(det, i * 2 + 1, header, "detection_labels")
        marker_array.markers.append(label_marker)
    
    return marker_array


def _get_bbox_corners(
    x: float, y: float, z: float,
    length: float, width: float, height: float,
    yaw: float
) -> np.ndarray:
    """
    Get 8 corners of a 3D bounding box.
    
    Args:
        x, y, z: Center position
        length, width, height: Box dimensions
        yaw: Rotation around Z axis (radians)
        
    Returns:
        corners: (8, 3) array of corner positions
    """
    # Half dimensions
    l, w, h = length / 2, width / 2, height / 2
    
    # Corners in local frame (before rotation)
    corners_local = np.array([
        [-l, -w, -h],  # 0: bottom-back-left
        [+l, -w, -h],  # 1: bottom-front-left
        [+l, +w, -h],  # 2: bottom-front-right
        [-l, +w, -h],  # 3: bottom-back-right
        [-l, -w, +h],  # 4: top-back-left
        [+l, -w, +h],  # 5: top-front-left
        [+l, +w, +h],  # 6: top-front-right
        [-l, +w, +h],  # 7: top-back-right
    ])
    
    # Rotation matrix around Z
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    
    # Rotate and translate
    corners_world = corners_local @ rotation.T + np.array([x, y, z])
    
    return corners_world


def filter_detections_by_range(
    detections: List,
    max_range: float = 50.0,
    min_range: float = 1.0
) -> List:
    """
    Filter detections by range from sensor.
    
    Args:
        detections: List of Detection3D dataclass objects
        max_range: Maximum detection range
        min_range: Minimum detection range
        
    Returns:
        Filtered list of detections
    """
    filtered = []
    for det in detections:
        dist = np.sqrt(det.x**2 + det.y**2)
        if min_range <= dist <= max_range:
            filtered.append(det)
    return filtered
