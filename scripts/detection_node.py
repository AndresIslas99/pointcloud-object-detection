#!/usr/bin/env python3
# =============================================================================
# Optimized 3D Object Detection for Jetson AGX Orin
# Author: Andrés Islas Bravo
# 
# Architecture:
#   - INSTANT detection (no temporal delay)
#   - Point cloud accumulation for rotating LiDAR
#   - Proper use_sim_time support for Isaac Sim
#   - Designed for Jetson AGX Orin (12-core CPU + Ampere GPU)
# =============================================================================

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time
from collections import deque
import threading

from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class Detection3DResult:
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
    track_id: int = -1
    num_points: int = 0
    hits: int = 1


@dataclass
class TrackedObject:
    detection: Detection3DResult
    last_seen: float
    first_seen: float
    hit_count: int = 1
    miss_count: int = 0


class OptimizedDetectionNode(Node):
    
    CLASS_COLORS = {
        "Worker": (0.0, 1.0, 0.0, 0.9),
        "Forklift": (1.0, 0.5, 0.0, 0.9),
        "Object": (0.3, 0.7, 1.0, 0.9),
    }
    
    SIZE_TEMPLATES = {
        "Worker": {"min": (0.2, 0.2, 0.8), "max": (1.2, 1.2, 2.2), "typical": (0.5, 0.5, 1.7)},
        "Forklift": {"min": (1.2, 0.6, 1.0), "max": (5.0, 3.0, 3.5), "typical": (2.5, 1.2, 2.0)}
    }
    
    def __init__(self):
        super().__init__('pointcloud_detection')
        
        # use_sim_time is built-in, don't declare it
        self._declare_parameters()
        self._get_parameters()
        
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_track_id = 0
        self.track_lock = threading.Lock()
        
        # Point cloud accumulation for rotating LiDAR
        # LiDAR: 10Hz scan rate, 360° FOV, ~64k points per full rotation
        self.accumulated_points: List[np.ndarray] = []
        self.accumulation_start_time = 0.0
        self.accumulation_window = 1.0 / self.scan_rate  # Accumulate for one full rotation
        self.last_header = None
        
        self.last_process_time = 0.0
        self.process_interval = 1.0 / self.publish_rate_limit if self.publish_rate_limit > 0 else 0
        
        self.frame_count = 0
        self.process_times = deque(maxlen=100)
        
        # QoS for Isaac Sim
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        
        self.pointcloud_sub = self.create_subscription(PointCloud2, self.input_topic, self.pointcloud_callback, qos)
        self.detection_pub = self.create_publisher(Detection3DArray, self.output_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        
        # Timer for processing accumulated points
        self.process_timer = self.create_timer(self.process_interval, self.process_accumulated)
        
        self._log_config()
    
    def _declare_parameters(self):
        self.declare_parameter('input_topic', '/front_3d_lidar/lidar_points')
        self.declare_parameter('output_topic', '/detections')
        self.declare_parameter('marker_topic', '/detection_markers')
        self.declare_parameter('min_range', 0.8)
        self.declare_parameter('max_range', 35.0)
        self.declare_parameter('min_height', -0.5)
        self.declare_parameter('max_height', 3.5)
        self.declare_parameter('ground_threshold', 0.10)
        self.declare_parameter('cluster_eps', 0.5)
        self.declare_parameter('cluster_min_points', 8)
        self.declare_parameter('score_threshold', 0.30)
        self.declare_parameter('min_hits_to_confirm', 1)
        self.declare_parameter('max_age_seconds', 1.5)  # Longer for stability
        self.declare_parameter('publish_rate_limit', 10.0)
        self.declare_parameter('frame_id', '')
        self.declare_parameter('scan_rate', 10.0)  # LiDAR scan rate Hz
        self.declare_parameter('marker_lifetime', 0.5)  # Longer lifetime for stability
    
    def _get_parameters(self):
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.min_range = self.get_parameter('min_range').value
        self.max_range = self.get_parameter('max_range').value
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        self.ground_threshold = self.get_parameter('ground_threshold').value
        self.cluster_eps = self.get_parameter('cluster_eps').value
        self.cluster_min_points = self.get_parameter('cluster_min_points').value
        self.score_threshold = self.get_parameter('score_threshold').value
        self.min_hits_to_confirm = self.get_parameter('min_hits_to_confirm').value
        self.max_age_seconds = self.get_parameter('max_age_seconds').value
        self.publish_rate_limit = self.get_parameter('publish_rate_limit').value
        self.frame_id_override = self.get_parameter('frame_id').value
        self.scan_rate = self.get_parameter('scan_rate').value
        self.marker_lifetime = self.get_parameter('marker_lifetime').value
    
    def _log_config(self):
        # Check if using sim time
        use_sim = self.get_parameter('use_sim_time').value if self.has_parameter('use_sim_time') else False
        self.get_logger().info("=" * 60)
        self.get_logger().info("OPTIMIZED 3D Detection - Jetson AGX Orin")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  use_sim_time: {use_sim}")
        self.get_logger().info(f"  Input: {self.input_topic}")
        self.get_logger().info(f"  Range: {self.min_range}m - {self.max_range}m")
        self.get_logger().info(f"  Cluster: eps={self.cluster_eps}, min_pts={self.cluster_min_points}")
        self.get_logger().info(f"  Score threshold: {self.score_threshold}")
        self.get_logger().info(f"  LiDAR scan rate: {self.scan_rate}Hz")
        self.get_logger().info(f"  Accumulation window: {self.accumulation_window*1000:.0f}ms")
        self.get_logger().info(f"  Marker lifetime: {self.marker_lifetime}s")
        self.get_logger().info(f"  sklearn: {SKLEARN_AVAILABLE}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Detection node ready!")
    
    def get_time_sec(self) -> float:
        """Get current time in seconds (uses sim time if enabled)"""
        return self.get_clock().now().nanoseconds / 1e9
    
    def pointcloud_callback(self, msg: PointCloud2):
        """Accumulate point clouds from rotating LiDAR"""
        points = self.pointcloud2_to_numpy(msg)
        if points is None or len(points) == 0:
            return
        
        current_time = self.get_time_sec()
        
        # Start new accumulation window if needed
        if len(self.accumulated_points) == 0:
            self.accumulation_start_time = current_time
        
        self.accumulated_points.append(points)
        self.last_header = msg.header
    
    def process_accumulated(self):
        """Process accumulated point clouds (called by timer)"""
        if len(self.accumulated_points) == 0 or self.last_header is None:
            return
        
        current_time = self.get_time_sec()
        elapsed = current_time - self.accumulation_start_time
        
        # Wait for accumulation window OR process if we have enough points
        total_points = sum(len(p) for p in self.accumulated_points)
        
        # Process if: window elapsed OR we have >50k points (nearly full scan)
        if elapsed < self.accumulation_window and total_points < 50000:
            return
        
        start_time = time.perf_counter()
        
        # Merge accumulated points
        if len(self.accumulated_points) == 1:
            points = self.accumulated_points[0]
        else:
            points = np.vstack(self.accumulated_points)
        
        # Clear accumulator
        self.accumulated_points = []
        
        frame_id = self.frame_id_override if self.frame_id_override else self.last_header.frame_id
        
        # Detect
        detections = self.detect(points, current_time)
        
        # Create header with current time
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        
        self.detection_pub.publish(self.create_detection_msg(detections, header))
        self.marker_pub.publish(self.create_marker_msg(detections, header))
        
        proc_time = time.perf_counter() - start_time
        self.process_times.append(proc_time)
        self.frame_count += 1
        
        if self.frame_count % 30 == 0:
            avg_ms = np.mean(self.process_times) * 1000
            fps = 1.0 / np.mean(self.process_times) if self.process_times else 0
            self.get_logger().info(
                f"Frame {self.frame_count}: {len(detections)} det | "
                f"{len(points)} pts | {avg_ms:.1f}ms | {fps:.0f}FPS | "
                f"{len(self.tracked_objects)} tracks"
            )
    
    def pointcloud2_to_numpy(self, msg: PointCloud2) -> Optional[np.ndarray]:
        try:
            x_off = y_off = z_off = None
            for f in msg.fields:
                if f.name == 'x': x_off = f.offset
                elif f.name == 'y': y_off = f.offset
                elif f.name == 'z': z_off = f.offset
            
            if None in (x_off, y_off, z_off):
                return None
            
            data = np.frombuffer(msg.data, dtype=np.uint8)
            point_step = msg.point_step
            num_points = msg.width * msg.height
            
            points = np.zeros((num_points, 3), dtype=np.float32)
            
            for i in range(num_points):
                off = i * point_step
                points[i, 0] = np.frombuffer(data[off + x_off:off + x_off + 4], dtype=np.float32)[0]
                points[i, 1] = np.frombuffer(data[off + y_off:off + y_off + 4], dtype=np.float32)[0]
                points[i, 2] = np.frombuffer(data[off + z_off:off + z_off + 4], dtype=np.float32)[0]
            
            valid = np.isfinite(points).all(axis=1)
            return points[valid]
        except Exception as e:
            self.get_logger().error(f"PointCloud2 error: {e}")
            return None
    
    def detect(self, points: np.ndarray, current_time: float) -> List[Detection3DResult]:
        # Filter
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        mask = (distances >= self.min_range) & (distances <= self.max_range)
        mask &= (points[:, 2] >= self.min_height) & (points[:, 2] <= self.max_height)
        points = points[mask]
        
        if len(points) < self.cluster_min_points:
            return []
        
        # Ground removal
        z_ground = np.percentile(points[:, 2], 3)
        points = points[points[:, 2] > z_ground + self.ground_threshold]
        
        if len(points) < self.cluster_min_points:
            return []
        
        # Cluster
        clusters = self.cluster(points)
        
        # Convert to detections
        raw = []
        for c in clusters:
            det = self.cluster_to_detection(c)
            if det:
                raw.append(det)
        
        raw = self.nms(raw)
        return self.track(raw, current_time)
    
    def cluster(self, points: np.ndarray) -> List[np.ndarray]:
        if not SKLEARN_AVAILABLE:
            return []
        
        if len(points) > 40000:
            idx = np.random.choice(len(points), 40000, replace=False)
            points = points[idx]
        
        labels = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_points, n_jobs=-1).fit(points[:, :2]).labels_
        
        clusters = []
        for label in set(labels):
            if label == -1:
                continue
            c = points[labels == label]
            if len(c) >= self.cluster_min_points:
                clusters.append(c)
        
        return clusters
    
    def cluster_to_detection(self, cluster: np.ndarray) -> Optional[Detection3DResult]:
        if len(cluster) < self.cluster_min_points:
            return None
        
        min_pt, max_pt = cluster.min(axis=0), cluster.max(axis=0)
        center = (min_pt + max_pt) / 2
        dims = max_pt - min_pt
        
        l, w, h = float(dims[0]), float(dims[1]), float(dims[2])
        
        if h < 0.25 or max(l, w, h) > 8.0 or min(l, w) < 0.08:
            return None
        
        class_id, class_name, conf = self.classify(l, w, h, len(cluster))
        
        if conf < self.score_threshold:
            class_id, class_name, conf = 2, "Object", self.score_threshold
        
        yaw = self.estimate_yaw(cluster[:, :2])
        if w > l:
            l, w = w, l
            yaw += np.pi / 2
        
        return Detection3DResult(
            class_id=class_id, class_name=class_name, confidence=float(conf),
            x=float(center[0]), y=float(center[1]), z=float(center[2]),
            length=l, width=w, height=h, yaw=float(yaw), num_points=len(cluster)
        )
    
    def classify(self, l: float, w: float, h: float, n: int) -> Tuple[int, str, float]:
        dims = np.array([l, w, h])
        best = (2, "Object", 0.35)
        
        for name, tmpl in self.SIZE_TEMPLATES.items():
            min_d, max_d, typ_d = np.array(tmpl["min"]), np.array(tmpl["max"]), np.array(tmpl["typical"])
            
            if np.all(dims >= min_d * 0.5) and np.all(dims <= max_d * 1.6):
                diff = np.abs(dims - typ_d) / (typ_d + 0.1)
                score = float(1.0 - np.mean(diff) * 0.35)
                score = max(0.35, min(0.95, score))
                
                if score > best[2]:
                    best = (0 if name == "Worker" else 1, name, score)
        
        return best
    
    def estimate_yaw(self, pts: np.ndarray) -> float:
        if len(pts) < 3:
            return 0.0
        try:
            centered = pts - pts.mean(axis=0)
            cov = np.cov(centered.T)
            if cov.shape == (2, 2):
                _, vecs = np.linalg.eigh(cov)
                return float(np.arctan2(vecs[1, 1], vecs[0, 1]))
        except:
            pass
        return 0.0
    
    def nms(self, dets: List[Detection3DResult]) -> List[Detection3DResult]:
        if len(dets) <= 1:
            return dets
        
        dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
        keep, suppressed = [], set()
        
        for i, d1 in enumerate(dets):
            if i in suppressed:
                continue
            keep.append(d1)
            for j in range(i + 1, len(dets)):
                if j in suppressed:
                    continue
                d2 = dets[j]
                if np.sqrt((d1.x - d2.x)**2 + (d1.y - d2.y)**2) < max(d1.length, d2.length) * 0.5:
                    suppressed.add(j)
        
        return keep
    
    def track(self, dets: List[Detection3DResult], t: float) -> List[Detection3DResult]:
        with self.track_lock:
            confirmed = []
            matched_tracks = set()
            
            for det in dets:
                best_tid, best_dist = None, 2.0
                
                for tid, trk in self.tracked_objects.items():
                    if tid in matched_tracks:
                        continue
                    dist = np.sqrt((det.x - trk.detection.x)**2 + (det.y - trk.detection.y)**2)
                    if dist < best_dist:
                        best_dist, best_tid = dist, tid
                
                if best_tid is not None:
                    trk = self.tracked_objects[best_tid]
                    # Smooth
                    det.x = 0.7 * det.x + 0.3 * trk.detection.x
                    det.y = 0.7 * det.y + 0.3 * trk.detection.y
                    trk.detection = det
                    det.track_id = best_tid
                    det.hits = trk.hit_count + 1
                    trk.hit_count += 1
                    trk.last_seen = t
                    trk.miss_count = 0
                    matched_tracks.add(best_tid)
                    
                    if trk.hit_count > 2:
                        det.confidence = min(0.95, det.confidence + 0.03 * trk.hit_count)
                    
                    if trk.hit_count >= self.min_hits_to_confirm:
                        confirmed.append(det)
                else:
                    self.tracked_objects[self.next_track_id] = TrackedObject(detection=det, last_seen=t, first_seen=t)
                    det.track_id = self.next_track_id
                    det.hits = 1
                    self.next_track_id += 1
                    
                    if self.min_hits_to_confirm <= 1:
                        confirmed.append(det)
            
            # Cleanup
            to_del = [tid for tid, trk in self.tracked_objects.items()
                      if tid not in matched_tracks and (t - trk.last_seen > self.max_age_seconds or trk.miss_count > 10)]
            for tid in to_del:
                del self.tracked_objects[tid]
            for tid in self.tracked_objects:
                if tid not in matched_tracks:
                    self.tracked_objects[tid].miss_count += 1
        
        return confirmed
    
    def create_detection_msg(self, dets: List[Detection3DResult], header: Header) -> Detection3DArray:
        msg = Detection3DArray()
        msg.header = header
        
        for det in dets:
            d = Detection3D()
            d.header = header
            d.bbox.center.position.x = float(det.x)
            d.bbox.center.position.y = float(det.y)
            d.bbox.center.position.z = float(det.z)
            q = self.euler_to_quat(0, 0, det.yaw)
            d.bbox.center.orientation.x = float(q[0])
            d.bbox.center.orientation.y = float(q[1])
            d.bbox.center.orientation.z = float(q[2])
            d.bbox.center.orientation.w = float(q[3])
            d.bbox.size.x = float(det.length)
            d.bbox.size.y = float(det.width)
            d.bbox.size.z = float(det.height)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(det.class_id)
            hyp.hypothesis.score = float(det.confidence)
            d.results.append(hyp)
            msg.detections.append(d)
        
        return msg
    
    def create_marker_msg(self, dets: List[Detection3DResult], header: Header) -> MarkerArray:
        markers = MarkerArray()
        
        # Convert marker_lifetime to nanoseconds
        lifetime_ns = int(self.marker_lifetime * 1e9)
        
        clear = Marker()
        clear.header = header
        clear.ns = "det"
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        
        for i, det in enumerate(dets):
            color = self.CLASS_COLORS.get(det.class_name, (0.5, 0.5, 0.5, 0.9))
            
            # Wireframe
            wire = Marker()
            wire.header = header
            wire.ns = "wire"
            wire.id = det.track_id if det.track_id >= 0 else i + 1000  # Use track_id for stable IDs
            wire.type = Marker.LINE_LIST
            wire.action = Marker.ADD
            wire.scale.x = 0.06
            wire.color.r, wire.color.g, wire.color.b, wire.color.a = color[0], color[1], color[2], 1.0
            wire.lifetime.nanosec = lifetime_ns
            
            corners = self.get_corners(det)
            for e1, e2 in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
                wire.points.append(self.pt(corners[e1]))
                wire.points.append(self.pt(corners[e2]))
            markers.markers.append(wire)
            
            # Box
            box = Marker()
            box.header = header
            box.ns = "box"
            box.id = det.track_id if det.track_id >= 0 else i + 1000
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = float(det.x)
            box.pose.position.y = float(det.y)
            box.pose.position.z = float(det.z)
            q = self.euler_to_quat(0, 0, det.yaw)
            box.pose.orientation.x = float(q[0])
            box.pose.orientation.y = float(q[1])
            box.pose.orientation.z = float(q[2])
            box.pose.orientation.w = float(q[3])
            box.scale.x = float(det.length)
            box.scale.y = float(det.width)
            box.scale.z = float(det.height)
            box.color.r, box.color.g, box.color.b = color[0], color[1], color[2]
            box.color.a = 0.25
            box.lifetime.nanosec = lifetime_ns
            markers.markers.append(box)
            
            # Label
            txt = Marker()
            txt.header = header
            txt.ns = "txt"
            txt.id = det.track_id if det.track_id >= 0 else i + 1000
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(det.x)
            txt.pose.position.y = float(det.y)
            txt.pose.position.z = float(det.z + det.height/2 + 0.4)
            txt.scale.z = 0.3
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = f"{det.class_name} {det.confidence:.0%}\nID:{det.track_id} hits:{det.hits}"
            txt.lifetime.nanosec = lifetime_ns
            markers.markers.append(txt)
        
        return markers
    
    def get_corners(self, det):
        c, s = np.cos(det.yaw), np.sin(det.yaw)
        R = np.array([[c, -s], [s, c]])
        l, w, h = det.length/2, det.width/2, det.height/2
        corners_2d = np.array([[-l,-w],[l,-w],[l,w],[-l,w]]) @ R.T
        return [np.array([det.x+xy[0], det.y+xy[1], det.z-h]) for xy in corners_2d] + \
               [np.array([det.x+xy[0], det.y+xy[1], det.z+h]) for xy in corners_2d]
    
    def pt(self, xyz):
        p = Point()
        p.x, p.y, p.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        return p
    
    @staticmethod
    def euler_to_quat(r, p, y):
        cy, sy, cp, sp, cr, sr = np.cos(y*0.5), np.sin(y*0.5), np.cos(p*0.5), np.sin(p*0.5), np.cos(r*0.5), np.sin(r*0.5)
        return [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]


def main(args=None):
    rclpy.init(args=args)
    node = OptimizedDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
