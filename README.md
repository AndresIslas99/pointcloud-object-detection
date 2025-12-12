# 3D Object Detection for LiDAR Point Clouds

Real-time 3D object detection using DBSCAN clustering for warehouse/industrial environments. Designed for NVIDIA Jetson AGX Orin with JetPack 6.x and ROS2 Humble.

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)
![Platform](https://img.shields.io/badge/Platform-Jetson%20AGX%20Orin-green)
![JetPack](https://img.shields.io/badge/JetPack-6.x-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **DBSCAN Clustering**: Efficient point cloud segmentation without pre-trained models
- **Temporal Filtering**: Reduces false positives by requiring consistent detections across frames
- **Size-Based Classification**: Detects workers and forklifts based on dimensional profiles
- **Docker Deployment**: Ready-to-use containers for JetPack 6.x
- **RViz Integration**: Real-time 3D visualization of detections
- **Isaac Sim Compatible**: Works with NVIDIA Isaac Sim for simulation testing

## Detection Classes

| Class | Typical Size (L×W×H) | Color |
|-------|---------------------|-------|
| Worker | 0.5 × 0.5 × 1.7 m | Green |
| Forklift | 2.5 × 1.2 × 2.0 m | Yellow |

## Quick Start

### Prerequisites

- NVIDIA Jetson AGX Orin with JetPack 6.x
- Docker with NVIDIA runtime
- ROS2 Humble (for host teleop)

### Docker Deployment (Recommended)

```bash
# Clone repository
git clone https://github.com/AndresIslas99/pointcloud-object-detection.git
cd pointcloud-object-detection/docker

# Build container
docker-compose build detection

# Enable X11 forwarding (for RViz)
xhost +local:docker

# Run detection
docker-compose run --rm detection
```

### Native Installation

```bash
# Install dependencies
sudo apt install ros-humble-vision-msgs ros-humble-tf2-ros python3-sklearn

# Clone to workspace
cd ~/ros2_ws/src
git clone https://github.com/AndresIslas99/pointcloud-object-detection.git

# Build
cd ~/ros2_ws
colcon build --packages-select pointcloud_detection_3d

# Source and run
source install/setup.bash
ros2 launch pointcloud_detection_3d detection.launch.py
```

## Usage

### Launch Options

```bash
# Basic detection
ros2 launch pointcloud_detection_3d detection.launch.py

# With custom parameters
ros2 launch pointcloud_detection_3d detection.launch.py \
    input_topic:=/lidar/points \
    min_hits_to_confirm:=5 \
    cluster_min_points:=30 \
    max_range:=20.0

# With RViz visualization
ros2 launch pointcloud_detection_3d detection.launch.py launch_rviz:=true
```

### Docker Services

```bash
# Detection only (no GUI)
docker-compose run --rm detection

# Detection with RViz
docker-compose run --rm detection-rviz

# RViz only (connect to existing node)
docker-compose run --rm --profile rviz rviz
```

### Robot Teleop (Isaac Sim)

```bash
# Install standard ROS2 teleop
sudo apt install ros-humble-teleop-twist-keyboard

# Run teleop
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Configuration

### Parameters (`config/detection_params.yaml`)

```yaml
# Range filtering
min_range: 1.5          # Minimum detection range (m)
max_range: 25.0         # Maximum detection range (m)
min_height: -0.2        # Minimum point height (m)
max_height: 2.5         # Maximum point height (m)

# Clustering
cluster_eps: 0.6        # DBSCAN epsilon (m)
cluster_min_points: 20  # Minimum points per cluster

# Classification
score_threshold: 0.55   # Minimum confidence score

# Temporal filtering (false positive reduction)
min_hits_to_confirm: 3  # Frames before confirming detection
max_age_seconds: 0.6    # Track timeout

# Performance
publish_rate_limit: 10.0  # Max publish rate (Hz)
```

### Tuning for False Positive Reduction

| Parameter | Effect | Trade-off |
|-----------|--------|-----------|
| `min_hits_to_confirm` ↑ | Fewer false positives | Slower initial detection |
| `cluster_min_points` ↑ | Filters small noise | May miss distant objects |
| `score_threshold` ↑ | Higher confidence only | May miss edge cases |
| `max_range` ↓ | Better precision | Reduced coverage |

## ROS2 Topics

### Subscribed
| Topic | Type | Description |
|-------|------|-------------|
| `/front_3d_lidar/lidar_points` | `sensor_msgs/PointCloud2` | Input LiDAR data |

### Published
| Topic | Type | Description |
|-------|------|-------------|
| `/detections_3d` | `vision_msgs/Detection3DArray` | 3D bounding boxes |
| `/detection_markers` | `visualization_msgs/MarkerArray` | RViz markers |

## Algorithm Overview

```
PointCloud Input
      │
      ▼
┌─────────────────┐
│ Range Filtering │  → Remove points outside [min_range, max_range]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ground Removal  │  → Estimate and remove ground plane
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DBSCAN Cluster  │  → Group nearby points into clusters
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Size Classify   │  → Match cluster dimensions to object classes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Temporal Filter │  → Require N consecutive detections
└────────┬────────┘
         │
         ▼
   Publish Detection
```

## Performance

Tested on Jetson AGX Orin (JetPack 6.1):

| Metric | Value |
|--------|-------|
| Processing Latency | 15-50 ms |
| Max Point Cloud Size | 100K+ points |
| Detection Rate | 10 Hz |
| False Positive Rate | <5% (with temporal filtering) |

## Project Structure

```
pointcloud_detection_3d/
├── config/
│   └── detection_params.yaml    # Detection parameters
├── docker/
│   ├── Dockerfile.jp6           # JetPack 6.x container
│   └── docker-compose.yml       # Service definitions
├── launch/
│   └── detection.launch.py      # Main launch file
├── rviz/
│   └── detection.rviz           # RViz configuration
├── scripts/
│   └── detection_node.py        # Main detection node
└── README.md
```

## Troubleshooting

### No detections appearing
1. Verify LiDAR topic: `ros2 topic echo /front_3d_lidar/lidar_points --once`
2. Check point count: May need to adjust `cluster_min_points` for sparse data
3. Verify range settings match your environment

### Too many false positives
1. Increase `min_hits_to_confirm` (3 → 5)
2. Increase `cluster_min_points` (20 → 30)
3. Raise `score_threshold` (0.55 → 0.65)

### Docker build fails
- Ensure JetPack 6.x is installed
- Run `docker-compose build --no-cache detection` for clean rebuild

## Future Improvements

- [ ] TensorRT PointPillars integration for higher accuracy
- [ ] Multi-class expansion (pallets, shelving, vehicles)
- [ ] Velocity estimation for moving objects
- [ ] Integration with Nav2 costmaps

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Andrés Islas Bravo**  
- GitHub: [@AndresIslas99](https://github.com/AndresIslas99)
- Email: andresislas99@icloud.com

## Acknowledgments

- NVIDIA Isaac Sim for simulation environment
- ROS2 Humble community
- scikit-learn for DBSCAN implementation
