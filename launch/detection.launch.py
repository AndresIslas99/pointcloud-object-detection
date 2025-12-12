#!/usr/bin/env python3
# =============================================================================
# Detection Launch File
# Author: Andrés Islas Bravo
# Description: Launch 3D object detection with optional RViz visualization
# =============================================================================

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('pointcloud_detection_3d')
    
    # =========================================================================
    # Launch Arguments
    # =========================================================================
    
    args = [
        DeclareLaunchArgument('input_topic', 
            default_value='/front_3d_lidar/lidar_points',
            description='Input point cloud topic'),
        
        DeclareLaunchArgument('output_topic',
            default_value='/detections',
            description='Output Detection3DArray topic'),
        
        DeclareLaunchArgument('marker_topic',
            default_value='/detection_markers',
            description='Output MarkerArray for RViz'),
        
        # Detection zone
        DeclareLaunchArgument('min_range', default_value='1.5',
            description='Minimum detection range (m)'),
        DeclareLaunchArgument('max_range', default_value='25.0',
            description='Maximum detection range (m)'),
        DeclareLaunchArgument('min_height', default_value='-0.2',
            description='Minimum height (m)'),
        DeclareLaunchArgument('max_height', default_value='2.5',
            description='Maximum height (m)'),
        
        # Ground removal
        DeclareLaunchArgument('ground_threshold', default_value='0.2',
            description='Height above ground to keep points'),
        
        # Clustering
        DeclareLaunchArgument('cluster_eps', default_value='0.6',
            description='DBSCAN epsilon (m)'),
        DeclareLaunchArgument('cluster_min_points', default_value='20',
            description='Minimum points per cluster'),
        
        # Classification
        DeclareLaunchArgument('score_threshold', default_value='0.55',
            description='Minimum confidence score'),
        
        # Temporal filtering
        DeclareLaunchArgument('min_hits_to_confirm', default_value='3',
            description='Hits before confirming detection'),
        DeclareLaunchArgument('max_age_seconds', default_value='0.6',
            description='Track timeout (s)'),
        
        # Performance
        DeclareLaunchArgument('publish_rate', default_value='10.0',
            description='Max publish rate (Hz)'),
        
        # Simulation
        DeclareLaunchArgument('use_sim_time', default_value='true',
            description='Use simulation time'),
        
        # Visualization
        DeclareLaunchArgument('launch_rviz', default_value='false',
            description='Launch RViz'),
        DeclareLaunchArgument('rviz_config',
            default_value=os.path.join(pkg_share, 'rviz', 'detection.rviz'),
            description='RViz config file'),
    ]
    
    # =========================================================================
    # Detection Node
    # =========================================================================
    detection_node = Node(
        package='pointcloud_detection_3d',
        executable='detection_node.py',
        name='pointcloud_detection',
        output='screen',
        parameters=[{
            'input_topic': LaunchConfiguration('input_topic'),
            'output_topic': LaunchConfiguration('output_topic'),
            'marker_topic': LaunchConfiguration('marker_topic'),
            'min_range': LaunchConfiguration('min_range'),
            'max_range': LaunchConfiguration('max_range'),
            'min_height': LaunchConfiguration('min_height'),
            'max_height': LaunchConfiguration('max_height'),
            'ground_threshold': LaunchConfiguration('ground_threshold'),
            'cluster_eps': LaunchConfiguration('cluster_eps'),
            'cluster_min_points': LaunchConfiguration('cluster_min_points'),
            'score_threshold': LaunchConfiguration('score_threshold'),
            'min_hits_to_confirm': LaunchConfiguration('min_hits_to_confirm'),
            'max_age_seconds': LaunchConfiguration('max_age_seconds'),
            'publish_rate_limit': LaunchConfiguration('publish_rate'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )
    
    # =========================================================================
    # RViz Node (optional)
    # =========================================================================
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('launch_rviz'))
    )
    
    return LaunchDescription(args + [detection_node, rviz_node])
