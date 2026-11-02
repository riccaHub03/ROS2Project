#!/usr/bin/env python3
"""
Launch File ROS 2: Stack di Percezione RGB-D e Grasping 6-DoF
Avvia:
  1. simulated_rgbd_camera_node (Camera RGB-D con de-proiezione reale e TF2 statici)
  2. grasp_planner_node (Pianificatore di presa 6-DoF con riferimento STL)
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # 1. Telecamera RGB-D Simulata
        Node(
            package='lbr_pybullet_tutorial', # o eseguibile diretto
            executable='simulated_rgbd_camera_node',
            name='simulated_rgbd_camera_node',
            output='screen',
            parameters=[{'publish_hz': 10.0}]
        ),

        # 2. Pianificatore di Grasping 6-DoF
        Node(
            package='lbr_pybullet_tutorial',
            executable='grasp_planner_node',
            name='grasp_planner_node',
            output='screen'
        )
    ])
