#!/usr/bin/env python3
"""
Script di Avvio Rapido: Nodi ROS 2 Telecamera RGB-D Simulata & Grasping 6-DoF.
Avvia contemporaneamente:
  1. SimulatedRGBDCameraNode (Pubblica /camera/color, /camera/depth, /camera/camera_info, /camera/points e TF2)
  2. GraspPlannerNode (Ascolta RGB-D e SAM3, calcola la posa di presa 6-DoF e pubblica /grasp/target_pose)
"""

import sys
import os

# Assicura percorsi corretti
for p in ["/workspace", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import rclpy
from rclpy.executors import MultiThreadedExecutor

from pipeline.simulated_rgbd_camera_node import SimulatedRGBDCameraNode
from pipeline.grasp_planner_node import GraspPlannerNode
from pipeline.config_loader import load_environment_config


def main(args=None):
    rclpy.init(args=args)
    config = load_environment_config()

    print("\n" + "="*70)
    print("AVVIO STACK ROS 2: TELECAMERA RGB-D SIMULATA & GRASPING 6-DoF")
    print("="*70)

    cam_node = SimulatedRGBDCameraNode(config=config)
    grasp_node = GraspPlannerNode(config=config)

    executor = MultiThreadedExecutor()
    executor.add_node(cam_node)
    executor.add_node(grasp_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\nArresto nodi in corso...")
    finally:
        cam_node.destroy_node()
        grasp_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
