"""ros2 launch surgical_executor pick_and_place.launch.py grasp_report:=data/outputs/grasp.json"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    params = os.path.join(get_package_share_directory("surgical_executor"), "config", "executor.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("grasp_report", default_value="",
                              description="grasp.json della pipeline (vuoto: data/outputs/grasp.json)"),
        DeclareLaunchArgument("config_path", default_value="",
                              description="YAML dell'ambiente (vuoto: config/surgical_environment.yaml)"),
        DeclareLaunchArgument("destination_tray", default_value="tray_1"),
        Node(
            package="surgical_executor", executable="pick_and_place", name="surgical_pick_and_place",
            output="screen",
            parameters=[params, {
                "grasp_report": LaunchConfiguration("grasp_report"),
                "config_path": LaunchConfiguration("config_path"),
                "destination_tray": LaunchConfiguration("destination_tray"),
            }],
        ),
    ])
