"""After colcon build: ros2 launch surgical_executor full_system.launch.py gui:=true."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def start(context):
    get = lambda name: LaunchConfiguration(name).perform(context)
    args = ["--profile", get("profile"), "--seed", get("seed"), "--motion-speed", get("motion_speed"),
            "--gui" if get("gui").lower() == "true" else "--headless"]
    for name, flag in (("instruction", "--instruction"), ("config", "--config"), ("tool", "--tool"),
                       ("ollama_url", "--ollama-url"), ("sam3_model", "--sam3-model")):
        if get(name):
            args += [flag, get(name)]
    return [Node(package="surgical_executor", executable="full_system", output="screen", arguments=args)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("profile", default_value="full_system"),
        DeclareLaunchArgument("instruction", default_value=""),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("seed", default_value="42"),
        DeclareLaunchArgument("motion_speed", default_value="1"),
        DeclareLaunchArgument("config", default_value=""),
        DeclareLaunchArgument("tool", default_value=""),
        DeclareLaunchArgument("ollama_url", default_value=""),
        DeclareLaunchArgument("sam3_model", default_value=""),
        OpaqueFunction(function=start),
    ])
