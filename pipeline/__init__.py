"""
Package autonomo della pipeline medica robotica (KUKA LBR iiwa + ROS 2 + SAM 3 + Qwen).
"""
from .config_loader import load_environment_config
from .frame_transformer import FrameTransformer
from .step5_grasp_geometry import run_grasp_geometry
