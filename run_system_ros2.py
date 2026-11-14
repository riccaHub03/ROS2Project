#!/usr/bin/env python3
"""ROS 2 simulation entry point. Prefer scripts/run_ros2_system.sh on Linux/WSL2."""
from pipeline.ros2.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
