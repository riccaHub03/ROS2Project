#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
# Il workspace e' opzionale: senza colcon build il container deve comunque avviarsi
if [ -f /workspace/ROS2_workspace/install/setup.bash ]; then
    source /workspace/ROS2_workspace/install/setup.bash
else
    echo "[entrypoint] ROS2_workspace non compilato (colcon build): salto il source" >&2
fi
exec "$@"
