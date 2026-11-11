#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
# La libreria Python `pipeline` (montata in /workspace) deve essere importabile dai nodi ROS 2
export PYTHONPATH="/workspace${PYTHONPATH:+:$PYTHONPATH}"
# Il workspace e' opzionale: senza colcon build il container deve comunque avviarsi
if [ -f /workspace/ROS2_workspace/install/setup.bash ]; then
    source /workspace/ROS2_workspace/install/setup.bash
else
    echo "[entrypoint] ROS2_workspace non compilato (scripts/ros2_build.sh): salto il source" >&2
fi
exec "$@"
