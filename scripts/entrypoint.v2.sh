#!/usr/bin/env bash
# =============================================================================
# entrypoint.v2.sh – Entrypoint per il container ROS 2 v2
# Sourca automaticamente ROS 2 e il workspace se presente
# =============================================================================
set -eo pipefail

# Source ROS 2
source /opt/ros/${ROS_DISTRO}/setup.bash

# Source workspace install se esiste (montato come volume)
if [ -f /workspace/install/local_setup.bash ]; then
    source /workspace/install/local_setup.bash
fi

if [ -f /workspace/ROS2_workspace/install/local_setup.bash ]; then
    source /workspace/ROS2_workspace/install/local_setup.bash
fi

exec "$@"
