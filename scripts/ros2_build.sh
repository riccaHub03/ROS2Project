#!/usr/bin/env bash
# Compila con colcon il package del progetto (ros2/surgical_executor), da eseguire DENTRO il container.
#   bash scripts/ros2_build.sh            # solo surgical_executor (veloce, non serve il resto del workspace)
#   bash scripts/ros2_build.sh --all      # anche i repository di ros2/workspace.repos in ROS2_workspace/src
# Dopo la build apri un nuovo terminale nel container (l'entrypoint carica ROS2_workspace/install).
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# ROS setup scripts intentionally read optional, unset environment variables.
set +u
source /opt/ros/humble/setup.bash
set -u
mkdir -p "$PROJECT_DIR/ROS2_workspace"
cd "$PROJECT_DIR/ROS2_workspace"

if [[ "${1:-}" == "--all" ]]; then
    [[ -d src ]] || { mkdir -p src && vcs import src < "$PROJECT_DIR/ros2/workspace.repos"; }
    colcon build --symlink-install --base-paths src "$PROJECT_DIR/ros2"
else
    colcon build --symlink-install --base-paths "$PROJECT_DIR/ros2" --packages-select surgical_executor
fi
