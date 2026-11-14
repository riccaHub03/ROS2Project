#!/usr/bin/env bash
# Self-contained Humble runtime on Linux/WSL2. Does not start/stop AI services.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROS2_IMAGE="${ROS2_IMAGE:-surgical-ros2-system}"

if [[ "${1:-}" == "build" ]]; then
    shift
    exec docker build -f "$PROJECT_DIR/docker/ros2-system.Dockerfile" -t "$ROS2_IMAGE" "$@" "$PROJECT_DIR"
fi
if ! docker image inspect "$ROS2_IMAGE" >/dev/null 2>&1; then
    echo "Costruisci prima il runtime: bash scripts/run_ros2_system.sh build" >&2
    exit 1
fi
options=(--rm --init --network host --shm-size=2g
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp/surgical-home -e PYTHONUNBUFFERED=1
    -e MPLCONFIGDIR=/tmp/surgical-matplotlib -e XDG_CACHE_HOME=/tmp/surgical-cache
    -e HF_HOME=/workspace/.cache/huggingface
    -v "$PROJECT_DIR:/workspace")
if [[ -d /mnt/wslg ]]; then
    options+=(-e "DISPLAY=${DISPLAY:-:0}" -e "WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}"
        -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir -v /mnt/wslg:/mnt/wslg
        -v /tmp/.X11-unix:/tmp/.X11-unix)
elif [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
    options+=(-e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix)
    if [[ -f "${XAUTHORITY:-$HOME/.Xauthority}" ]]; then
        options+=(-v "${XAUTHORITY:-$HOME/.Xauthority}:/tmp/surgical-xauthority:ro" -e XAUTHORITY=/tmp/surgical-xauthority)
    fi
fi
if [[ -n "${ROS_DOMAIN_ID:-}" ]]; then options+=(-e "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"); fi
if [[ "${1:-}" == "prepare" ]]; then
    exec docker run "${options[@]}" "$ROS2_IMAGE" python3 -m pipeline.ros2.prepare_assets
elif [[ "${1:-}" == "self-test" ]]; then
    exec docker run "${options[@]}" "$ROS2_IMAGE" python3 -m unittest tests.test_ros2_contracts tests.test_ros2_transport -v
else
    exec docker run "${options[@]}" "$ROS2_IMAGE" python3 run_system_ros2.py "$@"
fi
