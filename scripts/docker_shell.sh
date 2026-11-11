#!/usr/bin/env bash
# Apre (o crea) il container ROS 2 del progetto, con GUI via WSLg.
#
#   bash scripts/docker_shell.sh          # entra nel container (lo crea/avvia se serve)
#   bash scripts/docker_shell.sh build    # costruisce l'immagine da docker/Dockerfile
#
# Variabili opzionali: IMAGE_NAME (default surgical-robotics), CONTAINER_NAME (default surgical_ros2).
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-surgical-robotics}"
CONTAINER_NAME="${CONTAINER_NAME:-surgical_ros2}"

if [[ "${1:-}" == "build" ]]; then
    exec docker build -f "$PROJECT_DIR/docker/Dockerfile" -t "$IMAGE_NAME" "$PROJECT_DIR"
fi

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Immagine '$IMAGE_NAME' non trovata. Costruiscila con: bash scripts/docker_shell.sh build" >&2
    exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
        echo "Avvio del container ${CONTAINER_NAME}..."
        docker start "${CONTAINER_NAME}" >/dev/null
    fi
    echo "Accesso al container ${CONTAINER_NAME}..."
    exec docker exec -it "${CONTAINER_NAME}" bash
fi

echo "Creazione del container ${CONTAINER_NAME} (GUI WSLg abilitata, progetto montato in /workspace)..."
exec docker run -it \
    --name "${CONTAINER_NAME}" \
    --network host \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}" \
    -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir \
    -e PULSE_SERVER=/mnt/wslg/PulseServer \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/wslg:/mnt/wslg \
    -v "${PROJECT_DIR}":/workspace \
    "${IMAGE_NAME}" bash
