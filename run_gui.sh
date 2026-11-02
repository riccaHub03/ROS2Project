#!/usr/bin/env bash
# Script per avviare o entrare nel container ROS 2 con GUI WSLg abilitata
set -e

CONTAINER_NAME="container_ROS2_v3"
IMAGE_NAME="ambiente_robotica_v3"
WORKSPACE_DIR="/home/ricca/progetto_ROS"

# Se il container esiste già
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
    # Se non è in esecuzione, avvialo
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
        echo "Avvio del container ${CONTAINER_NAME}..."
        docker start "${CONTAINER_NAME}" >/dev/null
    fi
    echo "Accesso al container ${CONTAINER_NAME}..."
    docker exec -it "${CONTAINER_NAME}" bash
else
    echo "Creazione e avvio del container ${CONTAINER_NAME} con GUI abilitata..."
    docker run -it \
        --name "${CONTAINER_NAME}" \
        --network host \
        -e DISPLAY=:0 \
        -e WAYLAND_DISPLAY=wayland-0 \
        -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir \
        -e PULSE_SERVER=/mnt/wslg/PulseServer \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /mnt/wslg:/mnt/wslg \
        -v "${WORKSPACE_DIR}":/workspace \
        "${IMAGE_NAME}" bash
fi
