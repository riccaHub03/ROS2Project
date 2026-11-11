# =============================================================================
# Ambiente ROS 2 Humble + pipeline Python (PyBullet, SAM 3, GraspGen client)
#
# Build (dalla radice del repository):
#   docker build -f docker/Dockerfile -t surgical-robotics .
# oppure: bash scripts/docker_shell.sh build
# =============================================================================
FROM osrf/ros:humble-desktop-full

SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------
# 1. Dipendenze di sistema (APT)
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    ros-humble-launch-ros \
    ros-humble-ament-index-python \
    ros-humble-std-msgs \
    ros-humble-sensor-msgs \
    ros-humble-moveit \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    python3-pyqt5 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# 2. Dipendenze Python (pip)
# ---------------------------------------------------------
# PyTorch CPU-only
RUN pip3 install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Dipendenze dai package (sam3_ros, transformers, ecc.)
RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    Pillow==12.1.1 \
    transformers==5.2.0 \
    "scipy>=1.11"

# Dipendenze della pipeline (pybullet, trimesh, matplotlib, PyYAML, opencv, huggingface_hub...)
COPY requirements.txt requirements-graspgen-client.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------
# 3. Setup Ambiente ROS e Workspace
# ---------------------------------------------------------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
WORKDIR /workspace

