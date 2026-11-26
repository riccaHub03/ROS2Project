# Optional camera-enabled derivative of the ROS 2 system image.
# Build separately so the existing simulation image remains intact.
ARG BASE_IMAGE=surgical-ros2-system:latest
FROM ${BASE_IMAGE}
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-realsense2-camera \
    && rm -rf /var/lib/apt/lists/*
# The host camera venv provides pyrealsense2; the ROS driver uses librealsense.
