ARG BASE_IMAGE=ros:humble-ros-base
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-colcon-common-extensions build-essential git ca-certificates \
    ros-humble-launch-ros ros-humble-ament-index-python \
    ros-humble-control-msgs ros-humble-trajectory-msgs ros-humble-tf2-ros \
    ros-humble-sensor-msgs ros-humble-std-srvs \
    libgl1 libgl1-mesa-dri libegl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libxi6 libxrandr2 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-ros2-runtime.txt requirements-graspgen-client.txt /tmp/
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-ros2-runtime.txt
# The core-only build runs the real DDS/action tests without loading any AI model.
# Production builds include the CPU SAM3 stack; GraspGen stays on its GPU server.
ARG INSTALL_AI=1
RUN if [[ "$INSTALL_AI" == "1" ]]; then \
      python3 -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
      python3 -m pip install --no-cache-dir transformers==5.2.0 huggingface_hub; \
    fi
COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
WORKDIR /workspace
CMD ["python3", "run_system_ros2.py", "--headless"]
