# Build the official base first. This overlay fixes serving paths and installs
# missing runtime dependencies without editing or installing INTO the submodule.
FROM graspgen:1.0
RUN python -m pip install 'torch==2.1.0' 'torchvision==0.16.0' 'numpy==1.26.4' 'setuptools<78' \
    'viser==0.2.23' 'pyzmq==27.1.0' 'msgpack==1.1.2' 'msgpack-numpy==0.4.8' \
    'tensordict==0.2.1' 'transformers==4.46.3' 'huggingface-hub==0.25.2'
ENV MAX_JOBS=1
ENV PYTHONPATH=/code:/integration
ENV MPLCONFIGDIR=/tmp/matplotlib
# Git in the base image needs this in global configuration for bind mounts
# owned by the host user. Trust only the integration checkout and its submodule.
RUN git config --global --add safe.directory /integration \
    && git config --global --add safe.directory /integration/third_party/GraspGen
WORKDIR /code
# Both code trees will be mounted read-only; compiled pointnet2 is in the base.
CMD ["python", "-m", "pipeline.graspgen.server", "--code", "/integration/third_party/GraspGen", "--gripper-config", "/models/checkpoints/graspgen_robotiq_2f_140.yml", "--host", "0.0.0.0"]
