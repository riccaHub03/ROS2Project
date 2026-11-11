# Build the official base first. This overlay fixes serving paths and installs
# missing runtime dependencies without editing or installing INTO the submodule.
FROM graspgen:1.0
RUN python -m pip install 'numpy==1.26.4' 'setuptools<78' \
    'viser==0.2.23' 'pyzmq==27.1.0' 'msgpack==1.1.2' 'msgpack-numpy==0.4.8' \
    'tensordict==0.3.2' 'transformers==4.46.3' 'huggingface-hub==0.25.2'
ENV PYTHONPATH=/code:/integration
ENV MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /code
# Both code trees will be mounted read-only; compiled pointnet2 is in the base.
CMD ["python", "-m", "pipeline.graspgen.server", "--code", "/integration/third_party/GraspGen", "--gripper-config", "/models/checkpoints/graspgen_robotiq_2f_140.yml", "--host", "0.0.0.0"]
