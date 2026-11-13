#!/usr/bin/env bash
# Model code stays in the pinned submodule. Heavy actions run ONLY on Linux/NVIDIA.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ROOT/third_party/GraspGen"
MODELS="${GRASPGEN_MODELS:-$ROOT/data/external/GraspGenModels}"
PYTHON="${PIPELINE_PYTHON:-$ROOT/.venv/bin/python}"
IMAGE="${GRASPGEN_IMAGE:-ros2project-graspgen:cuda121}"
PORT="${GRASPGEN_PORT:-5556}"
GRASP_DATASET="$ROOT/data/external/PhysicalAI-Robotics-GraspGen"
OBJECT_DATASET="$ROOT/data/external/graspgen_objects"
cd "$ROOT"

usage() {
    cat <<'USAGE'
Usage: bash scripts/graspgen_demos.sh COMMAND [extra official demo arguments]
  init            initialize the pinned submodule (no update --remote)
  check           Mac-safe source / path preflight
  check-gpu       Linux driver, Docker, git-lfs and NVIDIA passthrough test
  models          download checkpoints + sample_data on PC; validate LFS files
  check-models    validate local models and author point clouds, no download
  build           corrected official CUDA 12.1 base + our runtime overlay on PC
  smoke-gpu       torch CUDA, PointNet extension and official sampler import
  shell           GPU container; code, integration and models read-only
  demo-pc         official author-data demo, explicitly --planner diffusion
  demo-mesh       official box mesh demo, explicitly --planner diffusion
  demo-scene      official scene demo (diffusion in this pinned version)
  server          provenance wrapper around the official ZMQ server
  dataset         legacy training dataset download (large, optional)
  train-shell     legacy GPU shell with training dataset/result mounts
  demo-collision  legacy scene demo with the official collision filter (Robotiq)
Env: PIPELINE_PYTHON, GRASPGEN_MODELS (absolute), GRASPGEN_IMAGE, GRASPGEN_PORT.
No heavy action is needed for Mac tests. CUDA 12.1 targets pre-Blackwell GPUs.
USAGE
}

gpu_host() {
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
        echo 'This action requires a Linux x86_64 NVIDIA PC; Mac runs client/tests only.' >&2; exit 1;
    }
    command -v nvidia-smi >/dev/null
    nvidia-smi >/dev/null
}

container() {
    gpu_host
    mkdir -p "$ROOT/data/external/graspgen_results"
    local run_args=(--rm)
    if [[ -t 0 && -t 1 ]]; then run_args+=(-it); fi
    if [[ "${GRASPGEN_TRAINING:-0}" == 1 ]]; then
        mkdir -p "$GRASP_DATASET" "$OBJECT_DATASET"
        run_args+=(-v "$GRASP_DATASET:/grasp_dataset:ro" -v "$OBJECT_DATASET:/object_dataset:ro")
    fi
    docker run "${run_args[@]}" --gpus all --shm-size 8g \
        -p "127.0.0.1:$PORT:5556" -p 127.0.0.1:8080:8080 \
        -v "$CODE:/code:ro" -v "$ROOT:/integration:ro" -v "$MODELS:/models:ro" \
        -v "$ROOT/data/external/graspgen_results:/results" "$IMAGE" "$@"
}

[[ $# -ge 1 ]] || { usage; exit 1; }
action="$1"
shift
case "$action" in
    init) git submodule update --init --recursive third_party/GraspGen ;;
    check) "$PYTHON" -m pipeline.graspgen.prepare ;;
    check-gpu) gpu_host; "$PYTHON" -m pipeline.graspgen.prepare --gpu ;;
    check-models) "$PYTHON" -m pipeline.graspgen.prepare --models "$MODELS" ;;
    dataset)
        gpu_host
        mkdir -p "$ROOT/data/external"
        if [[ ! -d "$GRASP_DATASET/.git" ]]; then
            GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GraspGen "$GRASP_DATASET"
        fi
        git -C "$GRASP_DATASET" lfs install --local
        git -C "$GRASP_DATASET" lfs pull
        ;;
    train-shell) GRASPGEN_TRAINING=1 container bash ;;
    models)
        gpu_host
        mkdir -p "$(dirname "$MODELS")"
        if [[ ! -d "$MODELS/.git" ]]; then
            GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/adithyamurali/GraspGenModels "$MODELS"
        fi
        git -C "$MODELS" lfs install --local
        git -C "$MODELS" lfs pull
        "$PYTHON" -m pipeline.graspgen.prepare --models "$MODELS"
        git -C "$MODELS" rev-parse HEAD
        ;;
    build)
        gpu_host
        recipe="$(mktemp -t graspgen-base.XXXXXX)"
        trap 'rm -f "$recipe"' EXIT
        "$PYTHON" -m pipeline.graspgen.prepare --write-base-dockerfile "$recipe"
        docker build -f "$recipe" --progress=plain --network=host -t graspgen:1.0 "$CODE"
        docker build -f "$ROOT/docker/graspgen-runtime.Dockerfile" -t "$IMAGE" "$ROOT/docker"
        ;;
    smoke-gpu)
        container python -c 'import torch; assert torch.__version__.split("+")[0] == "2.1.0", torch.__version__; assert torch.version.cuda == "12.1", torch.version.cuda; import pointnet2_ops._ext; from grasp_gen.grasp_server import GraspGenSampler; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
        ;;
    shell) container bash ;;
    demo-pc)
        container python scripts/demo_object_pc.py --planner diffusion \
            --sample_data_dir /models/sample_data/real_object_pc \
            --gripper_config /models/checkpoints/graspgen_robotiq_2f_140.yml \
            --return_topk --topk_num_grasps 100 --grasp_threshold -1 "$@"
        ;;
    demo-mesh)
        container python scripts/demo_object_mesh.py --planner diffusion \
            --mesh_file /models/sample_data/meshes/box.obj --mesh_scale 1.0 \
            --gripper_config /models/checkpoints/graspgen_robotiq_2f_140.yml \
            --output_file /results/box_grasps.yml "$@"
        ;;
    demo-scene)
        container python scripts/demo_scene_pc.py --sample_data_dir /models/sample_data/real_scene_pc \
            --gripper_config /models/checkpoints/graspgen_robotiq_2f_140.yml "$@"
        ;;
    demo-collision)
        container python scripts/demo_scene_pc.py --filter_collisions \
            --sample_data_dir /models/sample_data/real_scene_pc \
            --gripper_config /models/checkpoints/graspgen_robotiq_2f_140.yml "$@"
        ;;
    server)
        "$PYTHON" -m pipeline.graspgen.prepare --models "$MODELS"
        container python -m pipeline.graspgen.server --code /integration/third_party/GraspGen \
            --gripper-config /models/checkpoints/graspgen_robotiq_2f_140.yml --host 0.0.0.0 --port 5556
        ;;
    -h|--help|help) usage ;;
    *) usage; exit 1 ;;
esac
