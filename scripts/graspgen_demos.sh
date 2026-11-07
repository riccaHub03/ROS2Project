#!/usr/bin/env bash
# Utility per usare NVlabs/GraspGen (submodule in third_party/GraspGen) in modo separato dalla pipeline.
# Richiede GPU NVIDIA + driver CUDA + Docker con nvidia-container-toolkit.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ROOT/third_party/GraspGen"
MODELS="$ROOT/data/external/GraspGenModels"
GRASP_DATASET="$ROOT/data/external/PhysicalAI-Robotics-GraspGen"
OBJECT_DATASET="$ROOT/data/external/graspgen_objects"
RESULTS="$ROOT/data/external/graspgen_results"

usage() {
    cat <<USAGE
Uso: $0 <comando>

  init       inizializza il submodule third_party/GraspGen
  models     scarica checkpoint + sample_data (Hugging Face, serve git-lfs)
  dataset    scarica il dataset di prese (grande, solo per training)
  build      costruisce l'immagine Docker di GraspGen (lungo)
  shell      apre il container di inferenza (code su /code, modelli su /models)
  train-shell apre il container con dataset e risultati (training)
  demo-pc    demo: prese da nuvola di punti di un oggetto (Robotiq 2F-140)
  demo-mesh  demo: prese da mesh (box.obj di esempio)
  demo-scene demo: prese da nuvola di punti di scena
  demo-collision  demo con filtro collisioni (Franka Panda)

Le demo vanno lanciate DENTRO il container ('shell'); qui stampo il comando da incollare.
Gripper disponibili in \$MODELS/checkpoints: graspgen_franka_panda.yml, graspgen_robotiq_2f_140.yml, suction.
USAGE
    exit 1
}

[[ $# -ge 1 ]] || usage
case "$1" in
    init)    git -C "$ROOT" submodule update --init third_party/GraspGen ;;
    models)
        mkdir -p "$ROOT/data/external"
        git lfs install
        [[ -d "$MODELS" ]] || git clone https://huggingface.co/adithyamurali/GraspGenModels "$MODELS"
        ;;
    dataset)
        git lfs install
        [[ -d "$GRASP_DATASET" ]] || git clone https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GraspGen "$GRASP_DATASET"
        ;;
    build)   (cd "$CODE" && bash docker/build.sh) ;;
    shell)   (cd "$CODE" && bash docker/run.sh "$CODE" --models "$MODELS") ;;
    train-shell)
        mkdir -p "$OBJECT_DATASET" "$RESULTS"
        (cd "$CODE" && bash docker/run.sh "$CODE" --grasp_dataset "$GRASP_DATASET" --object_dataset "$OBJECT_DATASET" --results "$RESULTS")
        ;;
    demo-pc)
        echo "cd /code && python scripts/demo_object_pc.py --sample_data_dir /models/sample_data/real_object_pc --gripper_config /models/checkpoints/graspgen_robotiq_2f_140.yml" ;;
    demo-mesh)
        echo "cd /code && python scripts/demo_object_mesh.py --mesh_file /models/sample_data/meshes/box.obj --mesh_scale 1.0 --gripper_config /models/checkpoints/graspgen_robotiq_2f_140.yml" ;;
    demo-scene)
        echo "cd /code && python scripts/demo_scene_pc.py --sample_data_dir /models/sample_data/real_scene_pc --gripper_config /models/checkpoints/graspgen_robotiq_2f_140.yml" ;;
    demo-collision)
        echo "cd /code && python scripts/demo_scene_pc.py --filter_collisions --sample_data_dir /models/sample_data/real_scene_pc --gripper_config /models/checkpoints/graspgen_franka_panda.yml" ;;
    *) usage ;;
esac
