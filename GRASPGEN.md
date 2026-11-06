# GraspGen (NVlabs) - integrazione separata

[NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) e' incluso come **submodule** in
`third_party/GraspGen`. Per ora e' tenuto separato dalla pipeline: serve per provare i
tutorial e le demo con i loro dataset e le loro pinze (Franka Panda, Robotiq 2F-140, ventosa).

## Requisiti
- GPU NVIDIA con driver CUDA (non disponibile nel container ROS attuale: PyTorch e' solo CPU)
- Docker con `nvidia-container-toolkit`, `git-lfs`
- Ambiente separato: l'immagine Docker di GraspGen (non il container ROS2)

In alternativa, GraspGen offre un server ZMQ (`third_party/GraspGen/client-server/`) da far
girare su una macchina con GPU e da chiamare da qui.

## Uso
```bash
bash scripts/graspgen_demos.sh init     # submodule (dopo un clone: git submodule update --init)
bash scripts/graspgen_demos.sh models   # checkpoint + sample_data -> data/external/GraspGenModels
bash scripts/graspgen_demos.sh build    # immagine Docker
bash scripts/graspgen_demos.sh shell    # container: codice su /code, modelli su /models
bash scripts/graspgen_demos.sh demo-mesh  # stampa il comando da lanciare nel container
```
Dataset di training (grande): `dataset` e `train-shell`. I dati scaricati stanno in
`data/external/` (ignorato da git).

## Integrazione futura
Adattatore tra pipeline e GraspGen: nuvola di punti dello stage 4
(`sim_reconstructed_instrument.ply`) -> input GraspGen; prese in uscita -> formato stage 5.
La pinza del progetto e' custom: GraspGen supporta solo le tre pinze sopra
(per altre pinze vedi GraspGen-X o il training con dati propri).
