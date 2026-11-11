# Pipeline di percezione e presa per strumenti chirurgici

Simulazione 3D della percezione e della pianificazione di presa di strumenti chirurgici con un
**KUKA LBR iiwa**: **PyBullet** per la scena, **SAM 3** per la segmentazione, un LLM locale (**Ollama**)
per interpretare la richiesta clinica, nuvole di punti 3D e geometria di presa (PCA oppure **GraspGen**).
L'ambiente di riferimento è **Docker su WSL2 (Ubuntu)** con ROS 2 Humble.

## Avvio rapido

Prerequisiti: Windows 11 con WSL2 e WSLg, Docker raggiungibile da WSL. Tutti i comandi sono da eseguire in
WSL, dalla radice del repository.

```bash
# 1. Clona il progetto con il submodule GraspGen
git clone --recurse-submodules <URL-del-repository> && cd <cartella-del-repository>
#    (se hai già clonato senza submodule: git submodule update --init)

# 2. Costruisci l'immagine (ROS 2 Humble + dipendenze Python)
bash scripts/docker_shell.sh build

# 3. Entra nel container: il repository è montato in /workspace
bash scripts/docker_shell.sh
```

Dentro il container:

```bash
python3 scripts/generate_surgical_meshes.py          # mesh STL di esempio in data/inputs/
python3 run_pipeline_sim.py --tool Scalpel --vision ground-truth --headless   # catena completa, senza AI né finestra
bash scripts/run_sim_gui.sh --vision ground-truth    # stessa catena con la finestra PyBullet del KUKA
bash scripts/run_sim_gui.sh                          # con SAM 3 (serve l'accesso ai pesi, vedi sotto)
bash scripts/run_interactive.sh                      # controllo manuale W-S-A-D con il mouse
```

I risultati (report JSON, immagini, nuvole di punti) finiscono in `data/outputs/`.

### Componenti opzionali
- **Ollama** (interpretazione della frase clinica): `ollama run qwen2.5:3b` sull'host; senza, usa `--offline`.
- **SAM 3**: modello da Hugging Face (`transformers`); in alternativa `--vision ground-truth`.
- **GraspGen**: server su un PC Linux con GPU NVIDIA, il client è già nell'immagine. Setup in [docs/graspgen.md](docs/graspgen.md).
- **ROS 2**: il nodo `surgical_executor` (in `ros2/`) esegue il movimento pianificato; i repository esterni
  (`llm_ros`, `sam3_ros`, `lbr_fri_ros2_stack`, `fri`) si scaricano con `ros2/workspace.repos`.
  Build e uso in [docs/ros2.md](docs/ros2.md).

## Come funziona

Sei stadi indipendenti, ognuno un modulo Python che legge i report JSON dei precedenti e scrive il proprio:

```
plan → capture → segment → reference_pose → deproject → grasp   (+ movimento KUKA dimostrativo)
```

Ogni stadio si lancia da solo (`python3 -m pipeline.sim.segment --help`) oppure in catena con
`run_pipeline_sim.py` (`--from-stage`, `--stop-after`). Dettagli, contratto dei report e pipeline su immagine
reale: [docs/pipeline.md](docs/pipeline.md).

## Documentazione

| Documento | Contenuto |
|---|---|
| [docs/pipeline.md](docs/pipeline.md) | Architettura degli stadi, schema dei report, esecuzione e debug |
| [docs/guida-comandi.md](docs/guida-comandi.md) | Comandi di ogni stadio, opzioni comuni, dataset reale |
| [docs/ros2.md](docs/ros2.md) | Package ROS 2, build con colcon, esecuzione e limiti sul robot reale |
| [docs/simulazione-grafica.md](docs/simulazione-grafica.md) | Finestra PyBullet, controlli, posa della base, controllo manuale |
| [docs/graspgen.md](docs/graspgen.md) | GraspGen: server GPU, client, ordine delle prove |
| [docs/graspgen-verification.md](docs/graspgen-verification.md) | Registro delle verifiche già eseguite |

## Struttura del repository

```
config/            surgical_environment.yaml: fonte unica di scena, vassoi, camera, velocità, tempi, IK differenziale (opzionale)
data/              inputs/ (STL, ignorati da git) e outputs/ (generati, ignorati da git)
docker/            Dockerfile, entrypoint.sh, runtime del server GraspGen
docs/              documentazione (vedi tabella)
pipeline/          codice: stadi (plan, grasp, sim/, real/), common/, graspgen/, motion/ (piano di pick & place)
ros2/              package ROS 2 surgical_executor e workspace.repos
scripts/           docker_shell.sh, run_sim_gui.sh, run_interactive.sh e strumenti di sviluppo
tests/             test unittest
third_party/       GraspGen (submodule git)
run_pipeline_sim.py, run_pipeline_real.py   entrypoint della pipeline simulata e su immagine reale
interactive_controller.py                    controllo manuale del robot in PyBullet
```

## Test

```bash
python3 -m unittest discover -s tests
```

I test di `test_graspgen.py` richiedono il submodule `third_party/GraspGen` inizializzato.

## Dati non versionati

Git ignora mesh e immagini pesanti (`*.stl`, `*.ply`, `*.png`, `*.npy`, `*.mp4`), `data/inputs`, `data/outputs`,
pesi dei modelli (`models/`, `*.pt`), `ROS2_workspace/` e le cartelle di build di colcon.
