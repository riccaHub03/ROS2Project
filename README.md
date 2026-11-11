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
python3 scripts/generate_surgical_meshes.py          # mesh STL di esempio in models/surgical_instruments/
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

## Riferimento dei comandi

Ogni comando ha `--help` con l'elenco completo delle opzioni (`python3 -m pipeline.grasp --help`,
`python3 run_pipeline_sim.py --help`, ...). Tutti i comandi sono da lanciare dalla radice del repository (nel
container: `/workspace`). Gli stadi sono moduli: si lanciano con `python3 -m <modulo>`.

### Dati in ingresso

| Cosa | Dove | Note |
|---|---|---|
| Mesh STL/OBJ di test | `models/surgical_instruments/` | create da `python3 scripts/generate_surgical_meshes.py` (forbici e bisturi). Anche `data/inputs/` viene cercata |
| Dataset 3D reale (3D-COSI, 232 modelli) | `data/inputs/Surgical Instruments/` | poi `python3 scripts/build_instruments_catalog.py` scrive `data/inputs/instruments_catalog.json`; `run_pipeline_sim.py --list-tools` mostra gli strumenti |
| Foto reale del vassoio | `data/inputs/surgical_scene.jpg` (o `--image`) | solo pipeline reale |
| Depth allineata alla foto | `--depth` (`.npy` in metri o PNG 16 bit in mm) | senza depth la presa reale **non** è abilitata al movimento |
| Configurazione | `config/surgical_environment.yaml` (o `--config`) | scena, vassoi, camera, quote, velocità, IK differenziale |
| Configurazione GraspGen | `config/graspgen.yaml` (o `--graspgen-config`) | host, porta, gripper |

Variabili d'ambiente: `PIPELINE_OUTPUT_DIR` (cartella dei report, default `data/outputs`) e `PIPELINE_ROOT`
(radice dei dati, usata dai test).

### Opzioni comuni a tutti gli stadi

| Opzione | Significato |
|---|---|
| `--out-dir DIR` | cartella di report e artefatti (default `$PIPELINE_OUTPUT_DIR` o `data/outputs`) |
| `--report NOME.json` | nome del report scritto da questo stadio, creato dentro `--out-dir` |
| `--config FILE` | `surgical_environment.yaml` alternativo |
| `-v`, `--verbose` | log di debug |

### Pipeline simulata

```
plan → capture → segment → reference_pose → deproject → grasp
```

**Intera catena** con `run_pipeline_sim.py`:

```bash
python3 run_pipeline_sim.py --tool Scalpel --offline --vision ground-truth         # senza AI né finestra
python3 run_pipeline_sim.py --instruction "Passami il bisturi dal vassoio 2 al vassoio 1"
python3 run_pipeline_sim.py --tool Scissors --headless --vision ground-truth --exit-after-run   # col robot, senza finestra
python3 run_pipeline_sim.py --tool Scissors --gui --vision sam3                    # finestra PyBullet + KUKA
python3 run_pipeline_sim.py --tool Scalpel --from-stage deproject                  # riparte da uno stadio
python3 run_pipeline_sim.py --tool Scalpel --stop-after segment                    # si ferma dopo uno stadio
python3 run_pipeline_sim.py --list-tools                                           # catalogo strumenti
```

| Opzione | Significato |
|---|---|
| `--instruction TESTO` | comando clinico in linguaggio naturale (LLM, o parser con `--offline`) |
| `--tool NOME` | strumento (es. `Scalpel`, `Scissors`, `Bone Forceps`); ha la precedenza sul testo |
| `--list-tools` | stampa il catalogo ed esce |
| `--mode all\|single` | `all` (default): RGB-D e maschera di **tutti** gli strumenti; `single`: solo lo STL pianificato. Senza robot con `--vision ground-truth` il default è `single` |
| `--offline` | niente Ollama: parser deterministico |
| `--seed N` | posizione dello strumento nel vassoio (riproducibilità) |
| `--from-stage S`, `--stop-after S` | `plan`, `capture`, `segment`, `reference_pose`, `deproject`, `grasp`; gli stadi saltati devono avere il report su disco |
| `--vision sam3\|ground-truth` | SAM 3, oppure la maschera diagnostica del simulatore (niente GPU né pesi) |
| `--gui` / `--headless` | scena col KUKA e anteprima del trasferimento, con o senza finestra |
| `--no-motion` | mostra la presa senza muovere il robot (obbligatorio con GraspGen e la pinza attuale) |
| `--exit-after-run` | chiude la finestra a fine run |
| `--grasp-backend pca\|graspgen` | geometria PCA (default) o inferenza GraspGen remota |
| `--graspgen-config/-host/-port/-timeout-ms` | parametri del client GraspGen |
| `--config`, `--out-dir`, `-v` | vedi opzioni comuni |

Nella finestra: `P` pausa, `R` ripete il movimento, `Q`/`Esc` chiude.

**Stadi uno alla volta**: ognuno legge i report precedenti da `--out-dir` e scrive il proprio.

| Stadio | Comando | Legge | Scrive | Opzioni specifiche |
|---|---|---|---|---|
| plan | `python3 -m pipeline.plan --instruction "..." --offline` | – | `plan.json` | `--instruction`, `--tool`, `--offline`, `--ollama-url`, `--model` |
| capture | `python3 -m pipeline.sim.capture --mode single` | `plan.json` | `sim_capture.json`, `sim_capture_rgb.png`, `sim_capture_depth.npy`, `sim_capture_depth_vis.png`, `sim_capture_truth_mask.png` | `--plan`, `--mode all\|single` |
| segment | `python3 -m pipeline.sim.segment --backend ground-truth` | `plan.json`, `sim_capture.json` | `sim_segmentation.json`, `sim_segmentation_mask.png`, `sim_segmentation_overlay.png` (+ `_all_mask`, `_all_overlay` con `--mode all`) | `--plan`, `--capture`, `--backend sam3\|ground-truth`, `--no-all-instruments` |
| reference_pose | `python3 -m pipeline.sim.reference_pose --tool Scalpel --seed 42` | `plan.json` | `sim_reference_pose.json`, `sim_reference_pose.png` | `--plan`, `--tool`, `--seed` |
| deproject | `python3 -m pipeline.sim.deproject` | `sim_segmentation.json`, `sim_capture.json`, `sim_reference_pose.json` | `sim_deprojection.json`, `sim_deprojection_cloud.ply`, `sim_deprojection_frames.png` | `--segmentation`, `--capture`, `--reference-pose` |
| grasp | `python3 -m pipeline.grasp` | `sim_deprojection.json`, `plan.json` | `grasp.json`, `grasp_pca.json` (o `grasp_graspgen.json`), `grasp_zones.ply`, `grasp_overview.png`, `grasp_candidates.png` | `--deprojection`, `--plan`, `--grasp-backend`, `--graspgen-*` |
| overview | `python3 -m pipeline.sim.overview --backend ground-truth` | `plan.json` | `sim_overview.json` e le immagini della collezione intera | `--plan`, `--backend` |

Con `--gui`/`--headless` si aggiungono `graphical_run.json` (manifest della run), `sim_robot_scene.png`,
`sim_robot_grasp.png`, `sim_robot_result.png` e `sim_robot_motion.json` (movimento misurato, errori per fase,
riepilogo dell'IK differenziale se attivo).

### Pipeline su immagine reale

```
plan → capture → segment → deproject → grasp
```

```bash
python3 run_pipeline_real.py --instruction "Passami le forbici" --image foto.png --depth depth.npy \
    --intrinsics 615 615 320 240
```

| Opzione | Significato |
|---|---|
| `--image FILE` | foto del vassoio con tutti gli strumenti |
| `--depth FILE` | depth metrica allineata (`.npy` in m o PNG 16 bit in mm); senza, piano costante nominale e presa non abilitata |
| `--intrinsics FX FY CX CY` | intrinseche calibrate, in pixel di `--image` |
| `--min-confidence X` | soglia SAM 3 (default 0.3) |
| `--instruction`, `--from-stage`, `--stop-after`, `--grasp-backend`, `--graspgen-*`, `--config`, `--out-dir`, `-v` | come nella pipeline simulata (gli stadi sono `plan`, `capture`, `segment`, `deproject`, `grasp`) |

Stadi singoli: `python3 -m pipeline.real.capture | segment | deproject`. Scrivono `real_capture.json`,
`real_segmentation.json` (+ maschera e overlay), `real_deprojection.json` (+ nuvola `.ply`) e poi `grasp.json`:

```bash
python3 -m pipeline.real.segment --min-confidence 0.4
python3 -m pipeline.grasp --deprojection data/outputs/real_deprojection.json
```

### GraspGen (server GPU separato)

Il client è nell'immagine; il server gira su un PC Linux con GPU NVIDIA. Procedura completa in
[docs/graspgen.md](docs/graspgen.md). Comandi di supporto:

```bash
python3 -m pipeline.graspgen.prepare --gpu --models DIR          # preflight di driver, Docker e pesi
python3 -m pipeline.graspgen.server --gripper-config CFG.yml --host 0.0.0.0 --port 5556   # sul PC GPU
python3 -m pipeline.graspgen.bundle --deprojection data/outputs/sim_deprojection.json --destination bundle/
python3 -m pipeline.graspgen.import_cloud --input nuvola.ply --out-dir data/outputs --units m --frame base
python3 run_pipeline_sim.py --tool Scissors --offline --vision ground-truth --grasp-backend graspgen --no-motion
```

### Controllo manuale del robot

```bash
bash scripts/run_interactive.sh
python3 interactive_controller.py --tool Scissors --tray tray_2 --view camera --speed 0.35
```

| Opzione | Significato |
|---|---|
| `--tool NOME`, `--tray tray_1\|tray_2` | strumento e vassoio di partenza |
| `--view camera\|orbit\|wrist\|surgeon` | vista iniziale |
| `--speed M/S`, `--turbo X` | velocità di jog (default 0.35) e moltiplicatore con `Shift` (default 2.2) |
| `--no-monitor` | senza la finestra separata dell'endoscopio |
| `--mouse-pointer` | parte in modalità puntatore (default: tastiera) |
| `--seed N`, `--config FILE` | posizione dello strumento, configurazione |
| `--headless`, `--duration S` | senza finestra e con uscita automatica (test) |

Tasti: `W/S/A/D` muovono in XY, `Q/E` quota, `U/O` rotazione, `Spazio` pinza, `G` presa/rilascio, `F` monitor
endoscopico, `H` posa HOME, `V`/`C` cambio vista, `M` o rotella del mouse puntatore. Elenco completo in
[docs/simulazione-grafica.md](docs/simulazione-grafica.md).

### ROS 2

```bash
bash scripts/ros2_build.sh                                   # compila ros2/surgical_executor (--all: anche i repo esterni)
ros2 launch surgical_executor pick_and_place.launch.py destination_tray:=tray_1
ros2 run surgical_executor pick_and_place --ros-args -p grasp_report:=data/outputs/grasp.json
```

Legge `grasp.json`, **rifiuta** il movimento se non è `VALIDATED_...` e scrive `robot_trajectory.json`
(e il video). Parametri e limiti in [docs/ros2.md](docs/ros2.md).

### Utility e test

```bash
python3 -m pipeline.clean            # anteprima dei file generati in data/outputs
python3 -m pipeline.clean --yes      # cancella (restano .gitkeep e README.md)
python3 scripts/generate_surgical_meshes.py        # mesh di test in models/surgical_instruments/
python3 scripts/build_instruments_catalog.py       # indice del dataset 3D-COSI in data/inputs/
python3 scripts/render_layout_check.py             # controllo visivo del layout: data/outputs/layout_check_render.png
python3 -m unittest discover -s tests              # tutti i test
python3 -m unittest tests.test_standalone -v       # ogni stadio da solo, da cartella vuota
```

Gli script di `scripts/` **non hanno `--help`**: si lanciano senza argomenti e scrivono subito i file indicati.

## Report JSON e file generati

Ogni stadio scrive un report con lo stesso schema (v2) in `data/outputs/`:

```json
{ "schema_version": 2, "stage": "sim_segmentation", "status": "COMPLETED | FAILED",
  "timestamp": "...", "duration_s": 1.2,
  "inputs":    {"plan": ".../plan.json", "capture": ".../sim_capture.json"},
  "artifacts": {"mask_png": "...", "overlay_png": "..."},
  "data":      { "...contenuto dello stadio..." },
  "warnings":  [], "error": null }
```

Se uno stadio fallisce scrive comunque il report con `status: FAILED` e `error` (tipo, messaggio, traceback) ed esce
con codice 1: gli stadi a valle rifiutano i report `FAILED` o di schema diverso. **Il primo posto dove guardare in
caso di problemi è il JSON dello stadio che si è fermato.**

| Report | Prodotto da | `data` contiene (campi principali) |
|---|---|---|
| `plan.json` | plan | `instruction`, `action`, `target_instrument`, `source_tray`, `destination_tray`, `planner_backend` |
| `sim_capture.json` | capture | `mode`, `simulated_instrument`, `mesh_file`, `ground_truth_pose`, `camera` |
| `sim_segmentation.json` | segment | `segmentation_backend`, `num_instances`, `confidence_score`, `bbox_2d_pixels`, `centroid_2d_pixels` |
| `sim_reference_pose.json` | reference_pose | `source_stl_file`, `stl_dimensions_m`, `pose`, `wall_clearances_m` |
| `sim_deprojection.json` | deproject | `num_points`, `frames`, `coordinate_frame_id`, `robot_base_pose_world`, `motion_safe` |
| `real_capture/segmentation/deprojection.json` | pipeline reale | stessi campi della controparte simulata |
| `grasp.json` | grasp | vedi sotto |
| `sim_robot_motion.json` | `--gui`/`--headless` | `status`, `waypoints`, `measured_phase_errors`, `differential_ik` |
| `robot_trajectory.json` | nodo ROS 2 | traiettoria dei giunti registrata |

**`grasp.json`** è il report che decide se il robot può muoversi:

- `readiness`: `VALIDATED_GEOMETRY_READY_FOR_TRAJECTORY_PLAN` (presa pronta) oppure `NOT_READY_FOR_ROBOT_MOTION`
  con i motivi in `readiness_blockers` (frame EE nominale, estrinseche non verificate, depth non misurata, punto fuori dalla zona raggiungibile, nessun candidato fattibile...).
- `designated_grasp_geometry`: posa scelta (`grasp_centroid_base_frame_xyz_m`, `recommended_gripper_orientation.quaternion_xyzw`, apertura, compatibilità con la pinza). Tutto nel riferimento della base del robot.
- `grasp_candidates` e `selected_candidate_id`: i candidati con l'esito di IK, limiti di giunto e collisioni.
- `functional_zones_segmentation`, `principal_axes_base_frame`: zone dello strumento e assi PCA.
- `grasp_backend`: `pca` o `graspgen`.

### Uso da Python

```python
from pipeline import plan, grasp
from pipeline.sim import capture, segment, reference_pose, deproject

plan.run("Passami il bisturi", offline=True)
capture.run(mode="single")
segment.run(backend="ground-truth")
reference_pose.run(tool="Scalpel")
deproject.run()
report = grasp.run()
print(report.data["designated_grasp_geometry"], report.artifacts)
```

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
