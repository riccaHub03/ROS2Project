# 🦾 3D Surgical Perception & Grasping Simulation Pipeline

Pipeline modulare per la simulazione 3D della percezione e pianificazione di presa (grasping) di strumenti chirurgici mediante **PyBullet**, **Segment Anything 3 (SAM 3)**, **LLM Task Planning (Ollama)** e de-proiezione di nuvole di punti 3D.

Avvio grafico nativo sul Mac: `bash run_graphical.sh`, oppure doppio clic su
`Avvia_simulazione.command`. L'avvio usa SAM3 e mostra la geometria e il KUKA;
`bash run_graphical.sh --vision ground-truth` esegue il test senza AI. Dettagli,
controlli e limiti del movimento dimostrativo sono in [GRAFICA.md](GRAFICA.md).
La posa della base KUKA puo' essere traslata e ruotata rispetto al mondo nel
file YAML; `--config percorso.yaml` permette di confrontare configurazioni
diverse mantenendo fissi tavolo, vassoi e camera.

---

## 📋 Architettura della Pipeline (`run_pipeline_sim.py`)

La catena di montaggio in simulazione esegue sequenzialmente 5 fasi:

```
[Istruzione Clinica]
        │
        ▼
1. LLM Task Planner (Ollama Qwen 2.5) ──────► data/outputs/stage1_task_plan.json
        │
        ▼
2. PyBullet 3D Spawning & RGB-D Camera ────► data/outputs/sim_stage2_camera.json
        │                                    (+ sim_camera_rgb.png, sim_camera_depth.npy)
        ▼
3. SAM 3 Visual Segmentation ───────────────► data/outputs/sim_stage3_vision_mask.json
        │                                    (+ sim_camera_mask.png, sim_camera_overlay.png)
        ▼
3b. STL Tray Placement & PCA Analysis ─────► data/outputs/stage3_stl_placement.json
        │                                    (+ sim_stl_tray_placement.png)
        ▼
4. 3D Deprojection & Frame Transforms ─────► data/outputs/sim_stage4_transforms.json
   (Camera Frame -> Robot Base -> TCP)       (+ sim_reconstructed_instrument.ply)
        │
        ▼
5. Grasp Geometry & Functional Zones ──────► data/outputs/stage5_grasp_geometry.json
   (Segmentazione Anelli / Fusto / Lame)     (+ stage5_segmented_grasp_zones.ply)
```

---

## 🛠️ Requisiti e Installazione

### 1. Prerequisiti
- **Python**: 3.10 o 3.11
- **Ollama** (opzionale per lo Stage 1 LLM): modello `qwen2.5:3b` (`ollama run qwen2.5:3b`)

### 2. Installazione Dipendenze Python

```bash
# Crea un virtual environment (consigliato)
python3 -m venv venv
source venv/bin/activate

# Installa le dipendenze
pip install -r requirements.txt
```

---

## 🚀 Utilizzo Rapido

### Passo 1: Genera le Mesh Chirurgiche STL di Esempio (Zero Download Esterni)
Per testare immediatamente la pipeline senza dover scaricare gigabyte di dataset esterni:

```bash
python3 scripts/generate_surgical_meshes.py
```
Questo comando genererà le mesh parametriche (es. `surgical_scissors.stl`, `scalpel.stl`) in `data/inputs/`.

### Passo 2: Avvia la Pipeline di Simulazione

```bash
# Esempio 1: Test su Bisturi chirurgico (Single Tool Mode)
python3 run_pipeline_sim.py --tool "Scalpel"

# Esempio 2: Test su Forbici chirurgiche
python3 run_pipeline_sim.py --tool "Scissors"

# Esempio 3: Con comando clinico esplicito
python3 run_pipeline_sim.py --instruction "Passami il bisturi dal vassoio 2 al vassoio 1"

# Visualizza tutti gli strumenti disponibili a catalogo
python3 run_pipeline_sim.py --list-tools
```

---

## 📁 Struttura del Repository

```
├── config/
│   └── surgical_environment.yaml   # Configurazione tavoli, vassoi, estrinseci/intrinseci telecamera
├── data/
│   ├── inputs/                     # File STL/OBJ di input (ignorati da git, tracciati tramite .gitkeep)
│   └── outputs/                    # Output generati dalla simulazione (ignorati da git)
├── pipeline/
│   ├── config_loader.py            # Caricamento e fallback parametri ambiente YAML
│   ├── frame_transformer.py        # Matematica trasformazioni di coordinate (Camera -> Base -> TCP)
│   ├── mesh_loader.py              # Loader ed auto-scaling modelli STL/OBJ
│   ├── step1_llm_planner.py        # Stage 1: LLM Planner con interfaccia Ollama
│   ├── step5_grasp_geometry.py     # Stage 5: Calcolo assi PCA, segmentazione zone e presa
│   └── sim/
│       ├── sim_step2_camera_render.py   # Stage 2: Spawning scena PyBullet e render RGB-D
│       ├── sim_step3_sam3_camera.py     # Stage 3: Inferenza SAM 3 su feed telecamera
│       ├── sim_step3b_stl_placement.py  # Stage 3b: Calcolo PCA e bounding box vassoio
│       └── sim_step4_transforms.py      # Stage 4: De-proiezione nuvola di punti e frame
├── scripts/
│   ├── build_instruments_catalog.py    # Indicizzatore catalogo strumenti 3D
│   └── generate_surgical_meshes.py     # Generatore procedurale di mesh STL di test
├── requirements.txt                # Dipendenze Python
├── run_pipeline_sim.py             # Entrypoint principale della pipeline di simulazione
└── .gitignore                      # Esclusioni per build, dati pesanti e sub-repo
```

---

## 🔒 Policy di Esclusione Dati (`.gitignore`)

Il repository è configurato per **non tracciare**:
- Grandi archivi e mesh CAD/3D (`.obj`, `.stl`, `.ply`, `.fbx`)
- Dataset in input (`data/inputs/*`)
- Artefatti generati (`data/outputs/*`, log, video `.mp4`, immagini `.png`, `.npy`)
- Workspace esterni e sub-repository (`robot-visual-simulation-platform`, `ROS2_workspace`)
- Cartelle di build e installazione colcon/ROS (`build/`, `install/`, `log/`)
- Pesi e cache di modelli (`models/`, `.cache/`, checkpoint)
