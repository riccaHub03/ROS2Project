# 🦾 3D Surgical Perception & Grasping Simulation Pipeline

Pipeline modulare per la simulazione 3D della percezione e pianificazione di presa (grasping) di strumenti chirurgici mediante **PyBullet**, **Segment Anything 3 (SAM 3)**, **LLM Task Planning (Ollama)** e de-proiezione di nuvole di punti 3D.

Avvio grafico nativo sul Mac: `bash run_graphical.sh`, oppure doppio clic su
`Avvia_simulazione.command`. L'avvio usa SAM3 e mostra la geometria e il KUKA;
`bash run_graphical.sh --vision ground-truth` esegue il test senza AI. Dettagli,
controlli e limiti del movimento dimostrativo sono in [GRAFICA.md](GRAFICA.md).
La posa della base KUKA puo' essere traslata e ruotata rispetto al mondo nel
file YAML; `--config percorso.yaml` permette di confrontare configurazioni
diverse mantenendo fissi tavolo, vassoi e camera.

Lo stadio `grasp` supporta `--grasp-backend pca` (predefinito) e
`--grasp-backend graspgen`, anche negli avvii reale e grafico. GraspGen usa il
client ufficiale leggero sul Mac e il server ufficiale su un PC NVIDIA separato.
Setup, test, checkpoint e ordine delle prove sono in [GRASPGEN.md](GRASPGEN.md).
Con la pinza dimostrativa attuale usare `--no-motion`: le pose Robotiq 2F-140
si possono visualizzare, ma il movimento GraspGen viene bloccato.

---

## 📋 Architettura della Pipeline (`run_pipeline_sim.py`)

La catena di montaggio in simulazione esegue 6 stadi. Ogni stadio è un modulo indipendente
(`python -m <modulo> --help`) che **legge i report JSON degli stadi precedenti** e **scrive un
proprio report JSON** (più gli artefatti) in `data/outputs/` (o `--out-dir` / `$PIPELINE_OUTPUT_DIR`).

```
[Istruzione Clinica]
        │
        ▼
plan            python -m pipeline.plan              ► plan.json
        │
        ▼
capture         python -m pipeline.sim.capture       ► sim_capture.json
        │                                              (+ sim_capture_rgb.png, sim_capture_depth.npy)
        ▼
segment (SAM 3) python -m pipeline.sim.segment       ► sim_segmentation.json
        │                                              (+ sim_segmentation_mask.png, _overlay.png)
        ▼
reference_pose  python -m pipeline.sim.reference_pose► sim_reference_pose.json
        │                                              (+ sim_reference_pose.png)
        ▼
deproject       python -m pipeline.sim.deproject     ► sim_deprojection.json
        │                                              (+ sim_deprojection_cloud.ply, _frames.png)
        ▼
grasp           python -m pipeline.grasp             ► grasp.json
                                                       (+ grasp_zones.ply, grasp_overview.png)
```

La pipeline reale (`run_pipeline_real.py`) usa `pipeline.real.{capture,segment,deproject}` e
produce `real_capture.json`, `real_segmentation.json`, `real_deprojection.json`, poi `grasp.json`.

### Contratto dei report (schema v2)

```json
{
  "schema_version": 2, "stage": "sim_segmentation", "status": "COMPLETED | FAILED",
  "timestamp": "...", "duration_s": 1.2,
  "inputs":    {"plan": ".../plan.json", "capture": ".../sim_capture.json"},
  "artifacts": {"mask_png": "...", "overlay_png": "..."},
  "data":      { "...payload dello stadio..." },
  "warnings":  [], "error": null
}
```

Se uno stadio fallisce scrive comunque il proprio report con `status: FAILED` e `error`
(tipo, messaggio, traceback) e termina con exit code 1: il JSON indica subito dove si è rotto.
Gli stadi a valle rifiutano report `FAILED` o di schema diverso.

### Esecuzione e debug stadio per stadio

```bash
python run_pipeline_sim.py --tool Scalpel                 # intera catena
python run_pipeline_sim.py --tool Scalpel --from-stage segment --stop-after deproject
python -m pipeline.sim.segment --backend ground-truth     # un solo stadio, input dai report su disco
python -m pipeline.grasp --deprojection data/outputs/real_deprojection.json
```

### Pulizia e test degli stadi singoli

```bash
python -m pipeline.clean            # anteprima dei file generati in data/outputs
python -m pipeline.clean --yes      # cancella (.gitkeep e README.md restano)
python -m unittest tests.test_standalone -v   # ogni stadio da solo, da cartella vuota
```

Tutti i report (e gli artefatti) vengono scritti in `data/outputs/` (o `--out-dir`); `--report NOME.json`
è un nome di file creato dentro quella cartella.

Codice condiviso in `pipeline/common/` (`paths`, `report`, `cli`, `scene`, `vision`, `reconstruction`, `ply`).
Test: `python -m unittest discover -s tests`.

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

### Passo 3: Controllo Interattivo W-S-A-D e Puntatore del Mouse con Vista Telecamera

Per pilotare il robot in tempo reale con i comandi prioritari da tastiera (**W-S-A-D**) e la modalità puntatore del mouse (attivabile premendo la rotella):

```bash
# Avvio rapido
bash run_interactive.sh

# Oppure tramite python diretto (con strumento e vassoio a scelta)
python3 interactive_controller.py --tool "Scissors" --tray "tray_2" --view "camera"
```

- **Comandi Prioritari da Tastiera (W-S-A-D)**:
  - `W` / `S` (o `↑` / `↓`): Movimento Avanti ($+X$) / Indietro ($-X$)
  - `A` / `D` (o `←` / `→`): Movimento Sinistra ($+Y$ verso vassoio 1) / Destra ($-Y$ verso vassoio 2)
  - `Q` / `E` (o `PgUp` / `PgDn`): Quota verticale $Z$ Su / Giù
  - **`Shift` + [Tasto]**: **Modalità TURBO** (scatto veloce 2.2x per spostamenti rapidi)
  - `+` / `-`: Regola la velocità di base (da 80 a 1200 mm/s)
  - `U` / `O` (o `J` / `L`): Ruota l'orientamento $Yaw$ della pinza
  - `Spazio`: Apri / Chiudi pinza
  - `G` (o Click Destro): Presa / Rilascio dello strumento chirurgico
  - `F`: Mostra / Nascondi finestra separata **Monitor Chirurgico Endoscopico (TCP)**
  - `H`: Reset immediato alla posa iniziale `HOME`
  - `V` o `C`: Alterna tra la vista telecamera di progetto e la vista 3D libera
  - `T`: Centra l'inquadratura tra Vassoio 1 e Vassoio 2
- **Modalità Puntatore del Mouse**:
  - Premi la **Rotella del Mouse** (o il tasto `M`) per attivare/disattivare l'inseguimento del cursore sul vassoio.
  - In modalità puntatore, un cursore ciano mostra il punto target e il click sinistro esegue il Pick & Place automatico sul punto puntato.
  - Premendo un qualsiasi tasto `W-S-A-D` il controllo torna istantaneamente alla tastiera.
- **Finestra Monitor Chirurgico Endoscopico (TCP)**:
  - Seconda finestra in tempo reale montata sulla flangia della pinza con mirino centrale, telemetro laser distanza e indicatori di stato.
  - Prestazioni ottimizzate a 60 FPS con accelerazione hardware OpenGL, caching delle maschere e upscaling nitido anti-aliased.


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
│   ├── plan.py                     # Stadio plan: LLM Planner (Ollama) con fallback deterministico
│   ├── grasp.py                    # Stadio grasp: assi PCA, zone funzionali, posa di presa
│   ├── orchestrate.py              # Runner sequenziale (--from-stage / --stop-after)
│   ├── common/                     # paths, report (schema v2), cli, scene, vision (SAM 3), reconstruction, ply
│   ├── sim/
│   │   ├── capture.py              # Scena PyBullet e render RGB-D
│   │   ├── segment.py              # SAM 3 (o ground-truth diagnostico) sulla vista camera
│   │   ├── reference_pose.py       # Posa STL certificata nel vassoio (PCA)
│   │   ├── deproject.py            # De-proiezione nuvola di punti e frame
│   │   ├── pipeline.py             # Catena simulata completa
│   │   └── graphical.py, robot_scene.py, robot_motion_preview.py   # Run con KUKA (--gui/--headless)
│   └── real/                       # capture.py, segment.py, deproject.py, pipeline.py
├── scripts/
│   ├── build_instruments_catalog.py    # Indicizzatore catalogo strumenti 3D
│   └── generate_surgical_meshes.py     # Generatore procedurale di mesh STL di test
├── requirements.txt                # Dipendenze Python
├── run_pipeline_sim.py             # Entrypoint pipeline simulata (anche: python -m pipeline.sim.pipeline)
├── run_pipeline_real.py            # Entrypoint pipeline su immagine reale
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
