# Guida ai Comandi e Architettura Modulare

*3D Surgical Perception & Grasping Simulation Pipeline (PyBullet + SAM 3 + LLM)*

**ISTRUZIONE PRELIMINARE SULLA CARTELLA:** Tutti i comandi descritti in questo documento vanno eseguiti all'interno della cartella refactorizzata: `cd /workspace` (nel container Docker, dove il repository è montato).

## 1. Panoramica dell'Architettura Modulare

Il sistema è stato interamente disaccoppiato in stadi indipendenti e testabili. Ciascun modulo può essere eseguito singolarmente da riga di comando o importato in Python. Ogni modulo legge i report JSON dello stadio precedente e produce il proprio report JSON validato (Schema v2) insieme agli artefatti generati (PNG, NPY, PLY).

| Stadio | Nuovo Modulo Python | File Sostituito | Report e Artefatti Generati |
|---|---|---|---|
| 1. Plan | pipeline.plan | step1_llm_planner.py | plan.json |
| 2. Capture | pipeline.sim.capture | sim_step2_camera_render.py | sim_capture.json, rgb.png, depth.npy |
| 3. Segment | pipeline.sim.segment | sim_step3_sam3_camera.py | sim_segmentation.json, mask.png, overlay.png |
| 4. Reference Pose | pipeline.sim.reference_pose | sim_step3b_stl_placement.py | sim_reference_pose.json, tray_placement.png |
| 5. Deproject | pipeline.sim.deproject | sim_step4_transforms.py | sim_deprojection.json, cloud.ply, frames.png |
| 6. Grasp | pipeline.grasp | step5_grasp_geometry.py | grasp.json, grasp_zones.ply, overview.png |
| Extra. Overview | pipeline.sim.overview | (nuovo) | sim_overview.json, sim_overview_capture_rgb.png, sim_overview_capture_depth_vis.png, sim_overview_segmentation_all_mask.png, ..._all_overlay.png |

## 2. Opzioni Comuni a TUTTI i Moduli

Tutti i moduli condividono i seguenti parametri di controllo CLI:

| Flag CLI | Valore Predefinito | Descrizione Funzionale |
|---|---|---|
| -v, --verbose | False | Attiva i messaggi di log di debug dettagliati a terminale. |
| --out-dir | data/outputs/ | Cartella di output in cui vengono salvati i report JSON e gli artefatti. |
| --config | config/surgical_environment.yaml | Percorso del file YAML di configurazione della sala operatoria. |
| --report | `<nome_standard>.json` | Permette di sovrascrivere il nome del file JSON di report salvato. |
| --help | - | Mostra la guida a video con tutti i parametri disponibili per lo specifico stadio. |

## 3. Guida Completa all'Esecuzione Separata Stadio per Stadio

### Stadio 1: Pianificazione Clinica (LLM / Parser Offline)

Interpreta il comando clinico in linguaggio naturale o seleziona direttamente lo strumento chirurgico target, estraendo azione, vassoio sorgente e vassoio destinazione.

```bash
# 1. Con Ollama attivo (modello linguistico Qwen 2.5 3B):
python3 -m pipeline.plan --instruction "Passami il bisturi dal vassoio 2 al vassoio 1"

# 2. Modalità Deterministica / OFFLINE (rapida, senza richiedere Ollama):
python3 -m pipeline.plan --instruction "Passami le forbici" --offline

# 3. Forzando direttamente uno strumento clinico specifico dal catalogo:
python3 -m pipeline.plan --tool "Bone Forceps" --offline
```

**Input & Output Stadio 1:** Input: Testo naturale o flag --tool. Output: data/outputs/plan.json contenente l'azione e lo strumento target.

### Stadio 2: Rendering PyBullet & Telecamera RGB-D

Inizializza la scena di simulazione PyBullet con tavolo operatorio, vassoi inox e telecamera overhead Nadir. Spawna lo strumento e acquisisce immagine a colori e profondità metrica.

```bash
# 1. Modalità Singolo Strumento (legge il target da data/outputs/plan.json):
python3 -m pipeline.sim.capture --mode single

# 2. Modalità Scena Completa (tutti gli strumenti insieme): è il predefinito della pipeline completa
python3 -m pipeline.sim.capture --mode all

# 3. Specificando esplicitamente un file di piano personalizzato:
python3 -m pipeline.sim.capture --plan data/outputs/plan.json --mode single
```

**Input & Output Stadio 2:** Input: plan.json. Output: sim_capture.json, sim_capture_rgb.png (foto colore), sim_capture_depth.npy (matrice profondità metrica in metri).

### Stadio 3: Segmentazione Visiva 2D

Isola i pixel dello strumento target nell'immagine RGB della telecamera, generando la maschera binaria di segmentazione. Di default produce anche la **maschera di tutti gli strumenti** (seconda inferenza SAM 3 con il prompt "surgical instrument"); `--no-all-instruments` la salta.

```bash
# 1. Modalità DIAGNOSTICA / Ground-Truth (istantanea, basata su PyBullet ID, zero GPU):
python3 -m pipeline.sim.segment --backend ground-truth

# 2. Modalità AI con Segment Anything 3 (SAM 3):
python3 -m pipeline.sim.segment --backend sam3

# 3. Solo il bersaglio, senza la maschera di tutti gli strumenti:
python3 -m pipeline.sim.segment --backend sam3 --no-all-instruments
```

**Input & Output Stadio 3:** Input: plan.json e sim_capture.json (+ rgb.png). Output: sim_segmentation.json, sim_segmentation_mask.png (maschera b/n), sim_segmentation_overlay.png (contorno verde sovrapposto) e, per tutti gli strumenti, sim_segmentation_all_mask.png, sim_segmentation_all_overlay.png (un colore per strumento, bersaglio in verde) e sim_segmentation_all_instances.png (etichette a 16 bit).

### Stadio 4: Posa di Riferimento STL dal Dataset Reale

Cerca e carica la mesh 3D reale dai tuoi 5 GB di dataset (3D-COSI), posiziona il modello nel vassoio con il seed specificato e ne calcola la posa e gli assi principali PCA ideali.

```bash
# 1. Legge lo strumento pianificato da plan.json:
python3 -m pipeline.sim.reference_pose

# 2. Testando direttamente uno strumento del catalogo con seed di orientamento:
python3 -m pipeline.sim.reference_pose --tool "Scalpel" --seed 42

# 3. Con strumento a scelta (es. Pinza ossea o Forbici):
python3 -m pipeline.sim.reference_pose --tool "Bone Forceps" --seed 123
```

**Input & Output Stadio 4:** Input: plan.json o flag --tool. Output: sim_reference_pose.json (posa cartesiana XYZ e quaternione), sim_reference_pose.png (rendering con bounding box).

### Stadio 5: De-proiezione 3D e Trasformazione Sistemi di Coordinate

De-proietta la maschera 2D nello spazio 3D reale usando la mappa di profondità metrica. Calcola le trasformazioni di frame: Camera Frame -> Robot Base Frame -> TCP.

```bash
# 1. Esecuzione standard (risolve automaticamente i percorsi dai report su disco):
python3 -m pipeline.sim.deproject

# 2. Passando percorsi personalizzati:
python3 -m pipeline.sim.deproject --capture data/outputs/sim_capture.json --segmentation data/outputs/sim_segmentation.json --reference-pose data/outputs/sim_reference_pose.json
```

**Input & Output Stadio 5:** Input: capture, segmentation e reference-pose JSON. Output: sim_deprojection.json, sim_deprojection_cloud.ply (nuvola di punti 3D reale), sim_deprojection_frames.png (grafico delle terne di coordinate).

### Stadio 6: Geometria di Presa e Zone Funzionali

Esegue l'analisi PCA sulla nuvola 3D de-proiettata, segmenta le zone funzionali (anelli/impugnatura, fusto centrale, lame/punte) e calcola la posa di presa ottimale e l'apertura pinza.

```bash
# 1. Esecuzione standard per simulazione (legge sim_deprojection.json):
python3 -m pipeline.grasp

# 2. Specificando esplicitamente il file di deproiezione:
python3 -m pipeline.grasp --deprojection data/outputs/sim_deprojection.json

# 3. Esecuzione su dati acquisiti da telecamera reale:
python3 -m pipeline.grasp --deprojection data/outputs/real_deprojection.json
```

**Input & Output Stadio 6:** Input: sim_deprojection.json e plan.json. Output: grasp.json (posa presa XYZ, quaternione, apertura mm), grasp_zones.ply (nuvola 3D colorata per zone), grasp_overview.png.

### Panoramica dell'intera collezione (Extra)

Rende fuori schermo l'intera collezione `All_Instruments` e ne salva RGB, depth e la maschera di tutti gli strumenti. È informativa: la presa si calcola sulla scena STL. La esegue da sola la run con robot (`--gui`/`--headless`) dopo che la presa è pronta.

```bash
python3 -m pipeline.sim.overview --backend ground-truth   # senza SAM 3
python3 -m pipeline.sim.overview --backend sam3
```

**Output:** sim_overview.json, sim_overview_capture_rgb.png, sim_overview_capture_depth_vis.png (e _depth.npy in metri), sim_overview_segmentation_all_mask.png, sim_overview_segmentation_all_overlay.png, sim_overview_segmentation_all_instances.png, più il bersaglio isolato (sim_overview_segmentation_mask.png / _overlay.png).

## 4. Orchestrazione a Blocchi ed Esecuzione Rapida

**Modo predefinito (`--mode all`).** La pipeline completa lavora sull'intera collezione: RGB-D e maschera di **tutti** gli strumenti, poi usa lo STL del bersaglio. Con `--gui`/`--headless` la finestra mostra sempre lo STL trasportato da PyBullet (mai la maschera né l'OBJ) e la collezione viene resa fuori schermo, dopo la presa, in `sim_overview.json`. `--mode single` ripristina il comportamento precedente (solo lo STL). Senza robot, `--vision ground-truth` usa `single` per default: la maschera del simulatore ha bisogno di un corpo per strumento e la collezione è un corpo unico.

Grazie al coordinatore pipeline.orchestrate, puoi eseguire sia l'intera pipeline che sotto-blocchi sequenziali tramite run_pipeline_sim.py:

```bash
# 1. Esecuzione INTERA pipeline con visualizzazione diagnostica rapida (Zero AI):
python3 run_pipeline_sim.py --tool Scalpel --vision ground-truth --headless

# 2. Esecuzione grafica nativa con simulatore KUKA LBR iiwa:
python3 run_pipeline_sim.py --tool Scalpel --gui --vision ground-truth

# 3. Esegui solo dall'inizio fino alla segmentazione visiva (--stop-after):
python3 run_pipeline_sim.py --tool Scalpel --stop-after segment

# 4. Rilancia solo la parte 3D e Grasping da uno stadio intermedio (--from-stage):
python3 run_pipeline_sim.py --tool Scalpel --from-stage deproject

# 5. Esegui solo l'intervallo capture -> reference_pose:
python3 run_pipeline_sim.py --from-stage capture --stop-after reference_pose
```

## 5. Utilizzo del Dataset 3D Reale da 5 GB (3D-COSI Collection)

Non è necessario generare mesh sintetiche con scripts/generate_surgical_meshes.py. Il catalogo data/inputs/instruments_catalog.json contiene già tutti i 232 strumenti reali pronti per essere utilizzati.

```bash
# 1. Visualizza il catalogo completo di tutte le 27 categorie cliniche e 232 modelli:
python3 run_pipeline_sim.py --list-tools

# 2. Esempi di avvio rapido con strumenti specifici del catalogo reale:
python3 run_pipeline_sim.py --tool "Bone Forceps" --gui --vision ground-truth
python3 run_pipeline_sim.py --tool "Bulldog Clamps" --gui --vision ground-truth
python3 run_pipeline_sim.py --tool "Needle Holders" --gui --vision ground-truth
python3 run_pipeline_sim.py --tool "Scalpel" --gui --vision ground-truth
python3 run_pipeline_sim.py --tool "Scissors" --gui --vision ground-truth
```

## 6. Controllo Manuale Interattivo (W-S-A-D e Puntatore Mouse)

È possibile pilotare in tempo reale il robot KUKA con la tastiera, utilizzare il raycasting del mouse per il puntamento sul vassoio e aprire la finestra del Monitor Chirurgico Endoscopico (TCP):

```bash
# Avvio rapido tramite script dedicato:
bash scripts/run_interactive.sh

# Oppure tramite python diretto con parametri a scelta:
python3 interactive_controller.py --tool "Scissors" --tray "tray_2" --view "camera"

# Posizione dello strumento riproducibile (senza --seed cambia a ogni avvio):
python3 interactive_controller.py --tool "Scalpel" --seed 42
```

Note sul controller:

- Lo strumento nasce in posizione e orientamento casuali dentro il vassoio, con l'ingombro interamente dentro il bordo e il centro nella zona raggiungibile dal KUKA (corona tra 0.45 e 0.74 m dalla base).
- Le scorciatoie native di PyBullet (wireframe, riquadri AABB, ecc.) sono disattivate, così W/A/S/D e G non alterano la grafica.
- Gli slider (quota, yaw, pinza) agiscono solo quando li muovi e non sovrascrivono la tastiera.
- Il robot non si muove verso punti non raggiungibili: il target viene riportato all'ultimo punto valido.

| Tasto / Controllo | Azione Eseguita |
|---|---|
| W / S (o Freccia Su/Giu) | Movimento Cartesiano XY Avanti (+X) / Indietro (-X) |
| A / D (o Freccia Sx/Dx) | Movimento Cartesiano XY Sinistra (+Y verso Vassoio 1) / Destra (-Y verso Vassoio 2) |
| Q / E (o PgUp/PgDn) | Quota Verticale Z: Alza (+Z) / Abbassa (-Z) la pinza |
| Shift + [Tasto] | Modalita TURBO: scatto veloce a velocita 2.2x |
| + / - | Regolazione fine della velocita di jog (da 80 a 1200 mm/s) |
| U / O (o J / L) | Rotazione Yaw della pinza attorno all'asse verticale |
| Barra Spazio | Apre e Chiude le dita della pinza |
| G (o Click Destro) | Afferra / Rilascia lo strumento chirurgico nel vassoio |
| Rotella Mouse (o Tasto M) | Attiva/Disattiva inseguimento cursore del mouse (Pointer Mode) |
| Click Sinistro (in Pointer) | Esegue il ciclo automatico Pick & Place sul punto puntato |
| Tasto F | Apre / Chiude la finestra secondaria Monitor Chirurgico Endoscopico (TCP) |
| Tasto H | Reset immediato del robot alla posa iniziale HOME |
| Tasto V o C | Alterna la visuale: Telecamera Nadir di Progetto vs Vista 3D Libera |
| Tasto T | Alterna il focus tra Vassoio 1 e Vassoio 2 (sposta il target e la vista) |

## 7. Utilizzo da Script Python (API Programmatica)

Per integrare i singoli stadi all'interno di un codice Python custom senza passare da riga di comando:

```python
from pipeline import plan, grasp
from pipeline.sim import capture, segment, reference_pose, deproject

# 1. Pianificazione dell'azione clinica
rep_plan = plan.run("Passami il bisturi", offline=True)

# 2. Simulazione PyBullet e acquisizione RGB-D
rep_cap = capture.run(mode="single")

# 3. Segmentazione visiva (ground-truth o sam3)
rep_seg = segment.run(backend="ground-truth")

# 4. Analisi PCA posa di riferimento STL
rep_ref = reference_pose.run(tool="Scalpel")

# 5. De-proiezione nuvola di punti e frame transform
rep_dep = deproject.run()

# 6. Sintesi posa di presa e zone funzionali
rep_grasp = grasp.run()

# Accesso diretto ai dati del report
print("Posa presa calcolata:", rep_grasp.data["grasp_pose"])
print("Artefatti generati:", rep_grasp.artifacts)
```

## 8. GraspGen (NVlabs)

Il modello ufficiale in `third_party/GraspGen` è collegato tramite client/server
allo stadio `grasp`. Usare `--grasp-backend graspgen` negli avvii sim, reale e
grafico; PCA resta predefinito. Per setup client/PC GPU, test, geometria e blocco del
movimento con la pinza dimostrativa vedere [graspgen.md](graspgen.md).

## 9. Pipeline reale: controlli di sicurezza

```bash
python3 run_pipeline_real.py --image foto.png --depth depth.npy \
    --intrinsics FX FY CX CY --min-confidence 0.30
```

- `--depth`: depth metrica allineata all'immagine a colori (`.npy` in metri o PNG 16 bit in mm).
  Senza, viene usato un piano costante sulla superficie del vassoio e la presa **non** è abilitata al movimento.
- `--intrinsics`: calibrazione reale della camera (in pixel dell'immagine). Senza, vengono scalati dal YAML (nominali).
- `camera_overhead.extrinsics.verified: true` nel YAML, solo dopo la calibrazione mano-occhio: senza, la presa reale non è abilitata al movimento.
- La segmentazione **fallisce** sotto la confidenza minima (default 0.30) o se la maschera copre più del 25% dell'immagine.
- Il piano rifiuta istruzioni con vassoio di origine uguale alla destinazione; "dal vassoio 1 al vassoio 2" è rispettato.
- `grasp.json` riporta `readiness`: `VALIDATED_GEOMETRY_READY_FOR_TRAJECTORY_PLAN` solo se la pinza è compatibile,
  il punto è nella zona raggiungibile (`motion_control.workspace`, centrata sulla base), la depth è misurata, intrinseche ed
  estrinseche sono verificate e `motion_control.ee_reference_pose` (nel riferimento world) è impostata. Altrimenti `NOT_READY_FOR_ROBOT_MOTION` con i motivi in `readiness_blockers`.
- IK, collisioni e contatto **non** sono verificati (`feasibility`): vanno fatti da MoveIt/`lbr_fri_ros2_stack`.
- Il nodo ROS 2 `surgical_executor` (vedi [ros2.md](ros2.md)) è solo per la piattaforma di simulazione e rifiuta il movimento senza `grasp_report` validato.
