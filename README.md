# Pipeline di percezione e presa per strumenti chirurgici

Questo progetto insegna a un robot (il braccio **KUKA LBR iiwa**) a **riconoscere uno strumento chirurgico su un
vassoio e a calcolare come afferrarlo**. Parte da una frase ("passami il bisturi"), guarda il vassoio con una camera,
trova lo strumento nell'immagine, ne ricostruisce la forma in 3D e calcola dove e come chiudere la pinza.

Funziona in tre modi:

| Modo | Cosa serve | A cosa serve |
|---|---|---|
| **Simulazione** | solo il computer | provare tutto senza hardware: scena, robot e camera sono virtuali (PyBullet) |
| **Immagine reale** | una foto del vassoio e la sua profondità | usare dati veri, ma senza collegare nulla |
| **Camera dal vivo** | camera RGB-D, calibrazione, (robot) | collegare i dispositivi fisici: la pipeline legge direttamente dalla camera |

L'ambiente di riferimento è **Docker su WSL2 (Ubuntu)** con ROS 2 Humble.

> **Stato in una riga:** simulazione e pipeline reale sono pronte e provate; il **movimento del KUKA reale non è
> ancora collegato** (la pipeline si ferma al calcolo della presa). Dettagli in
> [Collegare i dispositivi fisici](#collegare-i-dispositivi-fisici).

## Indice

- [In breve](#in-breve) · [Glossario](#glossario)
- [Prima di cominciare](#prima-di-cominciare) · [Installazione e primo avvio](#installazione-e-primo-avvio)
- [Cosa vuoi fare?](#cosa-vuoi-fare) · [Come funziona la pipeline](#come-funziona-la-pipeline)
- [Collegare i dispositivi fisici](#collegare-i-dispositivi-fisici)
- [Riferimento dei comandi](#riferimento-dei-comandi) · [Report JSON e file generati](#report-json-e-file-generati)
- [Problemi comuni](#problemi-comuni)
- [Documentazione](#documentazione) · [Struttura del repository](#struttura-del-repository) · [Test](#test) · [Dati non versionati](#dati-non-versionati)

---

## In breve

```
frase  →  piano  →  foto del vassoio  →  strumento trovato  →  forma 3D  →  punto di presa  →  braccio
"passami   (plan)    (capture)           (segment)             (deproject)   (grasp)             (movimento)
 il bisturi"
```

Ogni freccia è uno **stadio**: un piccolo programma che legge il risultato del precedente e scrive il proprio, in un
file di testo chiamato **report** (formato JSON). Si possono lanciare tutti di fila o uno alla volta, e se qualcosa va
storto basta aprire il report dello stadio che si è fermato per capire perché.

Il risultato finale è `grasp.json`: contiene la posa di presa (dove mettere la pinza, con che orientamento e
apertura) e dice se è **abilitata al movimento** oppure no, con i motivi.

## Glossario

Se un termine non ti è chiaro, cerca qui.

| Termine | Significato |
|---|---|
| **Container / Docker** | una "scatola" con tutto il software già installato (ROS 2, Python, librerie). Si lavora dentro la scatola, così funziona uguale per tutti |
| **WSL2 / WSLg** | Linux dentro Windows; WSLg permette alle finestre grafiche Linux di comparire su Windows |
| **Pipeline** | la catena di stadi, dalla frase alla posa di presa |
| **Stadio** | un passo della catena (`plan`, `capture`, `segment`, ...). Ognuno legge e scrive report |
| **Report JSON** | file di testo con il risultato di uno stadio, in `data/outputs/` |
| **PyBullet** | simulatore fisico: crea tavolo, vassoi, robot e camera virtuali |
| **Simulazione** | tutto è virtuale, nessun dispositivo collegato |
| **`--headless`** | senza finestra grafica |
| **`ground-truth`** | modalità di prova: la maschera dello strumento la dà il simulatore (perfetta), senza usare l'intelligenza artificiale |
| **SAM 3** | modello di intelligenza artificiale che "ritaglia" nell'immagine lo strumento cercato (segmentazione). È software, **non un dispositivo** |
| **Maschera** | immagine in bianco e nero: bianco dove c'è lo strumento |
| **Ollama / LLM** | programma che capisce la frase ("passami il bisturi") e la trasforma in un piano. Facoltativo: `--offline` usa un parser semplice |
| **RGB-D / profondità (depth)** | immagine a colori più, per ogni pixel, la distanza dalla camera in metri |
| **Nuvola di punti** | i punti 3D dello strumento, ottenuti da maschera + profondità |
| **PCA** | calcolo che trova gli assi principali (lungo, largo, spesso) dello strumento: serve a decidere dove afferrarlo |
| **GraspGen** | alternativa alla PCA: una rete neurale (su un PC con GPU NVIDIA) che propone molte prese |
| **IK (cinematica inversa)** | dato dove voglio la pinza, calcola gli angoli dei 7 giunti del braccio |
| **End-effector / TCP / flangia** | l'estremità del braccio dove sta la pinza |
| **Pinza** | la "mano" del robot. Quella usata qui è dimostrativa, con apertura massima 80 mm |
| **Frame / riferimento** | un sistema di coordinate (origine e assi). `world` = il mondo, `base` = la base del robot |
| **TF** | in ROS 2, il "registro" che dice dove si trova ogni frame rispetto agli altri |
| **Intrinseche** | parametri interni della camera (lunghezza focale, centro dell'immagine) |
| **Estrinseche** | dove sta la camera nel mondo (posizione e orientamento) |
| **Calibrazione mano-occhio** | misura che trova le estrinseche della camera rispetto al robot. Va fatta da chi monta il sistema |
| **Frame ottico** | convenzione della camera: x a destra, y in basso, z in avanti |
| **YAML** | file di configurazione leggibile; qui `config/surgical_environment.yaml` |
| **STL / OBJ** | formati dei modelli 3D degli strumenti |
| **`motion_safe`** | etichetta che dice "i dati sono abbastanza affidabili per muovere il robot" |
| **Pre-filtro** | controllo di fattibilità approssimato (IK, limiti, collisioni) fatto in simulazione: non sostituisce i controlli di sicurezza del robot |

## Prima di cominciare

- Windows 11 con **WSL2 e WSLg** (per le finestre grafiche) e **Docker** raggiungibile da WSL.
- Tutti i comandi di questo documento si eseguono in WSL (terminale Ubuntu), **dalla radice del repository**, tranne
  dove è detto che vanno dentro il container (là la radice è `/workspace`).
- Leggere un comando: `python3 -m pipeline.grasp --opzione valore`
  - `python3 -m <modulo>` lancia uno stadio; `python3 file.py` lancia uno script;
  - le parole che iniziano con `--` sono **opzioni**; ogni comando ha `--help` che le elenca tutte
    (es. `python3 run_pipeline_sim.py --help`).

## Installazione e primo avvio

### Passo 1 — Scarica il progetto e prepara il container

```bash
# 1. Clona il progetto con il submodule GraspGen
git clone --recurse-submodules <URL-del-repository> && cd <cartella-del-repository>
#    (se hai già clonato senza submodule: git submodule update --init)

# 2. Costruisci l'immagine (ROS 2 Humble + dipendenze Python): richiede tempo e connessione
bash scripts/docker_shell.sh build

# 3. Entra nel container: il repository è montato in /workspace
bash scripts/docker_shell.sh
```

Il prompt cambia: sei dentro il container. Per uscire, `exit`. La volta successiva basta il comando 3.
Se hai già un'immagine o un container con altri nomi, puoi riusarli:
`IMAGE_NAME=<immagine> CONTAINER_NAME=<container> bash scripts/docker_shell.sh`
(i default sono `surgical-robotics` e `surgical_ros2`).

### Passo 2 — Prima prova (dentro il container)

```bash
python3 scripts/generate_surgical_meshes.py          # crea due strumenti di prova (forbici e bisturi) in models/surgical_instruments/
python3 run_pipeline_sim.py --tool Scalpel --vision ground-truth --headless   # catena completa, senza AI né finestra
```

Cosa dovresti vedere: una serie di righe di log, una per stadio, e alla fine
`pipeline completed in ... (plan -> sim_capture -> sim_segmentation -> ... -> grasp)` e
`readiness: VALIDATED_GEOMETRY_READY_FOR_TRAJECTORY_PLAN`. I risultati sono in `data/outputs/`.

### Passo 3 — Guarda il robot (dentro il container)

```bash
bash scripts/run_sim_gui.sh --vision ground-truth    # catena completa con la finestra PyBullet del KUKA
bash scripts/run_sim_gui.sh                          # con SAM 3 (serve l'accesso ai pesi, vedi sotto)
bash scripts/run_interactive.sh                      # controllo manuale con tastiera e mouse
```

Nella finestra: `P` pausa, `R` ripete il movimento, `Q`/`Esc` chiude.

I risultati (report JSON, immagini, nuvole di punti) finiscono sempre in `data/outputs/`.

### Componenti opzionali

| Componente | A cosa serve | Come averlo |
|---|---|---|
| **Ollama** | interpretare la frase clinica | `ollama run qwen2.5:3b` sull'host; senza, usa `--offline` |
| **SAM 3** | segmentazione con intelligenza artificiale | modello da Hugging Face (`transformers`); in alternativa `--vision ground-truth` |
| **GraspGen** | proporre prese con una rete neurale | server su un PC Linux con GPU NVIDIA, il client è già nell'immagine. Setup in [docs/graspgen.md](docs/graspgen.md) |
| **ROS 2** | eseguire il movimento pianificato | il nodo `surgical_executor` (in `ros2/`); i repository esterni (`llm_ros`, `sam3_ros`, `lbr_fri_ros2_stack`, `fri`) si scaricano con `ros2/workspace.repos`. Build e uso in [docs/ros2.md](docs/ros2.md) |

## Cosa vuoi fare?

| Voglio... | Comando (nel container) | Dove guardare |
|---|---|---|
| provare tutto senza AI né finestra | `python3 run_pipeline_sim.py --tool Scalpel --offline --vision ground-truth` | `data/outputs/grasp.json` |
| vedere il robot muoversi in simulazione | `bash scripts/run_sim_gui.sh --vision ground-truth` | la finestra; `sim_robot_motion.json` |
| usare l'intelligenza artificiale (SAM 3) | `bash scripts/run_sim_gui.sh` | `sim_segmentation_overlay.png` |
| dare una frase al posto del nome | `python3 run_pipeline_sim.py --instruction "Passami il bisturi dal vassoio 2 al vassoio 1"` | `plan.json` |
| rilanciare solo una parte | `python3 run_pipeline_sim.py --tool Scalpel --from-stage deproject` | il report dello stadio |
| fermarmi a metà per controllare | `python3 run_pipeline_sim.py --tool Scalpel --stop-after segment` | i file dello stadio |
| pilotare il robot a mano | `bash scripts/run_interactive.sh` | la finestra |
| usare una foto vera con la sua profondità | `python3 run_pipeline_real.py --image foto.png --depth depth.npy --intrinsics 615 615 320 240` | `real_capture_rgb.png`, `grasp.json` |
| collegare la camera fisica | vedi [Procedura per la prima prova](#procedura-per-la-prima-prova) | `real_preflight.json` |
| provare la camera senza averla | `python3 scripts/fake_ros2_camera.py &` poi `python3 -m pipeline.real.preflight --ee-frame fake_ee` | l'esito a schermo |
| usare GraspGen | `... --grasp-backend graspgen --no-motion` (serve il server GPU) | `grasp_candidates.png` |
| ripulire i file generati | `python3 -m pipeline.clean --yes` | `data/outputs/` |
| lanciare i test | `python3 -m unittest discover -s tests` | l'esito a schermo |

## Come funziona la pipeline

Sei stadi indipendenti, ognuno un modulo Python che legge i report JSON dei precedenti e scrive il proprio:

```
plan → capture → segment → reference_pose → deproject → grasp   (+ movimento KUKA dimostrativo)
```

| Stadio | Cosa fa, in parole semplici |
|---|---|
| `plan` | capisce la frase e decide quale strumento prendere e tra quali vassoi spostarlo |
| `capture` | "scatta la foto": immagine a colori e profondità del vassoio (in simulazione la crea PyBullet) |
| `segment` | trova lo strumento nell'immagine e ne produce la maschera (con SAM 3, o col simulatore in modalità `ground-truth`) |
| `reference_pose` | (solo simulazione) posa certificata con cui lo strumento è appoggiato nel vassoio |
| `deproject` | trasforma maschera + profondità in una nuvola di punti 3D nel riferimento del robot |
| `grasp` | calcola dove e come afferrare lo strumento, verifica che il braccio possa arrivarci e scrive `grasp.json` |

Ogni stadio si lancia da solo (`python3 -m pipeline.sim.segment --help`) oppure in catena con `run_pipeline_sim.py`
(`--from-stage`, `--stop-after`). Dettagli, contratto dei report e pipeline su immagine reale:
[docs/pipeline.md](docs/pipeline.md).

---

## Collegare i dispositivi fisici

La pipeline reale può acquisire **dal vivo** da una camera RGB-D ROS 2 (`--source ros2`) e leggere dalla TF la
posa della camera e dell'end-effector. Prima di ogni prova c'è un controllo che non muove nulla
(`pipeline.real.preflight`). **Il movimento del KUKA reale non è ancora collegato**: vedi "Stato" qui sotto.

### Stato

| Parte | Stato |
|---|---|
| Acquisizione RGB-D da camera ROS 2 (colore, profondità allineata, `CameraInfo`) | **pronta** (`--source ros2`), provata con una camera finta |
| Intrinseche dalla camera (`CameraInfo`) | **pronte** |
| Posa camera dalla TF (calibrazione mano-occhio pubblicata come trasformazione) | **pronta**, serve la calibrazione vera |
| Posa dell'end-effector dalla TF a ogni cattura | **pronta** (`--ee-frame`) |
| Segmentazione SAM 3, nuvola di punti, geometria di presa, verifica IK/collisioni | pronte (software, nessun dispositivo) |
| Preflight prima della prova | **pronto** |
| Calibrazione mano-occhio | **da fare** con il vostro setup: il codice non la esegue |
| Movimento del KUKA reale (MoveIt / ros2_control, driver della pinza) | **non implementato**: il nodo `surgical_executor` parla solo con la piattaforma di simulazione |
| Pinza reale | la presa è calcolata per la pinza dimostrativa da 80 mm; con la Robotiq 2F-140 (GraspGen) il movimento resta bloccato |

SAM 3 **non è un dispositivo**: è il modello di segmentazione e gira come software su un PC. I dispositivi sono la
camera RGB-D, il braccio KUKA e la pinza.

### Cosa deve pubblicare l'hardware

La camera e il robot devono "pubblicare" su ROS 2 queste informazioni:

| Cosa | Tipo | Requisiti |
|---|---|---|
| Immagine a colori | `sensor_msgs/Image` | `rgb8` o `bgr8`, di preferenza rettificata (niente distorsione) |
| Profondità | `sensor_msgs/Image` | **allineata al colore** (stessa risoluzione), `16UC1` in mm o `32FC1` in m |
| Parametri camera | `sensor_msgs/CameraInfo` | calibrata (matrice `K` non nulla); `frame_id` = frame **ottico** (x destra, y basso, z avanti) |
| Calibrazione mano-occhio | TF `world → frame ottico della camera` | posa del frame **ottico**, non di `camera_link` |
| Posa dell'end-effector | TF `world → frame flangia/TCP` | es. `lbr_link_ee` dal driver del KUKA |

`world` è il riferimento in cui la pipeline esprime camera, vassoi e base del robot (`environment.kuka_base` nel
YAML è la posa della base in `world`). Se la vostra TF non ha un frame `world` e usa `lbr_link_0` come radice,
passate `--world-frame lbr_link_0` e lasciate `kuka_base` con posizione e rotazione nulle: il preflight segnala
l'incoerenza. I nomi dei topic e dei frame sono in `camera_overhead.ros2` del YAML e si sovrascrivono da riga di comando.

Esempio di pubblicazione della calibrazione mano-occhio come TF statica (i numeri vengono dalla vostra calibrazione):

```bash
ros2 run tf2_ros static_transform_publisher --x 0.60 --y -0.42 --z 1.15 \
    --qx 0.707107 --qy 0.707107 --qz 0.0 --qw 0.0 --frame-id world --child-frame-id camera_color_optical_frame
```

### Procedura per la prima prova

Tutto nel container (`bash scripts/docker_shell.sh`), con ROS 2 caricato dall'entrypoint.

1. **Prova a secco senza hardware**: una camera finta pubblica gli stessi topic e le stesse TF.
   ```bash
   python3 scripts/fake_ros2_camera.py &                       # terminale 1
   python3 -m pipeline.real.preflight --ee-frame fake_ee       # terminale 2: tutto OK tranne l'avviso sulle estrinseche
   python3 -m pipeline.real.capture --source ros2 --ee-frame fake_ee
   ```
   La camera finta prova solo l'acquisizione e la catena dei frame, non la calibrazione né la segmentazione.
2. **Collegare la camera reale** e il driver (profondità allineata al colore). Controllare i topic con
   `ros2 topic list` e `ros2 topic hz <topic>`; adattare `camera_overhead.ros2` nel YAML se i nomi differiscono.
3. **Calibrare** (mano-occhio) e pubblicare la TF `world → frame ottico`. Verificare con
   `ros2 run tf2_ros tf2_echo world <frame_ottico>`.
4. **Collegare il KUKA** con `lbr_fri_ros2_stack` e verificare la TF `world → flangia`.
5. **Preflight**:
   ```bash
   python3 -m pipeline.real.preflight --ee-frame lbr_link_ee
   ```
   Stampa `OK / WARN / FAIL` con il rimedio e scrive `data/outputs/real_preflight.json`. Esce con codice 1 se c'è
   un `FAIL`. Controlla: topic e risoluzione, profondità valida, sincronia colore/profondità, intrinseche,
   distorsione, TF della camera e dell'end-effector, coerenza tra profondità e posa della camera, estrinseche
   verificate nel YAML, pesi SAM 3, Ollama (e il server GraspGen con `--grasp-backend graspgen`).
6. **Dichiarare verificata la calibrazione** solo dopo averla misurata: `camera_overhead.extrinsics.verified: true`.
   Senza, ogni presa resta `NOT_READY_FOR_ROBOT_MOTION`. Con la camera in posizione:
   ```bash
   python3 -m pipeline.real.preflight --ee-frame lbr_link_ee    # atteso: "pronto. Una cattura adesso risulterebbe motion_safe"
   ```
7. **Pipeline senza movimento**, ispezionando le immagini e i report a ogni stadio:
   ```bash
   python3 run_pipeline_real.py --source ros2 --ee-frame lbr_link_ee --instruction "Passami le forbici" --stop-after capture
   python3 run_pipeline_real.py --source ros2 --ee-frame lbr_link_ee --instruction "Passami le forbici"
   ```
   Controllare `real_capture_rgb.png`, la maschera `real_segmentation_overlay.png`, la nuvola
   `real_deprojection_cloud.ply` e in `grasp.json` il campo `readiness` (e `readiness_blockers`).
8. **Il movimento** non parte da qui: `surgical_executor` è per la simulazione. Per il robot reale va scritto il
   backend (vedi [docs/ros2.md](docs/ros2.md)). Finché non c'è, la pipeline termina con `grasp.json` e la posa
   di presa nel riferimento della base.

### Come la pipeline decide se la presa è abilitata al movimento

`real_capture.json` ha `motion_safe: true` solo se **tutte** queste condizioni sono vere: profondità misurata,
intrinseche da `CameraInfo`, posa della camera letta dalla TF e `camera_overhead.extrinsics.verified: true`.
Inoltre `grasp.json` richiede una posa dell'end-effector non nominale (meglio dalla TF con `--ee-frame`, oppure
`motion_control.ee_reference_pose`) e almeno un candidato di presa fattibile (IK, limiti di giunto, collisioni nel
modello della cella). Altrimenti `readiness` è `NOT_READY_FOR_ROBOT_MOTION` e `readiness_blockers` elenca i motivi.

### Limiti da conoscere

- La verifica di IK e collisioni è un pre-filtro su un modello PyBullet con la geometria dell'iiwa 14 e una pinza
  dimostrativa: non sostituisce MoveIt né i controlli di sicurezza del robot.
- La calibrazione mano-occhio la esegue chi monta il sistema. Una posa nominale o sbagliata sposta tutte le prese.
- Le immagini vengono trattate come pinhole: usare un flusso rettificato.
- L'uso in ambito clinico richiede validazione e conformità normativa che questo codice non fornisce.

---

## Riferimento dei comandi

Questa sezione elenca tutto: opzioni, file letti e file scritti. Ogni comando ha `--help` con l'elenco completo
(`python3 -m pipeline.grasp --help`, `python3 run_pipeline_sim.py --help`, ...). Tutti i comandi sono da lanciare dalla
radice del repository (nel container: `/workspace`). Gli stadi sono moduli: si lanciano con `python3 -m <modulo>`.

### Dati in ingresso

| Cosa | Dove | Note |
|---|---|---|
| Mesh STL/OBJ di test | `models/surgical_instruments/` | create da `python3 scripts/generate_surgical_meshes.py` (forbici e bisturi). Anche `data/inputs/` viene cercata |
| Dataset 3D reale (3D-COSI, 232 modelli) | `data/inputs/Surgical Instruments/` | poi `python3 scripts/build_instruments_catalog.py` scrive `data/inputs/instruments_catalog.json`; `run_pipeline_sim.py --list-tools` mostra gli strumenti |
| Foto reale del vassoio | `data/inputs/surgical_scene.jpg` (o `--image`) | pipeline reale da file; con `--source ros2` si legge dalla camera |
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

La sorgente è un **file** (default) o la **camera ROS 2** dal vivo (`--source ros2`, vedi
[Collegare i dispositivi fisici](#collegare-i-dispositivi-fisici)).

```bash
# da file
python3 run_pipeline_real.py --instruction "Passami le forbici" --image foto.png --depth depth.npy \
    --intrinsics 615 615 320 240
# dal vivo: camera RGB-D + TF
python3 -m pipeline.real.preflight --ee-frame lbr_link_ee                  # prima di ogni prova
python3 run_pipeline_real.py --source ros2 --ee-frame lbr_link_ee --instruction "Passami le forbici"
```

| Opzione | Significato |
|---|---|
| `--source file\|ros2` | file su disco (default) oppure camera ROS 2 dal vivo |
| `--image FILE` | foto del vassoio con tutti gli strumenti (solo `--source file`) |
| `--depth FILE` | depth metrica allineata (`.npy` in m o PNG 16 bit in mm); senza, piano costante nominale e presa non abilitata |
| `--intrinsics FX FY CX CY` | intrinseche calibrate in pixel di `--image` (con `ros2` vengono da `CameraInfo`) |
| `--color-topic`, `--depth-topic`, `--camera-info-topic` | topic della camera (solo `ros2`; default in `camera_overhead.ros2` del YAML) |
| `--world-frame` | frame TF in cui la pipeline esprime le pose (default `world`) |
| `--camera-frame` | frame TF ottico della camera (default: `frame_id` di `CameraInfo`) |
| `--ee-frame` | frame TF della flangia/TCP: la posa dell'end-effector viene letta a ogni cattura |
| `--timeout S`, `--depth-frames N` | attesa dei messaggi (default 10 s) e fotogrammi di profondità per la mediana (default 5) |
| `--min-confidence X` | soglia SAM 3 (default 0.3) |
| `--instruction`, `--from-stage`, `--stop-after`, `--grasp-backend`, `--graspgen-*`, `--config`, `--out-dir`, `-v` | come nella pipeline simulata (gli stadi sono `plan`, `capture`, `segment`, `deproject`, `grasp`) |

Stadi singoli: `python3 -m pipeline.real.capture | segment | deproject` (`capture` accetta le stesse opzioni di
sorgente). Scrivono `real_capture.json` (+ `real_capture_rgb.png`, `real_capture_depth.npy`),
`real_segmentation.json` (+ `real_segmentation_mask.png`, `real_segmentation_overlay.png`),
`real_deprojection.json` (+ `real_deprojection_cloud.ply`) e poi `grasp.json`:

```bash
python3 -m pipeline.real.capture --source ros2 --ee-frame lbr_link_ee
python3 -m pipeline.real.segment --min-confidence 0.4
python3 -m pipeline.real.deproject
python3 -m pipeline.grasp --deprojection data/outputs/real_deprojection.json
```

**Preflight** (non muove nulla): `python3 -m pipeline.real.preflight [--source ros2|file] [--ee-frame F]
[--grasp-backend pca|graspgen]`. Scrive `real_preflight.json` ed esce con codice 1 se c'è un `FAIL`.
**Camera finta** per prove a secco: `python3 scripts/fake_ros2_camera.py --help`.

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

Pilota il braccio a mano nella finestra PyBullet.

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

---

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

Come si legge: `stage` e `status` dicono chi è e se è andato bene; `inputs` sono i report da cui dipende;
`artifacts` sono i file collegati (immagini, nuvole di punti); `data` è il risultato vero e proprio; `warnings` sono
gli avvisi; `error` spiega il problema se lo stadio è fallito.

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
| `real_capture.json` | capture reale | come la simulata, più `source_type` (`real_surgical_tray_image` o `live_ros2_rgbd`), `depth_source`, `intrinsics_calibrated`, `extrinsics_verified`, `motion_safe` e, dal vivo, `acquisition` (topic, frame, sincronia, pixel validi) ed `ee_reference_pose` (da TF) |
| `real_segmentation.json`, `real_deprojection.json` | pipeline reale | stessi campi della controparte simulata; la deproiezione riporta `ee_reference_source` (`tf`, `configured` o `nominal`) |
| `real_preflight.json` | preflight | `checks` (stato, dettaglio, rimedio), `ready_for_motion` |
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

Se preferisci chiamare gli stadi da un tuo script invece che dal terminale:

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

---

## Problemi comuni

| Cosa vedi | Perché e cosa fare |
|---|---|
| `Immagine 'surgical-robotics' non trovata` | l'immagine Docker non è stata costruita (o ha un altro nome): `bash scripts/docker_shell.sh build`, oppure usa `IMAGE_NAME` e `CONTAINER_NAME` |
| `Nessuna mesh trovata: aggiungi gli STL in data/inputs` | mancano i modelli 3D: `python3 scripts/generate_surgical_meshes.py` (strumenti di prova) oppure metti il dataset in `data/inputs/` |
| `Catalog not found. Generate it with scripts/build_instruments_catalog.py` (con `--list-tools`) | il catalogo del dataset reale non è stato creato. Con le mesh di prova puoi usare comunque `--tool Scalpel` / `--tool Scissors` |
| Il primo stadio (`plan`) impiega circa 12 secondi e dice `deterministic parser in use` | Ollama non è raggiungibile e la pipeline usa il parser semplice. Per non aspettare usa `--offline` |
| Il preflight dice `SAM 3: pesi non in cache` | i pesi del modello non sono scaricati (il caricamento è offline). Scaricali con l'accesso Hugging Face, oppure usa `--vision ground-truth` |
| Il preflight dice `FAIL` su camera o TF | leggi la riga `->` sotto il FAIL: indica il rimedio (driver non avviato, topic con altro nome, TF non pubblicata). I nomi si cambiano in `camera_overhead.ros2` |
| `grasp.json` ha `NOT_READY_FOR_ROBOT_MOTION` | non è un errore: la presa non è abilitata al movimento. Guarda `readiness_blockers` (estrinseche non verificate, depth non misurata, posa EE nominale, punto fuori dalla zona raggiungibile, nessun candidato fattibile) |
| Uno stadio dice `upstream report not found` | manca il report dello stadio precedente: lancia quello prima, oppure non usare `--from-stage` su stadi che non hai mai eseguito |
| Uno stadio dice `FAILED` | apri il suo `.json` in `data/outputs/`: il campo `error` spiega il motivo |
| 19 test di `test_graspgen` falliscono | il submodule è vuoto: `git submodule update --init` |
| Compare la riga `pybullet build time: ...` | è normale: è l'avviso di caricamento di PyBullet |
| La finestra grafica non si apre | serve WSLg (prova un'altra app grafica Linux da WSL); la GUI non funziona con `--headless` |

---

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
scripts/           docker_shell.sh, run_sim_gui.sh, run_interactive.sh, ros2_build.sh, fake_ros2_camera.py e strumenti di sviluppo
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

### Profili GraspGen e sistema completo

Il nuovo launcher compone gli stessi moduli di presa e movimento con due profili:

```bash
.venv/bin/python run_system.py --profile graspgen_only --gui
.venv/bin/python run_system.py --profile full_system --check
.venv/bin/python run_system.py --profile full_system \
  --instruction "Pick up the Bone Forceps from tray 2 and place it in tray 1." --gui
```

`full_system` richiede Ollama e SAM3 effettivi: nessun fallback al parser o alla maschera
PyBullet. Rimane una simulazione single-object con presa ideale. Preparazione dei modelli,
output e limiti sono descritti in [docs/full-system.md](docs/full-system.md).
