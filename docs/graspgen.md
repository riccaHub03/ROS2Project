# GraspGen collegato alla pipeline

> **Nota sull'ambiente.** Questo documento nasce da una verifica fatta su un client Mac con `.venv`. I comandi `.venv/bin/python` equivalgono a `python3` dentro il container Docker del progetto (vedi il [README](../README.md)); `<RADICE_PROGETTO>` è la cartella del repository (`/workspace` nel container). Il server GraspGen resta su un PC Linux NVIDIA separato.

Lo stadio `pipeline.grasp` legge la nuvola di `deproject`, invia XYZ al **client
ufficiale ZMQ** e riceve le pose generate e i punteggi del discriminatore
ufficiale. Il modello rimane integralmente nel submodule NVlabs/GraspGen,
commit `2dd8852e1be60f5f9d277fafcc621835cdf59110`, senza modifiche.

La configurazione è `config/graspgen.yaml`. PCA rimane predefinito. GraspGen
non ripiega su PCA: errori di connessione, checkpoint, geometria o risposta
scrivono `FAILED` e restituiscono exit code 1.

## Separazione degli ambienti

| Ambiente | Python / dipendenze | Funzione |
| --- | --- | --- |
| ROS2Project su Mac o PC | Python 3.11, `requirements-core.txt`, `requirements-graspgen-client.txt` | Pipeline, geometria, PCA, client ZMQ, test, preview |
| Visione opzionale ROS2Project | `requirements.txt` | SAM3, PyTorch e torchvision del progetto; Ollama esterno per il planner |
| GraspGen sul PC | Container Linux x86_64, CUDA 12.1, torch 2.1.0 / torchvision 0.16.0 | Generatore e discriminatore ufficiali con pesi preaddestrati |

Non installare `third_party/GraspGen` con `pip install -e` nella `.venv` del
progetto: richiederebbe versioni di torch/transformers diverse da SAM3.
Il client viene caricato direttamente da `grasp_gen/serving/zmq_client.py`:
nessun import di torch/CUDA è necessario per usarlo.

Il profilo GPU preparato è per Turing/Ampere/Ada/Hopper con driver compatibile
CUDA 12.1 (tipicamente serie 525 o successiva). **Blackwell richiede un profilo
CUDA 12.8+ da verificare separatamente**: `check-gpu` lo segnala. Non usare alla
cieca il Dockerfile CUDA 12.8 upstream, che nella versione fissata mescola
wheel torch/CUDA differenti. Non è stata eseguita alcuna build GPU sul Mac.

## Preparazione e verifica sul Mac

Da questa directory:

```bash
cd <RADICE_PROGETTO>
# Solo per un ambiente nuovo; per quello esistente usare .venv/bin/python.
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-core.txt -r requirements-graspgen-client.txt
bash scripts/graspgen_demos.sh init
bash scripts/graspgen_demos.sh check
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
.venv/bin/python -m unittest discover -s tests -v
```

I test usano server **simulati**, su porte effimere di `127.0.0.1`, e il client
ufficiale vero. Non caricano checkpoint. Un sandbox che proibisce i socket
locali deve consentire il loopback per eseguirli; non serve una GPU.
`tests/test_graspgen.py` comprende anche la pipeline PyBullet completa headless
con base traslata e ruotata e verifica sia la visualizzazione sia il blocco del
movimento. I test preesistenti del controller interattivo restano nella suite.

Per generare una nuvola dei nostri strumenti senza SAM3/Ollama:

```bash
.venv/bin/python run_pipeline_sim.py --tool Scissors --offline --vision ground-truth \
  --stop-after deproject --out-dir data/outputs/scissors_cloud
```

Richiede gli STL e il catalogo già previsti dal progetto, oppure `--config`
con `surgical_instrument.stl_path` valido. L'output è in metri nella base robot.
Il controller manuale rimane disponibile con `bash scripts/run_interactive.sh`.

## Ricreare ROS2Project sul PC

Trasferire la repository con le modifiche locali (non basta clonare la versione
remota se queste modifiche non sono state pubblicate). Non trasferire `.venv`:
contiene binari macOS. Sul PC, dalla directory ROS2Project:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-core.txt -r requirements-graspgen-client.txt
bash scripts/graspgen_demos.sh init
bash scripts/graspgen_demos.sh check
```

Questo basta per client, test e pipeline con `--offline --vision ground-truth`.
Per SAM3 aggiungere le dipendenze di `requirements.txt` **solo in questa
`.venv`**, predisporre accesso/cache dei pesi SAM3 come nel README, e avviare
Ollama con il modello/configurazione già usati dal progetto. `--offline`
elimina Ollama nella pipeline simulata; `--vision ground-truth` elimina SAM3.
La pipeline reale da foto usa SAM3 e il planner esistente.

I file `requirements-core.txt` e `requirements-graspgen-client.txt` fissano
le dipendenze dirette verificate sul Mac. Non sono un lock delle dipendenze
transitive per tutte le piattaforme. Dopo la prima installazione funzionante
sul PC salvare `pip freeze` e il digest dell'immagine per riprodurre anche
le versioni transitive:

```bash
.venv/bin/python -m pip freeze > data/external/ros2project-linux-freeze.txt
docker image inspect ros2project-graspgen:cuda121 --format '{{.Id}}'
```

### Dati ignorati da Git e percorsi trasferibili

Per riusare una deproiezione salvata, creare sul Mac un bundle con report,
nuvola e configurazione di riferimento. Il comando richiede una destinazione
nuova e conserva il report originale per audit:

```bash
.venv/bin/python -m pipeline.graspgen.bundle \
  --deprojection data/outputs/scissors_cloud/sim_deprojection.json \
  --destination data/external/transfer/scissors
```

Trasferire `data/external/transfer/scissors/` al PC mantenendo la directory.
`deprojection.json` usa `cloud.ply` relativo al report e funziona su entrambi
i computer. I report normali possono contenere percorsi assoluti del Mac:
usare il bundle evita sostituzioni manuali errate. La configurazione nel
bundle è una copia di riferimento; quella operativa va passata esplicitamente
con `--graspgen-config` o rimane `config/graspgen.yaml`.

Per la pipeline completa trasferire anche STL/OBJ, immagini e cataloghi in
`data/inputs/`, la collezione `surgical-instruments-collection/` se usata,
e le proprie configurazioni. Esempio dal Mac, sostituendo host e directory:

```bash
rsync -av --exclude .git --exclude .venv --exclude .cache --exclude __pycache__ \
  --exclude 'data/external/GraspGenModels' \
  <RADICE_PROGETTO>/ USER@PC:/PERCORSO/ROS2Project/
```

Questa copia comprende modifiche locali e dati ignorati salvo le esclusioni;
la destinazione deve essere un checkout Git del progetto, così il gitlink del
submodule resta disponibile. Eseguire lì `init`. Controllare e aggiornare
`stl_path`, cataloghi con percorsi assoluti ed eventuali configurazioni proprie.
Ricostruire il catalogo, se necessario, con
`python scripts/build_instruments_catalog.py`, che cerca il dataset nelle
cartelle `data/inputs/c_Surgical Instruments/Surgical Instruments` o
`data/inputs/Surgical Instruments`.
I checkpoint GraspGen e `sample_data` si scaricano sul PC; cache SAM3 e altri
asset ignorati vanno trasferiti separatamente o riscaricati. Non copiare token
Hugging Face o file `.env` senza valutarne esplicitamente il contenuto.

## Preparazione del runtime NVIDIA

Installare sul PC Linux x86_64 driver NVIDIA, Docker Engine,
NVIDIA Container Toolkit e git-lfs, poi:

```bash
bash scripts/graspgen_demos.sh check-gpu
bash scripts/graspgen_demos.sh models
bash scripts/graspgen_demos.sh check-models
bash scripts/graspgen_demos.sh build
bash scripts/graspgen_demos.sh smoke-gpu
```

`check-gpu` verifica driver, architettura, daemon Docker e passthrough con
`docker run --gpus all ... nvidia-smi`; può scaricare la piccola immagine Ubuntu
usata dal controllo. `models` clona
[GraspGenModels](https://huggingface.co/adithyamurali/GraspGenModels), esegue LFS
pull e valida entrambe le reti e `sample_data/real_object_pc/*.json`. Non scarica
il dataset di training. Non aggiorna automaticamente un checkout di modelli
esistente: registrarne la revisione stampata e conservarla per i confronti.

Percorsi verificati nel codice fissato:

```text
third_party/GraspGen/grasp_gen/serving/zmq_client.py
third_party/GraspGen/grasp_gen/serving/zmq_server.py
third_party/GraspGen/client-server/graspgen_server.py
third_party/GraspGen/scripts/demo_object_pc.py
third_party/GraspGen/docker/graspgen_cuda121.dockerfile

data/external/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml
data/external/GraspGenModels/checkpoints/graspgen_robotiq_2f_140_gen.pth
data/external/GraspGenModels/checkpoints/graspgen_robotiq_2f_140_dis.pth
data/external/GraspGenModels/sample_data/real_object_pc/*.json
```

`build` genera una **copia temporanea** del Dockerfile ufficiale: rimuove il
backport `pickle5` incompatibile con Python 3.10, allinea torchvision e le wheel
PyG a torch 2.1.0/CUDA 12.1, fissa NumPy 1.26.4, setuptools<78 e OpenCV<4.12,
e compila PointNet senza build isolation per architetture 7.5/8.0/8.6/8.9/9.0.
Il nostro overlay aggiunge dipendenze runtime mancanti (Viser, client ZMQ,
transformers/tensordict). Questi aggiustamenti sono nella nostra repo;
il submodule e gli algoritmi restano invariati. La ricetta base upstream
installa anche dipendenze opzionali: build e import devono essere verificati
sul PC con `smoke-gpu`, non sono certificati dal solo controllo statico Mac.

Non vengono usati `docker/serve.dockerfile` o `docker/run_server.sh` upstream:
nella revisione fissata puntano a `tools/graspgen_server.py`, che non esiste.
Il nostro wrapper importa direttamente `GraspGenZMQServer` ufficiale.
Il codice viene montato in sola lettura; gli output della demo mesh vanno in
`data/external/graspgen_results/` (container: `/results`). La struttura
`/integration/third_party/GraspGen` conserva il percorso relativo del gitdir
necessario a verificare il commit anche nel container.

Variabili supportate dallo script: `PIPELINE_PYTHON`, `GRASPGEN_MODELS`
(percorso assoluto), `GRASPGEN_IMAGE`, `GRASPGEN_PORT`.
L'ambiente GPU è ricreabile con le due ricette Docker; non richiede di
modificare o installare la `.venv` ROS2Project nel container.

## Ordine delle prove sul PC

### 1. Demo ufficiale sui dati degli autori

```bash
bash scripts/graspgen_demos.sh demo-pc
# Oppure batch senza Viser:
bash scripts/graspgen_demos.sh demo-pc --no-visualization
# Controllo aggiuntivo su box.obj:
bash scripts/graspgen_demos.sh demo-mesh --no-visualization
```

La demo PC visuale apre Viser su `http://127.0.0.1:8080` e richiede Invio per
passare all'oggetto successivo. Le opzioni aggiuntive sono quelle degli script
ufficiali. **Il wrapper passa sempre `--planner diffusion`**: la demo della
revisione attuale ha invece `graspmoe` come default. GraspMoE/OBB è un'estensione
successiva e non deve essere confusa con il primo confronto al paper v1.
Il checkpoint Robotiq pubblicato usa backbone PointNet: il percorso diffusion
è quello richiesto, ma questo non equivale a riprodurre ogni ablation/PTv3 e
ogni numero del paper.

### 2. Gli stessi dati attraverso il nostro backend

In un terminale lasciare il server in esecuzione:

```bash
bash scripts/graspgen_demos.sh server
```

Il wrapper verifica commit, integrità del submodule, nomi delle pinze nei tre
blocchi YAML e presenza di entrambi i pesi, calcola SHA256 e li espone nei
metadata. La classe server ufficiale carica `GraspGenSampler`, che carica
**generatore + discriminatore** e chiama `run_inference`; non è un modello
ricreato nel progetto. Porta host: solo loopback `127.0.0.1:5556`.

In un secondo terminale, dalla root del progetto:

```bash
source .venv/bin/activate
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
# Tutti gli oggetti della stessa directory della demo ufficiale:
for sample in data/external/GraspGenModels/sample_data/real_object_pc/*.json; do
  name="$(basename "$sample" .json)"
  python -m pipeline.graspgen.import_cloud --input "$sample" --units m --frame base \
    --out-dir "data/outputs/authors/$name"
  python -m pipeline.grasp --grasp-backend graspgen \
    --deprojection "data/outputs/authors/$name/imported_deprojection.json" \
    --out-dir "data/outputs/authors/$name" || break
done
```

Qui `--frame base` etichetta esplicitamente il frame dei dati di esempio come
frame locale di confronto, con base/world coincidenti: **non è una calibrazione
robot degli autori**. La nuvola grezza `pc` è la stessa; la demo ufficiale
applica un filtro outlier prima di `run_inference`, che ne applica un altro.
Il backend usa un solo filtro server e un cap deterministico di 4096 punti.
Pertanto confrontare validità, pose, distribuzione e punteggi registrando il
preprocessing; non pretendere identità numerica da due inferenze stocastiche.

### 3. Nuvole dei nostri strumenti già salvate

```bash
python -m pipeline.grasp --grasp-backend graspgen \
  --deprojection data/external/transfer/scissors/deprojection.json \
  --out-dir data/outputs/scissors_graspgen
# Confronto PCA sul report originale locale (PCA usa il PLY della pipeline):
python -m pipeline.grasp --grasp-backend pca \
  --deprojection data/outputs/scissors_cloud/sim_deprojection.json \
  --out-dir data/outputs/scissors_pca
```

Il secondo comando richiede il report/Ply originale valido sul PC. Si può
anche usare `import_cloud` per PLY ASCII/NPY già in metri; per un frame camera
servono `--frame camera --T-base-from-cloud extrinsic.json` (matrice 4x4).
Per una base non coincidente col mondo passare `--base-pose base.json` nel
formato `RobotBaseFrame.to_dict()`. I frame sconosciuti vengono rifiutati.

### 4. Pipeline completa, inizialmente visuale e senza movimento

```bash
# Catena simulata senza finestra, senza SAM3/Ollama, poi overview PNG:
python run_pipeline_sim.py --tool Scissors --offline --vision ground-truth \
  --grasp-backend graspgen --out-dir data/outputs/full_graspgen
# Scena KUKA/Robotiq candidate, robot fermo:
python run_pipeline_sim.py --gui --vision ground-truth --tool Scissors \
  --grasp-backend graspgen --no-motion --out-dir data/outputs/gui_graspgen
# Stessa interfaccia tramite lo script Mac (anche con server remoto):
bash scripts/run_sim_gui.sh --vision ground-truth --grasp-backend graspgen --no-motion
# Quando SAM3 e Ollama sono pronti:
python run_pipeline_sim.py --gui --vision sam3 --tool Scissors \
  --grasp-backend graspgen --no-motion --out-dir data/outputs/sam3_graspgen
# Pipeline da immagine reale: termina con inferenza/report, non muove hardware.
python run_pipeline_real.py --image data/inputs/surgical_scene.jpg \
  --grasp-backend graspgen --out-dir data/outputs/real_graspgen
```

Per riusare stadi esistenti della pipeline sim/reale usare `--from-stage grasp`
e lo stesso `--out-dir`. Tutti gli avvii accettano `--graspgen-config`,
`--graspgen-host`, `--graspgen-port`, `--graspgen-timeout-ms`. Omettere
`--grasp-backend` mantiene PCA e il comportamento dimostrativo precedente.
La GUI mostra linee nominali della Robotiq e il TCP; non sostituisce il modello
robot visuale con una simulazione fisica Robotiq.

Per chiamare il server PC dal Mac mantenendo loopback, aprire un tunnel:

```bash
ssh -N -L 5556:127.0.0.1:5556 USER@PC
# In un altro terminale Mac, stessi comandi pipeline con default host 127.0.0.1.
```

Il client disabilita l'attesa infinita ufficiale (`wait_for_server=False`):
metadata e inferenza hanno ciascuno un timeout configurabile, default 60 s.
Non ritenta una richiesta scaduta; rilanciare dopo aver controllato il server.
La GPU può ancora completare il lavoro dopo il timeout del client.

## Geometria, report e pinza

Tutte le coordinate sono in **metri**. Il PLY non codifica unità: il report
`pointcloud_units=m` o, per report legacy, l'asserzione `input.units=m` è
necessaria; vengono controllati finitezza, cardinalità ed estensione metrica.
Non vengono applicati fattori di scala automatici o il filtro PCA del piano.
I limiti di estensione individuano errori evidenti ma non dimostrano da soli
la scala corretta. La pinza va calibrata con dati misurati.

Si usa `T_A_B` per trasformare coordinate colonna da B ad A:

```text
P_base = T_base_cloud P_cloud
P_centered = P_base - mean(P_base)
T_base_gripper = Translation(mean(P_base)) T_centered_gripper
T_base_tcp = T_base_gripper T_gripper_tcp
T_world_tcp = T_world_base T_base_tcp
```

Il sampler ufficiale ricentra internamente e **restituisce già** la traslazione
nel frame della nuvola ricevuta: il backend ripristina solo il proprio centro
esterno. Non sottrae/aggiunge quel centro due volte. Nuvole deproject normali
sono già nella base; un input world usa l'inversa della posa della base;
un input camera richiede una trasformazione esplicita, mai dedotta dal nome.
La posa della base deve essere presente nel report, anche se identità.
Il supporto a base traslata/ruotata è conservato, incluse tutte le rotazioni
nelle pose e gli offset TCP. Quaternioni della pipeline: `[x,y,z,w]`.

Per Robotiq il frame ufficiale è la **base pinza**, `+Z` avvicinamento e `+X`
chiusura. Il TCP nominale è a `[0,0,0.195]` m in quel frame; la nostra
convenzione TCP chiude lungo `+Y`, quindi `T_gripper_tcp` comprende anche
una rotazione di -90° attorno a Z. Non usare il solo offset di 0.10 m del
TCP dimostrativo KUKA come se fosse la geometria Robotiq. Le linee di preview
usano larghezza nominale 0.13603458 m e profondità 0.195 m dal file ufficiale,
non un'apertura predetta: il protocollo non restituisce una larghezza per presa.

Report scritti:

| File | Contenuto |
| --- | --- |
| `grasp.json` | Interfaccia corrente della pipeline, anche `FAILED` se GraspGen fallisce |
| `grasp_pca.json` | Ultima esecuzione PCA, preservata quando si lancia GraspGen |
| `grasp_graspgen.json` | Ultima esecuzione GraspGen, schema StageReport v2 |
| `grasp_graspgen_candidates.json` | Tutti i candidati restituiti, ordinati globalmente per score decrescente |
| `grasp_graspgen_input_centered.ply` | Nuvola inviata dopo il cap, ASCII arrotondato al micrometro |
| `grasp_graspgen_input_centered.npy` | Array float32 esatto inviato al server |
| `grasp_graspgen_overview.png` | Preview CPU dei candidati e del TCP scelto |

Usare cartelle `--out-dir` diverse per conservare ogni esperimento. Un
`--report` personalizzato cambia l'interfaccia di quella chiamata e conserva
comunque la copia per backend. I report includono configurazione risolta,
parametri, metadata server, revisione, hash del PLY, hash dei pesi/config,
centro e frame, indice originale server, punteggio, pose base/world originali
e TCP, selezione massima e controlli effettuati/non effettuati. Lo score del
discriminatore non è una misura validata del successo fisico nel nostro setup.

Il checkpoint deve corrispondere a `model_gripper=robotiq_2f_140`; altri
checkpoint o metadata incompatibili sono rifiutati prima dell'invio della
nuvola. Questa integrazione definisce TCP e visualizzazione solo per la
Robotiq 2F-140, pur esistendo altre pinze nel framework ufficiale.
Per un server ufficiale avviato senza il nostro wrapper si può impostare
`server.require_provenance: false`: rimangono obbligatori i controlli base su
pinza/modello/configurazione, e l'assenza di provenienza blocca il movimento.
Per la prima validazione usare il wrapper e mantenere `true`.

**La pinza `visual_parallel_jaws` attuale da 80 mm è diversa dalla Robotiq.**
Il movimento GraspGen richiede tutti i controlli seguenti: richiesta esplicita
in configurazione, stessa pinza, compatibilità verificata, trasformazione TCP
verificata, evidenza documentata, profondità/extrinseci verificati nel report
e provenienza server. `MotionPreview` verifica inoltre la propria pinza
reale di scena: quella attuale impedisce comunque di eseguire prese Robotiq.
Non basta cambiare booleani nello YAML per rendere la demo un robot Robotiq.
Un executor futuro compatibile dovrà usare `T_base_tcp` e
`require_motion_allowed`, oltre ai propri controlli di traiettoria/collisione.

Né la deproiezione simulata né quella reale marcano oggi la geometria come
verificata per il movimento GraspGen. La pipeline reale usa una **profondità
nominale costante derivata da una foto**, non una misurazione RGB-D. Le zone
chirurgiche A/B/C del PCA non vengono attribuite ai candidati GraspGen.
Collisioni di scena, IK, presa sul tessuto/strumento, apertura effettiva,
traiettoria e contatto fisico rimangono da verificare separatamente.
`physical_grasp_validated` rimane sempre `false`, anche quando la preview riesce.

## Verifica consegnata e lavoro GPU residuo

Sul Mac: suite completa, test ZMQ con risposte simulate, conversioni metriche,
ricentramento, pose con rotazioni e offset, base non identità, ordinamento,
report/snapshot separati, errori server, mismatch checkpoint, risposte vuote,
NaN/matrici non rigide, timeout metadata/inferenza, import/bundle trasferibile,
flag launcher e pipeline completa headless con movimento bloccato.
Preflight di commit e percorsi, sintassi shell e generazione della ricetta
Docker corretta sono controllabili senza GPU. Vedere anche
`docs/graspgen-verification.md` per il resoconto dell'esecuzione locale.

Sul PC restano da eseguire: download completo e controllo dei pesi reali,
build CUDA e smoke test, demo ufficiale diffusion, confronto sugli stessi dati
nel backend, nuvole degli strumenti, pipeline con SAM3 e server reale,
visualizzazione nativa e successiva calibrazione/hardware compatibile.
Questi controlli Mac verificano l'integrazione e il contratto del protocollo;
non provano inferenza reale né riproducono risultati sperimentali del paper.

Riferimenti: [codice ufficiale fissato](https://github.com/NVlabs/GraspGen/tree/2dd8852e1be60f5f9d277fafcc621835cdf59110),
[checkpoint Robotiq](https://huggingface.co/adithyamurali/GraspGenModels/blob/main/checkpoints/graspgen_robotiq_2f_140.yml),
[paper v1](https://arxiv.org/abs/2507.13097v1), letto anche dal PDF locale
`../Paper/2507.13097v1.pdf` (sezioni 3.1–3.2: generazione diffusion, ricentramento,
valutazione con discriminatore).
