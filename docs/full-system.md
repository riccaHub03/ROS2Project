# Full system: LLM → SAM3 → GraspGen → PyBullet

Due profili indipendenti, con lo stesso modulo di presa e movimento:

- `graspgen_only`: parser deterministico, maschera PyBullet, GraspGen e percorso verificato.
- `full_system`: Ollama obbligatorio, SAM3 obbligatorio, GraspGen e lo stesso percorso verificato. Nessun fallback a parser, ground-truth o PCA.

Entrambi eseguono una scena **single-object** e una presa con **vincolo fisso ideale**. La prima scena configurata contiene Bone Forceps. Questo modulo non introduce ancora corpi multioggetto, collegamenti ROS o validazione fisica della presa.

## 1. Preparazione locale

Dalla root del progetto, la `.venv` deve già contenere le dipendenze di simulazione e client GraspGen. Il profilo diagnostico non richiede PyTorch o SAM3.

Per il profilo completo installare la parte SAM3 CPU, separata dal container CUDA GraspGen. Le versioni seguenti seguono il checkout tutor `sam3_ros` fissato nel progetto; l’esecuzione locale Python 3.11 richiede comunque verifica effettiva del modello:

```bash
cd /home/alice/Projects/ROS2Project
.venv/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-system.txt
.venv/bin/python -m pip check
```

Il modello `facebook/sam3` richiede un account Hugging Face autorizzato e l’accettazione delle condizioni del modello. Dopo aver ottenuto accesso:

```bash
.venv/bin/hf auth login
.venv/bin/python scripts/download_sam3.py
```

Il runtime SAM3 legge solo file locali e usa la CPU. Non scarica pesi durante una run. Si può indicare una directory già disponibile con `--sam3-model /percorso/al/modello`.

Il downloader salva soltanto gli asset Transformers in `data/external/sam3-prepared`, evitando la seconda copia dei pesi `sam3.pt`. Il launcher rileva automaticamente la directory preparata.

Su questo PC Ollama è predisposto localmente in `.local/ollama`, senza sudo. Per riavviarlo dopo aver chiuso il servizio:

```bash
bash scripts/run_ollama.sh
```

Il binario locale si usa con `.local/ollama/bin/ollama`, per esempio `.local/ollama/bin/ollama list`. I modelli sono in `.local/ollama/models`; servizio, pesi e log sono esclusi da Git. Il servizio libera la RAM del LLM dopo la richiesta, prima di caricare SAM3.

Installare Ollama sull’host secondo la propria configurazione e predisporre il modello. Se il servizio non è già attivo, in un terminale:

```bash
ollama serve
```

In un altro terminale, solo se il modello non è già presente:

```bash
ollama pull qwen2.5:3b
```

Avviare GraspGen se non è già attivo:

```bash
cd /home/alice/Projects/ROS2Project
bash scripts/graspgen_demos.sh server
```

Attendere che il modello sia pronto. Il launcher controlla i servizi; non avvia né arresta processi esterni automaticamente.

## 2. Controllo senza inferenza o movimento

```bash
.venv/bin/python run_system.py --profile full_system --check
```

Verifica import, STL, URDF Robotiq/hash, disponibilità del modello Ollama, cache/import SAM3 e metadati/provenienza del server GraspGen. Restituisce codice nonzero se manca un requisito. Non crea una run e non dimostra qualità di segmentazione o corretto caricamento di tutti i pesi: quelli vengono verificati realmente durante lo stadio SAM3.

Per controllare solo il percorso diagnostico:

```bash
.venv/bin/python run_system.py --profile graspgen_only --check
```

## 3. Esecuzione

Solo GraspGen, con la scena predefinita Bone Forceps:

```bash
.venv/bin/python run_system.py --profile graspgen_only --gui
```

Sistema completo:

```bash
.venv/bin/python run_system.py --profile full_system \
  --instruction "Pick up the Bone Forceps from tray 2 and place it in tray 1." \
  --gui
```

Senza finestra:

```bash
.venv/bin/python run_system.py --profile full_system \
  --instruction "Pick up the Bone Forceps from tray 2 and place it in tray 1." \
  --headless
```

`--exit-after-run` chiude la finestra al termine. `--seed` controlla il posizionamento simulato; non promette determinismo dell’inferenza diffusion. Ogni avvio genera una nuova cartella `data/outputs/<profilo>_<data>_<id>/`. Con `--out-dir`, la cartella indicata deve essere inesistente. La run della registrazione resta intatta.

Parametri dei servizi:

```bash
.venv/bin/python run_system.py --profile full_system \
  --ollama-url http://127.0.0.1:11434/v1/chat/completions \
  --llm-model qwen2.5:3b \
  --sam3-model facebook/sam3 \
  --gui
```

Per un altro strumento fornire **sia** `--config` con la mesh della scena **sia** `--tool` con il suo nome canonico. Nel profilo completo il nome dichiara cosa contiene la scena; non sostituisce la scelta del LLM. Una richiesta di uno strumento diverso viene rifiutata. Le stringhe sinonime non ancora mappate devono essere estese esplicitamente nel contratto, senza normalizzare arbitrariamente il target.

## 4. Cosa succede

1. Il preflight verifica i requisiti del profilo.
2. Il task planner interpreta la frase. Nel profilo completo JSON e semantica sono validati: pick_and_place, target della scena, keyword coerente, vassoi conosciuti e distinti. Errori e risposte malformate interrompono la run.
3. PyBullet costruisce la scena e salva RGB-D e camera.
4. Il worker esegue SAM3 CPU oppure la maschera diagnostica secondo il profilo. SAM3 seleziona l’istanza con score massimo, senza unire tutte le istanze. Il profilo completo richiede score finito accettato e maschera non vuota.
5. La maschera e la depth diventano nuvola in base robot. Nel profilo completo il riferimento noto dello strumento non può correggere il centroide: una ricostruzione esterna al vassoio fallisce.
6. Il backend GraspGen esistente centra la nuvola, interroga il server e conserva candidati, score e trasformazioni.
7. Il medesimo modulo GraspGen della modalità diagnostica cerca una presa geometricamente ammissibile, stima l’apertura e pianifica i giunti con controlli campionati. Refinement e collisioni usano CAD/pose note della scena: il sistema non è puramente basato sulla sola osservazione RGB-D.
8. L’esecutore PyBullet realizza Approach → Descend → Close gripper - ideal constraint → Lift → Transfer → Place → Open gripper → Retreat → Return.
9. Il report distingue effettivo uso dei modelli, verifiche del movimento e deposito. Non certifica forza/attrito/stabilità della presa.

`R` ripete la pianificazione e il movimento sulla scena ripristinata usando la percezione già calcolata; non rilancia il LLM o SAM3. Per una nuova prova end-to-end avviare una nuova run. `P` pausa, `G` candidati, `Q` esce.

## 5. Output

- `run_manifest.json`: configurazione, ID della richiesta/run, preflight, stato finale, backend effettivi, evidenze e indice degli artefatti.
- `summary.md`: riepilogo leggibile e collegamenti agli output.
- `plan.json`: task e modello/endpoint Ollama effettivi.
- `sim_segmentation.json` e overlay: backend, modello, score e maschera.
- `sim_deprojection.json` e cloud: ricostruzione e frame.
- `grasp_graspgen.json`: proposte originali; `selected` è il massimo score.
- `grasp_simulation_selection.json`: `simulation_selected` è la presa derivata scelta per l’esecuzione.
- `graspgen_sim_motion_validation.json` e NPZ: scarti, correzione CAD, controlli e traiettoria.
- `sim_robot_motion.json`: esito eseguito, errori di tracking e deposito.

`COMPLETED` nel nuovo manifest richiede movimento completato, controlli superati e deposito verificato; in `full_system` anche evidenza effettiva di Ollama e SAM3. Il solo completamento della GUI non basta. `physical_grasp_validated` resta falso.

## 6. Struttura e compatibilità

- `pipeline/system/cli.py`: interfaccia utente.
- `pipeline/system/registry.py` e `config/system/`: composizione dei profili.
- `pipeline/system/preflight.py`: controlli indipendenti.
- `pipeline/system/contracts.py`: validazione del task, maschera e ricostruzione.
- `pipeline/system/orchestrator.py`: output esclusivo, delega all’esecuzione esistente e raccolta delle evidenze.
- `pipeline/plan.py`: flag opzionali strict; il comportamento legacy mantiene il fallback.
- `pipeline/sim/graphical.py`: riceve opzioni dal nuovo orchestratore e riutilizza worker, scena e movimento esistenti.

`run_pipeline_sim.py` e `scripts/graspgen_demos.sh` restano disponibili. Non vengono modificati il modello upstream GraspGen o i checkpoint. Non vengono introdotte dipendenze ROS nella venv Python 3.11.

Le convenzioni dei tutor restano quelle fissate in `data/external/tutor-sources.json` / `ros2/workspace.repos`. Gli adattatori topic/action `llm_ros`, `sam3_ros` e LBR Stack sono la successiva integrazione ROS: non vengono dichiarati già attivi da questo modulo nativo.

## 7. Validazione di questa implementazione

I test isolati verificano obbligatorietà LLM, validazione task, compatibilità del parser offline, profili sullo stesso backend GraspGen, assenza di esecuzione dopo preflight fallito, protezione degli output e criteri di successo del manifest.

Aggiornamento della preparazione locale: il preflight conferma tutti i controlli positivi (simulazione, asset, Ollama, SAM3, GraspGen). È riuscita una vera richiesta Ollama sul task Bone Forceps; SAM3 ha caricato i pesi verificati SHA-256 ed eseguito il prompt `forceps` sulla cattura della run registrata, con score 0.8895 e maschera di 928 pixel. La prova SAM3 è in `.local/sam3-smoke/`; il riepilogo del controllo servizi è in `.local/full-system-preflight.json`. Queste verifiche non includono ancora una nuova esecuzione completa del movimento con tutti i modelli attivi.

## Velocità della simulazione

Aggiungere `--motion-speed 1.5` oppure `--motion-speed 2` al nuovo launcher per aumentare i riferimenti di velocità del robot. Il valore predefinito resta `1`; intervallo consentito: `0.25–2`. Vale per entrambi i profili e non accelera LLM, SAM3 o la ricerca dei candidati.

Il fattore scala i riferimenti cartesiani (base 0.04 m/s e 0.20 rad/s) e quello articolare (base 0.35 rad/s), rispettando eventuali limiti inferiori della configurazione. Accelerazione, jerk, collisioni e tracking restano verificati sulla traiettoria temporizzata. Il planner può quindi produrre un aumento inferiore al fattore richiesto. Chiusura/apertura della pinza mantengono la durata originale. Il valore usato è registrato nei settings del `run_manifest.json`.
