# Full system attraverso ROS 2

Questo percorso esegue una richiesta completa attraverso un coordinatore ROS 2 e un server di simulazione separato. L'istruzione e RGB-D viaggiano su topic DDS; le sette fasi del braccio e le due fasi della pinza vengono inviate come action ROS. La traiettoria è quella verificata dal planner GraspGen esistente.

Il modulo fa parte del repository: il codice è in `pipeline/ros2/`, l'entrypoint è
`run_system_ros2.py` e l'avvio del runtime è `scripts/run_ros2_system.sh`.
Le modifiche si trasferiscono sull'altro PC con il normale commit/push/pull del progetto.

Il backend di esecuzione è un **server di simulazione PyBullet** che implementa `FollowJointTrajectory` e `GripperCommand`. Non è un nodo `ros2_control`, un driver FRI o un collegamento al KUKA fisico. Scena, FK/IK e validatore usano lo stesso **iiwa 7 R800 ufficiale**, definito in `config/robot.yaml`, con Robotiq ufficiale e presa con vincolo ideale. [Modello, flangia, TCP e verifiche](robot-model.md). Il passaggio al robot fisico richiede un backend e una validazione dedicati.

## Avvio sull'altro PC Linux / WSL2

Eseguire dalla radice del repository. Docker deve essere disponibile. Mantenere i dati chirurgici in `data/inputs/Surgical Instruments/` e i pesi SAM3 preparati in `data/external/sam3-prepared/`, come nella run precedente. Il server GraspGen e Ollama restano quelli già usati sul PC.

Prima preparazione del nuovo runtime:

```bash
git pull --ff-only
git submodule update --init third_party/GraspGen
bash scripts/run_ros2_system.sh build
bash scripts/run_ros2_system.sh prepare
bash scripts/run_ros2_system.sh self-test
```

La build crea l'immagine `surgical-ros2-system`: ROS 2 Humble, Python compatibile con `rclpy`, simulazione, client GraspGen e SAM3 CPU. I pesi e gli strumenti non sono copiati nell'immagine. `prepare` scarica i checkout iiwa 7 e Robotiq alle revisioni fissate, se assenti, e genera URDF/metadata con percorsi relativi, utilizzabili sia sull'host sia in `/workspace`. Non aggiorna un checkout esistente a una revisione diversa. Dopo il pull di questa modifica eseguire nuovamente `prepare`, anche se la Robotiq era già pronta.

Avviare i servizi AI già predisposti, se non sono in funzione:

```bash
# Terminale 1: Ollama; lasciare aperto se lo script avvia il servizio
bash scripts/run_ollama.sh
```

```bash
# Terminale 2: GraspGen sulla GPU NVIDIA
bash scripts/graspgen_demos.sh server
```

Nel terminale del sistema:

```bash
bash scripts/run_ros2_system.sh --check
bash scripts/run_ros2_system.sh --gui \
  --instruction "Pick up the Bone Forceps from tray 2 and place it in tray 1."
```

Per una prova senza finestra:

```bash
bash scripts/run_ros2_system.sh --headless \
  --instruction "Pick up the Bone Forceps from tray 2 and place it in tray 1."
```

Gli avvii successivi richiedono soltanto i servizi AI e l'ultimo comando. Il launcher non avvia o arresta Ollama/GraspGen. La finestra si chiude quando la missione e la verifica del deposito terminano; Q/Esc annullano la run. Il manifest finale viene scritto anche in caso di errore o interruzione.

Il container usa la rete host su Linux/WSL2, quindi i default `127.0.0.1:11434` e `127.0.0.1:5556` raggiungono i servizi del PC. Per un Ollama remoto usare `--ollama-url`; per GraspGen adattare `server` in `config/graspgen_simulation.yaml`. Per un SAM3 in altra directory passare `--sam3-model /workspace/<directory-relativa-al-repository>`.

Su WSLg il launcher monta i socket grafici disponibili. Su Linux desktop usa DISPLAY e il file Xauthority esistenti. In assenza di un display accessibile usare `--headless`.

Non eseguire il nuovo nodo ROS con una `.venv` Python 3.11 creata separatamente dai binding Humble: il container usa un solo interprete compatibile. La pipeline nativa precedente rimane disponibile tramite `run_system.py`.

## Cosa accade durante una run

1. Il preflight controlla import ROS, asset e hash del modello iiwa 7, modello Robotiq, Ollama, SAM3 e provenance del server GraspGen. Un controllo fallito termina prima del movimento.
2. La richiesta CLI viene pubblicata sul topic `instruction` e ricevuta dal coordinatore. Il piano LLM resta sottoposto ai contratti strict del profilo completo.
3. Il processo di simulazione crea la scena e acquisisce RGB-D. Pubblica colore, profondità allineata e `CameraInfo` con lo stesso timestamp/frame.
4. Il coordinatore riceve i pixel via DDS e controlla dimensioni, formato, intrinseche e identità con la cattura della scena. Questi pixel diventano gli input della segmentazione/ricostruzione. La maschera ground truth rimane diagnostica.
5. Il worker separato esegue SAM3, ricostruzione e client GraspGen. Le inferenze non occupano il thread che riceve messaggi ROS. Non sono introdotti fallback ai parser o alle maschere del simulatore nel profilo completo.
6. Il simulatore seleziona e corregge geometricamente una presa, verifica IK/limiti/collisioni e conserva in memoria tutte le fasi autorizzate. Esporta lo stesso NPZ articolare della pipeline nativa, insieme a un indice con hash e ID della run.
7. Il coordinatore invia una action per ogni fase, rispettando l'ordine. Il server accetta solo nomi giunti, campioni, tempi e apertura della fase corrente. Controlla anche lo stato articolare iniziale misurato. Traiettorie alterate, vecchie o fuori ordine vengono rifiutate.
8. Il servo PyBullet esegue i campioni autorizzati. Lo stato misurato viene pubblicato a 60 Hz; l'action restituisce feedback e risultato. La fase successiva parte solo dopo il successo della precedente.
9. Dopo il rientro vengono verificati assestamento e deposito. Il coordinatore registra evidenze ROS e chiude il manifest.

La percezione riusa le librerie del progetto, Ollama via HTTP e GraspGen via ZMQ. I package esterni `llm_ros` e `sam3_ros` non vengono dichiarati attivi: il coordinatore ROS chiama il worker esistente e ne pubblica gli output. Le trasformazioni e le proposte sono pubblicate anche come messaggi ROS per ispezione.

## Interfacce della run

Ogni run ha un namespace esclusivo, stampato all'avvio e salvato nel manifest, ad esempio `/surgical_sim/run_<id>`. Nessun nome di comando punta ai topic del robot fisico.

| Suffisso nel namespace | Tipo | Contenuto |
|---|---|---|
| `instruction` | `std_msgs/String` | Istruzione della singola richiesta |
| `task`, `status`, `result` | `std_msgs/String` | Task JSON, avanzamento ed esito |
| `camera/color` | `sensor_msgs/Image` | Snapshot RGB `rgb8` |
| `camera/depth` | `sensor_msgs/Image` | Snapshot allineato `32FC1`, metri |
| `camera/info` | `sensor_msgs/CameraInfo` | Intrinseche della stessa acquisizione |
| `grasp_pose`, `grasp_candidates` | `geometry_msgs/PoseStamped`, `PoseArray` | TCP selezionato e proposte in base robot |
| `joint_states` | `sensor_msgs/JointState` | Posizioni e velocità misurate dei sette giunti |
| `tcp_pose` | `geometry_msgs/PoseStamped` | Posa TCP misurata nel mondo |
| `arm/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | Una fase articolare validata |
| `gripper/gripper_cmd` | `control_msgs/action/GripperCommand` | Apertura totale della pinza, metri; effort 0 perché ideale |
| `initialize`, `plan`, `finish`, `stop` | `std_srvs/Trigger` | Coordinamento interno della scena |

TF pubblica world → base, world → frame ottico e world → TCP. I frame sono prefissati con l'ID della run per evitare collisioni con altri simulatori. I report geometrici mantengono il frame logico `lbr_link_0`.

Il server accetta soltanto traiettorie di posizioni con timestamp iniziale zero e temporizzazione identica ai campioni del planner. Non supporta traiettorie arbitrarie, goal pianificati nel futuro o tolleranze personalizzate: le tolleranze server sono 0,05 rad lungo il percorso e 0,01 rad all'arrivo, oltre ai controlli geometrici esistenti. Una cancellazione o un errore blocca la run; non riprende automaticamente.

## Risultati e criterio di completamento

Output in `data/outputs/ros2_full_system_<data>_<id>/`:

- `ros2_request.json`, `environment.yaml`, `runtime_versions.json`: richiesta, configurazione effettiva e dipendenze; il manifest include seed, commit e stato della working tree.
- Report di percezione e GraspGen già presenti nella pipeline nativa.
- `ros2_execution_plan.json`: fasi, ID, hash del NPZ validato e descrizione/hash del robot.
- `ros2_measured_joint_states.json`: campioni ricevuti realmente su ROS.
- `ros2_run.json`: sette action braccio, due action pinza, feedback e trasporto RGB-D.
- `sim_robot_motion.json`, immagini finali e log del processo `ros2_simulation.log`.
- `run_manifest.json` e `summary.md`: esito finale.

`COMPLETED` richiede movimento e deposito verificati, evidenza di GraspGen e, nel profilo completo, Ollama e SAM3; richiede inoltre RGB-D ricevuto via ROS, tutte e nove le action riuscite e feedback misurato. Il completamento delle action da solo non certifica il deposito.

`motion.allowed` per l'hardware e `physical_grasp_validated` rimangono falsi. L'autorizzazione separata vale solo per la scena simulata corrente. La correzione CAD e le collisioni campionate conservano gli stessi limiti della demo precedente.

## Uso senza lo script Docker

Dentro un ambiente Humble con le dipendenze del nuovo runtime:

```bash
python3 run_system_ros2.py --check
python3 run_system_ros2.py --headless
```

L'avvio Python non richiede una build colcon. Per esporre anche il launcher tramite il package ROS già presente:

```bash
bash scripts/ros2_build.sh
source ROS2_workspace/install/setup.bash
ros2 launch surgical_executor full_system.launch.py gui:=true
```

`pipeline` deve essere nel PYTHONPATH; l'entrypoint del container lo imposta a `/workspace`.

## Verifica del collegamento

`self-test` usa ROS 2 reale e un servo PyBullet reale, con piccole traiettorie di fixture: verifica camera DDS, sette action braccio, due pinza, feedback, rifiuti e arresto. Non carica modelli AI e non dichiara riuscita una presa fisica.

Per verificare anche il planner geometrico e il deposito con i risultati neurali già registrati, dentro il runtime:

```bash
python3 -m tests.ros2_recorded_run \
  --source-run data/outputs/full_system_20261114_111550_9145ef8f \
  --robot-model iiwa7_r800 \
  --out-dir data/outputs/ros2_iiwa7_check_01
```

La directory destinazione deve essere nuova. La prova controlla posa, pixel e profondità dello strumento registrato, riusa soltanto gli output neurali e ricalcola selezione, IK, collisioni e tutte le traiettorie sull'iiwa 7. Marca il manifest `validation_mode=recorded_proposals_replanned_for_iiwa7`: è una nuova esecuzione ROS/planner/fisica, non una nuova inferenza AI. Omettendo `--robot-model`, il test riproduce esplicitamente il modello della registrazione (iiwa 14 per le vecchie run), con controllo dell'intera cattura. Il launcher di produzione non contiene questo adattatore di replay.

Verifica del 14 novembre 2026: 189 test nell'ambiente Linux/Humble, tutti superati, inclusi confronti FK/IK
indipendenti per iiwa 7 e i due TCP. La run `ros2_iiwa7_validation_20261114_2` ha ripianificato le proposte
registrate sul nuovo modello e completato il deposito, con 7 action braccio e 2 pinza riuscite e feedback
misurato ricevuto via ROS. La nuova inferenza con tutti i modelli e la finestra grafica vanno provate sul PC di destinazione.

Riferimenti delle interfacce: [FollowJointTrajectory / controller ROS](https://control.ros.org/humble/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html#actions),
[definizione GripperCommand Humble](https://github.com/ros-controls/control_msgs/blob/humble/control_msgs/action/GripperCommand.action).
