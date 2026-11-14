# GraspGen e trasferimento KUKA in PyBullet

Questa modalità collega il modello GraspGen ufficiale al progetto locale.
Non è una riproduzione del sistema robotico del paper e non abilita l'hardware.

## Fonti dei tutor e cosa viene riusato

| Fonte | Uso nel progetto |
| --- | --- |
| [GraspGen, revisione fissata](https://github.com/NVlabs/GraspGen/tree/2dd8852e1be60f5f9d277fafcc621835cdf59110) | Generatore, discriminatore, checkpoint e convenzioni base pinza/+Z. Codice ufficiale invariato. |
| [Robotiq description](https://github.com/robotiq/robotiq-ros/tree/5b7f9189cb26197a4f6850937a1ee4ce667a3577/grippers/robotiq_description) | Xacro, giunti mimic e mesh di collisione 2F-140. Le stesse mesh sono usate per visualizzazione e verifiche. |
| [Piattaforma tutor: interfacce](https://github.com/jackysqing-max/robot-visual-simulation-platform/blob/74ee2e67d795b6a90c903e1226f918d333fee021/docs/interfaces.md) | Convenzioni XYZW, frame di link/flangia distinti dal TCP, modello IK separato dalla fisica, verifica FK, limiti e feedback. Adattati i riferimenti di velocità 0,04 m/s, 0,20 rad/s e 0,35 rad/s; soglia 0,25 rad per discontinuità IK e verifica finale 1 mm / 0,005 rad. Non viene eseguito il suo nodo ROS/JTC in questa modalità nativa. |
| [Piattaforma tutor: piano studenti](https://github.com/jackysqing-max/robot-visual-simulation-platform/blob/74ee2e67d795b6a90c903e1226f918d333fee021/docs/student_plan.md) | Separazione percezione, proposta, fattibilità ed esecuzione; provenance della cattura e motivi dei rifiuti. M4/M5 richiedono prove ancora da eseguire. |
| `llm_ros` / `sam3_ros` | Le revisioni scaricate coincidono con quelle già fissate in `ros2/workspace.repos`. I rispettivi ambienti restano separati dal modello GraspGen. La prova diagnostica usa parser offline e maschera ground-truth. |
| [LBR Stack Humble](https://github.com/lbr-stack/lbr_fri_ros2_stack/tree/52f816bd59cbd2390efc890cca0ad61389e4ccb3) | Fonte dell'interfaccia ROS di traiettoria e futura integrazione FRI. Il backend nativo non è un driver del KUKA reale. |

La piattaforma tutor dichiara esplicitamente che i demo del braccio non includono
un planner anticollisione completo o una strategia di contatto della pinza.
Il suo Workbench supporta utensili rigidi; non fornisce già una presa articolata
Robotiq completa. Collegare solo i topic non risolve questi aspetti.

## Preparazione degli asset

Su questo PC le repository selezionate sono in `data/external/`; revisioni e URL
sono registrati in `data/external/tutor-sources.json`. Le licenze originali rimangono
nei checkout. Il submodule GraspGen non è stato aggiornato.

Il preparatore richiede `requirements-robot-description.txt` (Xacro 2.1.1), espande
una copia temporanea degli Xacro Robotiq, risolve i riferimenti dei package e genera
`data/external/robotiq-prepared/robotiq_2f_140.urdf` e `description.json`.
Le DAE visuali sono sostituite con le corrispondenti mesh di collisione ufficiali
per evitare differenze di supporto DAE in PyBullet. Gli originali non sono modificati.
La tabella apertura/angolo deriva dalla FK statica degli URDF, non da un rapporto
lineare inventato. Il preparatore non apre un client fisico o un server.

Per riprodurre gli asset dopo aver scaricato il checkout Robotiq alla revisione
indicata sopra:

```bash
.venv/bin/python -m pip install -r requirements-robot-description.txt
.venv/bin/python -m pipeline.robot_assets
.venv/bin/python scripts/prepare_robotiq_description.py
```

Il frame radice del modello preparato coincide con la convenzione GraspGen
(chiusura X, avvicinamento Z). Il TCP del progetto è a [0,0,0.195] m con rotazione
-90 gradi attorno a Z: chiusura Y, avvicinamento Z. Il montaggio visuale usa
l'inversa di questa rotazione. FK/IK e collisioni condividono questa convenzione.
Il TCP è il riferimento nominale GraspGen, non una calibrazione sperimentale.
Il braccio è l'iiwa 7 R800 ufficiale, condiviso con il validatore e la cinematica;
origine, flangia, limiti e verifica sono descritti in [robot-model.md](robot-model.md).

## Esecuzione manuale

Terminale 1, dalla root `/home/alice/Projects/ROS2Project`:

```bash
bash scripts/graspgen_demos.sh server
```

Lasciare aperto il server dopo `Model loaded and ready for inference` e
`GraspGen ZMQ server listening`. Nessun altro container deve occupare 5556/8080.

Terminale 2, dalla stessa root:

```bash
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python run_pipeline_sim.py \
  --gui --mode single --offline --vision ground-truth \
  --tool "Bone Forceps" \
  --config /home/alice/Projects/ROS2Project/config/graspgen_simulation_forceps.yaml \
  --grasp-backend graspgen --graspgen-sim-motion \
  --graspgen-config /home/alice/Projects/ROS2Project/config/graspgen_simulation.yaml \
  --out-dir "/home/alice/Projects/ROS2Project/data/outputs/graspgen_pickplace_$(date +%Y%m%d_%H%M%S)"
```

La configurazione separata indica una mesh 3D-COSI Bone Forceps SmoothFusion
esistente. Nessun fallback sintetico è ammesso. Un output già contenente una run
viene rifiutato. Per osservare soltanto, aggiungere `--no-motion`.
Q esce dalla finestra; P mette in pausa l'animazione; R ripete la pianificazione
nella stessa scena ripristinata. G mostra/nasconde le altre proposte.

## Selezione ed esecuzione

Vengono esaminati i candidati in ordine di score della proposta originale.
Il modello GraspGen non predice l'apertura. `grasp_refinement.py` cerca una
correzione CAD limitata a 40 mm verso l'alto e 15 mm lungo l'asse di chiusura,
senza cambiare rotazione. Usa la geometria articolata effettiva: entrambi i
polpastrelli devono trovarsi entro 1,5 mm dal target, senza penetrazione, e tutta
la chiusura deve rispettare i margini dell'ambiente. Posa originale, correzione,
apertura e distanze sono conservate in `geometric_refinement` della selezione.
Il punteggio originale non è un nuovo punteggio della posa corretta.
Il pregrasp è arretrato di 12 cm lungo l'asse di avvicinamento del candidato.
Non si impone più un filtro arbitrario di 30 gradi dalla verticale: orientamenti
laterali possono comunque fallire per collisioni o raggiungibilità.

Il profilo `graspgen_simulation.yaml` mantiene i 200 campioni già generati per
batch anziché troncarli ai primi 100: non aumenta il batch GPU. I pesi,
il planner diffusion e le pose restano quelli ufficiali. Lo score non garantisce
raggiungibilità o assenza di collisioni con i vassoi.

Il validatore riutilizza un solo modello della cella. Controlla prima la pinza
alla posa candidata e durante la chiusura, poi cerca IK con più inizializzazioni.
La discesa viene risolta a ritroso dalla presa (4 mm / 0,03 rad tra waypoint),
per conservare il ramo IK. Avvicinamento e trasferimento usano una connessione
nello spazio dei giunti; se collidente, una ricerca RRT-Connect limitata a 4 s.
Questo evita di imporre una lunga linea cartesiana dalla HOME che può incastrare
l'IK differenziale ai limiti articolari. La risalita dalla posa di deposito
arretra lungo l'asse della pinza prima del rientro.

I waypoint vengono temporizzati con spline quintiche, velocità e accelerazione
nulle agli estremi e verifica numerica di velocità, accelerazione e jerk.
Il controllo delle collisioni viene ripetuto sui campioni temporizzati: una spline
che esce dal corridoio valido non viene eseguita. Non è una implementazione cuRobo.
Un budget di 30 s per candidato e 180 s complessivi limita la pianificazione;
scadenza del budget significa ricerca incompleta, non prova di irraggiungibilità.
La validazione controlla la stessa traiettoria eseguita,
ogni quattro tick da 1/240 s e agli estremi, oltre ai limiti dei giunti.
I controlli runtime usano anche i giunti misurati a 60 Hz. Un errore ferma
l'avanzamento e comanda di mantenere la posizione. Nessun candidato ammesso
significa nessun movimento; gli scarti vengono conservati.

La scena e il validatore usano gli stessi link Robotiq e una rappresentazione
convessa della mesh dello strumento, non più un box globale per questo percorso.
La geometria convessa può rifiutare prese attraverso aperture/concavità reali.
I polpastrelli possono toccare il target alla chiusura ideale, senza concedere
penetrazione. I vassoi conservano un margine minimo di 0,5 mm ai polpastrelli.
La pianificazione precede l'animazione: il candidato in esame è blu, con fase
e tempo trascorso. Q interrompe la ricerca. Il log e il report aggiornato a ogni
scarto includono fase, motivo e tempo speso.
I controlli campionati non certificano assenza continua di collisioni.

**Le dita seguono cinematicamente i giunti mimic.** Non c'è un controllore di forza
validato: il trasporto usa ancora il vincolo fisso ideale del progetto. Le collisioni
fisiche pinza-target sono disabilitate nella scena in questa modalità ideale per
non presentare il reset cinematico delle dita come un esperimento di contatto;
il modello separato continua a verificarne le distanze geometriche.

## Output e limiti

Oltre ai report della pipeline:

- `graspgen_sim_motion_validation.json`: run/cattura, hash della nuvola, revisioni
  del modello Robotiq, scarti, candidato e apertura selezionati, eventuale errore runtime;
- `graspgen_sim_joint_path.npz`: campioni articolari della traiettoria ammessa;
- `grasp_simulation_selection.json`: report GraspGen con `simulation_selected`;
- `sim_robot_motion.json`: tracking, deposito e risultato del trasporto ideale;
- `sim_robot_grasp.png` / `sim_robot_result.png`: scena con presa e scena finale;
- `graphical_run.json`: stato complessivo, inclusi fallimenti.

`selected` nel report GraspGen resta il massimo score. `simulation_selected` è
la posa derivata ammessa al movimento: verde nella GUI, con etichetta CAD refined.
L'array `candidates` conserva le proposte originali. In caso di rifiuto si mostra solo
il massimo score in arancione, esplicitamente non autorizzato al movimento.
`motion.allowed` e `physical_grasp_validated` restano falsi.

Il successo del trasporto ideale non valida attrito, contatti, forza, stabilità,
zone funzionali chirurgiche o l'esecuzione sul KUKA reale. SAM3/Ollama e i nodi ROS
richiedono preparazione e validazione separate; non sono implicitamente attivati
in questa prova diagnostica.

## Diagnosi della run 20261113_120843

Il report originale è conservato: 55 candidati scartati per stallo dell'IK
differenziale durante l'avvicinamento, 22 per collisione/margine insufficiente,
23 perché la sezione della mesh non attraversava entrambi i lati della pinza.
Non si trattava di un'inferenza GraspGen fallita. Queste modifiche correggono la
strategia di pianificazione e la diagnostica; non certificano che una delle
nuove pose sarà eseguibile prima della prova manuale.

Nel [paper, §4.5](https://arxiv.org/html/2507.13097v1#S4.SS5), gli autori passano
un insieme di pose a cuRobo, che applica collisioni e IK. GraspGen non costituisce
il controllore del braccio. La logica di campionamento cartesiano e verifica FK
è confrontata anche con `data/external/tutor-simulation-platform/src/`
`lbr_pybullet_tutorial/lbr_pybullet_tutorial/planning.py` alla revisione già fissata.

## Stato di validazione

Durante questa modifica sono stati preparati asset, ispezionati sorgenti e
controllata la sintassi. Per richiesta dell'utente non sono stati avviati server,
inferenze, simulazioni o test automatici. Le nuove geometrie e il percorso completo
restano da verificare manualmente. `tests/test_graspgen_sim_motion.py` contiene
controlli di regressione predisposti ma non eseguiti.
