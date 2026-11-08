# Avvio grafico sul Mac

Apri `Avvia_simulazione.command` con doppio clic, oppure esegui dal Terminale:

```bash
cd /home/ricca/progetto_ROS
bash run_graphical.sh
```

Si apre una finestra PyBullet nativa con KUKA LBR iiwa, pinza visuale, due vassoi
e lo STL selezionato. La run calcola RGB-D, maschera, nuvola di punti e presa PCA,
poi mostra avvicinamento, sollevamento, trasferimento e rilascio.
La linea gialla indica il percorso previsto e quella verde il TCP misurato.
La scena rimane aperta dopo il risultato. Usa i controlli PyBullet per orbitare
e ingrandire; premi P per mettere in pausa, R per ripetere il movimento concluso,
Q o Esc per chiudere. Il terminale riporta le fasi e gli eventuali errori.

## Due modalita' esplicite

L'avvio predefinito usa **SAM3** per la segmentazione e prova a interpretare
l'istruzione con Ollama. Il parser di riserva dell'LLM e' registrato nel piano JSON.

Per selezionare lo strumento:

```bash
bash run_graphical.sh --tool Scissors
```

Servono torch, transformers, i pesi SAM3 gia' nella cache Hugging Face e,
per usare il modello linguistico, Ollama con `qwen2.5:3b`. Un errore SAM3 o
una confidenza sotto 0.15 viene mostrato nella finestra e interrompe il
movimento. Il parser di riserva dell'LLM viene registrato nel piano JSON.
La GUI resta reattiva mentre SAM3 lavora in un processo separato.

Per un test rapido senza inferenza AI:

```bash
bash run_graphical.sh --vision ground-truth
```

Questa modalita' e' **diagnostica**: il piano usa il parser deterministico e la
maschera proviene dall'identificativo dell'oggetto nel simulatore. Verifica la
catena geometrica e la visualizzazione; non misura LLM o SAM3.

Altri esempi:

```bash
bash run_graphical.sh --tool Scalpel
bash run_graphical.sh --no-motion
```

## Provare diverse posizioni della base

La posa del robot si configura in `environment.kuka_base`. Posizione in metri,
orientamento roll/pitch/yaw in radianti, entrambi rispetto al riferimento fisso
`world`. Tavolo, vassoi e camera restano nelle loro posizioni nel mondo.

Per conservare la configurazione predefinita, copia il file completo:

```bash
cp config/surgical_environment.yaml config/prova-base.yaml
```

Nel file copiato modifica i due campi, per esempio:

```yaml
environment:
  kuka_base:
    position_xyz_m: [0.04, 0.01, 0.0]
    orientation_rpy_rad: [0.0, 0.0, 0.12]
```

Questo esempio sposta il robot di 4 cm lungo X e 1 cm lungo Y e lo ruota di
circa 6,9 gradi attorno a Z. Mantieni gli altri campi del file completo.
Avvia ogni prova con:

```bash
bash run_graphical.sh --config config/prova-base.yaml
```

Per controllare rapidamente la sola catena geometrica e il movimento:

```bash
.venv/bin/python run_pipeline_sim.py --headless --vision ground-truth --tool Scissors --config config/prova-base.yaml
```

I parametri vengono letti all'avvio: dopo ogni modifica riavvia la prova.
Una configurazione che rende le pose irraggiungibili viene rifiutata dal
controllo IK prima di iniziare il trasferimento.

Le nuvole e le pose di presa sono espresse nella base robot; i waypoint della
simulazione sono in `world`. I JSON includono la posa della base e il riferimento
dei dati. Quando cambi base, riesegui `deproject` e `grasp` per rigenerare le prese.
L'altezza di approccio e quella di sollevamento restano quote Z del mondo,
anche se la base e' inclinata.

Per una camera fissa, `camera_overhead.extrinsics.position_world_frame_xyz_m`
specifica la posizione in world. Il vecchio nome `position_robot_frame_xyz_m`
resta accettato come alias per le configurazioni precedenti, con valori in world.

## Controllo manuale interattivo: W-S-A-D e Puntatore del Mouse

Per muovere liberamente il robot sulla scena chirurgica con la tastiera (**W-S-A-D**) e visualizzare la scena dalla telecamera:

```bash
bash run_interactive.sh
```

- **W / S / A / D**: movimento cartesiano continuo (avanti, indietro, sinistra, destra);
- **Q / E**: solleva o abbassa la quota verticale $Z$;
- **U / O** (oppure **J / L**): ruota l'orientamento della pinza ($Yaw$);
- **Spazio**: apre e chiude le dita della pinza;
- **G** (oppure **Click Destro**): presa o rilascio dello strumento;
- **Rotella del mouse** (oppure tasto **M**): attiva/disattiva la modalità puntatore in cui il robot segue il mouse;
- **V / C**: passa dalla vista telecamera alla vista 3D libera;
- **H**: ripristina la posa iniziale HOME.


## Significato del movimento

La posa calcolata guida un TCP posto a 100 mm dalla flangia. Le pose intermedie
sono verificate con IK e limiti articolari, e i giunti sono comandati con i
servo di posizione di PyBullet. Gli errori misurati vengono salvati.

La pinza ha dita visuali mobili. Il trasporto dello strumento usa un
**vincolo fisso ideale**: non dimostra stabilita' dei contatti o forza di presa.
Il percorso non e' certificato senza collisioni e non viene inviato a ROS o
a un robot fisico. Si tratta di un'anteprima funzionale da usare per sviluppare
e controllare gli stage geometrici del progetto.

La mesh viene centrata, orientata sul piano e ridotta con decimazione quadrica,
senza sostituirla con un involucro convesso. Camera, report di posizionamento e
robot usano lo stesso strumento e la stessa posa. La maschera diagnostica include
solo i pixel visibili del corpo target.

## Risultati

In `data/outputs/` trovi:

- `graphical_run.json`: backend, stato (`data.state`), errore eventuale e riepilogo della run;
- `sim_robot_scene.png`: scena iniziale con il robot;
- `sim_capture_rgb.png`, `sim_capture_depth.npy`: acquisizione della scena (report `sim_capture.json`);
- `sim_segmentation_overlay.png`: maschera sovrapposta all'immagine (report `sim_segmentation.json`);
- `sim_deprojection_cloud.ply`: geometria nel riferimento robot (report `sim_deprojection.json`);
- `grasp.json`: proposta della presa, con verifiche ancora da svolgere (`data.readiness`);
- `sim_robot_motion.json`: waypoint, errori e modello di presa impiegato;
- `sim_robot_grasp.png`, `sim_robot_result.png`: presa proposta e risultato del movimento.

I singoli file descrivono l'ultima run. Una run fallita puo' lasciare artefatti
precedenti: `graphical_run.json` e' il riferimento per lo stato corrente.
Un blocco impedisce due run grafiche contemporanee nella stessa cartella.

Per la verifica senza finestra:

```bash
.venv/bin/python run_pipeline_sim.py --headless --vision ground-truth --tool Scissors
```

L'esecuzione senza finestra termina con codice diverso da zero se la percezione,
l'IK, il tracking o il deposito dimostrativo falliscono.

## GraspGen

Gli avvii grafici accettano anche `--grasp-backend graspgen` e le opzioni
`--graspgen-config`, `--graspgen-host`, `--graspgen-port`, `--graspgen-timeout-ms`.
PCA resta predefinito. Per GraspGen iniziare con `--no-motion`: vengono mostrate
le prese nominali della Robotiq, mentre la pinza robot dimostrativa da 80 mm
rimane incompatibile e il movimento è bloccato anche premendo R.
Preparazione del server GPU e comandi completi in [GRASPGEN.md](GRASPGEN.md).
