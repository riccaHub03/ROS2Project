# Avvio grafico sul Mac

Apri `Avvia_simulazione.command` con doppio clic, oppure esegui dal Terminale:

```bash
cd /Users/alice/Desktop/Progetti/P6/ROS2Project
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

- `graphical_run.json`: backend, stato, errore eventuale e riepilogo della run;
- `sim_robot_scene.png`: scena iniziale con il robot;
- `sim_camera_rgb.png`, `sim_camera_depth.npy`: acquisizione della scena;
- `sim_camera_overlay.png`: maschera sovrapposta all'immagine;
- `sim_reconstructed_instrument.ply`: geometria nel riferimento robot;
- `stage5_grasp_geometry.json`: proposta della presa, con verifiche ancora da svolgere;
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
