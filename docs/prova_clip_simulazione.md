# Prova della presa dalla clip nel simulatore

Questa prova usa una registrazione RGB-D e termina in PyBullet. **Nessun comando è inviato al robot fisico.** Il modello dell'oggetto è una approssimazione derivata dalla sola superficie visibile: non è una ricostruzione completa delle forbici e non prova una presa fisica.

## Dati di partenza

Nel frame 300 Qwen ha prodotto il target `surgical scissors` dall'istruzione esplicita. SAM 3 ha segmentato lo strumento ad anelli sulla destra; la maschera e la profondità hanno prodotto 1.261 punti. GraspGen ha generato un nuovo insieme di 200 candidati a partire dalla stessa nuvola, usando il generatore/discriminatore Robotiq 2F-140. I report dell'inferenza sono in `data/outputs/clip_graspgen_200/`. Non è stata eseguita di nuovo la segmentazione durante la prova simulata: sono stati riutilizzati i risultati registrati.

## Come è stato costruito il collegamento

`scripts/test_clip_sim.py` legge il report originale della cattura reale, la nuvola e le prese, verifica l'hash del PLY e la provenienza del server e non modifica i report sorgente. Il simulatore ordinario continua a richiedere una cattura simulata: l'adattatore per clip è separato e non abilita l'esecuzione su hardware.

1. La nuvola salvata e le prese erano nel riferimento nominale usato dalla pipeline reale. Per ritornare al frame ottico si usa l'inversa della matrice camera→base salvata nella cattura.
2. Si applica `T_world_from_color_nominal` della clip, ricavata dal piano della registrazione. Poi si trasla l'intera scena in modo da posizionare lo strumento sul vassoio sorgente simulato. **La stessa trasformazione rigida viene applicata a tutti i punti, alle basi pinza e ai TCP dei candidati.** La matrice completa è nel manifest.
3. La nuvola visibile viene convertita in un **involucro convesso** chiuso. Il modello riempie anelli e concavità e inventa il volume nascosto; è pertanto un oggetto geometrico ausiliario, non le forbici reali o la loro mesh CAD. Nel risultato iniziale appare come un cuneo.
4. La cella usa il KUKA iiwa 7 e una Robotiq 2F-140 nominale. La ricerca verifica geometria di chiusura, IK, limiti dei giunti e collisioni campionate durante le traiettorie contro robot, tavolo e vassoi modellati. Scene e ostacoli della clip diversi dall'oggetto non sono stati ricostruiti.

La scala di profondità `0.001` m/unità resta un'ipotesi; la distorsione della camera non è corretta e posa del piano, posizione dei vassoi e TCP sono nominali o ipotetici. Questo esercizio valuta una scena sintetica coerente con i dati, non la reale disposizione della cella.

## Esito misurato

- Le prime 100 prese originali del frame 300 e le 100 del frame 0 sono state tutte rifiutate dalla verifica geometrica. La prova con un nuovo insieme di 200 candidati ha raggiunto la pianificazione del trasferimento per alcune pose; una è stata rifiutata solo durante il ritorno finale alla posa di parcheggio.
- Una seconda esecuzione ha omesso **solo** il ritorno facoltativo a quella posa, mantenendo il controllo dell'allontanamento dopo il rilascio. Il candidato `server_index=194`, il 195° in ordine di punteggio, è stato ammesso dopo 194 scarti. Il suo punteggio originale era circa 0,0867 e la posa è stata corretta geometricamente di 30 mm verso l'alto; il punteggio non è stato ricalcolato dopo la correzione.
- Il robot simulato ha eseguito avvicinamento, discesa, chiusura, sollevamento, trasferimento, deposito, apertura e allontanamento. Il proxy è risultato dentro il vassoio di destinazione, in contatto e fermo. La durata simulata del moto è circa 44,6 s.
- I giunti della pinza sono mossi cinematicamente e il trasporto usa un vincolo fisso ideale. Non sono stati verificati attrito, forza, presa stabile, precisione di calibrazione, zone chirurgiche o sicurezza della cella fisica. `physical_grasp_validated` e l'abilitazione hardware restano falsi.

Risultati: `data/outputs/clip_sim_300_200_retreat/clip_simulation_result.json`, `graspgen_sim_motion_validation.json`, `graspgen_sim_joint_path.npz`, `scene_initial.png`, `scene_final.png`, `clip_simulation_manifest.json`. I report contengono le singole ragioni degli scarti e la traiettoria verificata. Il video verificato si trova nell'esecuzione separata `data/outputs/clip_sim_300_video/clip_simulation.mp4` (46,6 s).

## Ripetere la prova

Da una root già predisposta con asset iiwa/Robotiq e risultati GraspGen salvati:

```bash
cd /home/alice/Desktop/ROS2Project/ROS2Project
MPLCONFIGDIR=/tmp/ros2project-mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/test_clip_sim.py \
  --run-dir data/outputs/clip_graspgen_200 \
  --out-dir data/outputs/clip_sim_nuova \
  --omit-return --video
```

Serve `ffmpeg` per il video. L'output deve essere una cartella nuova. Senza `--omit-return` si richiede anche il rientro alla posa di parcheggio: per questo insieme di prese la prova è fallita perché nessuna delle 200 ha superato tutte le fasi, incluso quel rientro. Senza `--video` non si registra il filmato.

La ripetizione sopra usa le 200 pose già salvate. Per generarne un nuovo insieme dal frame 300, con Ollama non necessario ma con il server GraspGen attivo, usare una nuova cartella:

```bash
mkdir -p data/outputs/clip_graspgen_nuovo
cp data/outputs/clip_scissors_gpu/frame_000300/{clip_input.json,plan.json,real_capture.json,real_deprojection.json,real_deprojection_cloud.ply} data/outputs/clip_graspgen_nuovo/
.venv/bin/python -m pipeline.grasp \
  --deprojection data/outputs/clip_graspgen_nuovo/real_deprojection.json \
  --plan data/outputs/clip_graspgen_nuovo/plan.json \
  --out-dir data/outputs/clip_graspgen_nuovo --grasp-backend graspgen \
  --graspgen-config config/graspgen_simulation.yaml
```

Poi passare `--run-dir data/outputs/clip_graspgen_nuovo` all'adattatore. GraspGen è stocastico: pose e risultato della pianificazione possono cambiare. I 12 test `tests.test_clip_sim` e `tests.test_graspgen_sim_motion` sono stati eseguiti e superati.
