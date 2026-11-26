# Dalla richiesta in linguaggio naturale al deposito simulato

## Scopo e risultato della prova

Questo documento descrive le prove eseguite sul PC del progetto il 25 e 26 novembre 2026, usando immagini registrate in `execution/clip/`. La sequenza verificata è:

```text
Istruzione dell'utente
        ↓
LLM Qwen 2.5 tramite Ollama → piano strutturato e nome dello strumento
        ↓
Coppia RGB-D della clip + intrinseche → SAM 3 → maschera 2D
        ↓
Maschera + profondità → deproiezione → nuvola di punti 3D
        ↓
GraspGen → proposte di presa con pose e punteggi
        ↓
Adattamento alla cella PyBullet → selezione e traiettoria verificata
        ↓
Presa ideale, sollevamento, trasferimento e deposito simulati
```

**Risultato:** sul frame 300 è stato individuato uno strumento ad anelli sulla destra dell'immagine; la pipeline ne ha ricostruito 1.261 punti. Da un insieme di 200 proposte GraspGen, una è stata ammessa dal pianificatore e usata per spostare un modello approssimato dello strumento fra due vassoi **in simulazione**.

**Confine della prova:** dal 26 novembre un singolo comando della pipeline reale può elaborare un frame della clip dall’LLM fino al deposito simulato opzionale. È stato eseguito sul frame 300; i frame 0 e 300 erano stati provati fino a GraspGen anche senza simulazione. Il deposito avviene esclusivamente in PyBullet e conserva risultati e ipotesi separatamente. Il modello spostato è una geometria convessa ricavata dalla nuvola, non una copia fedele dell'oggetto. La chiusura e il trasporto sono idealizzati. Nessuna camera o robot fisico era coinvolto e nessun comando di movimento è stato inviato al KUKA reale.

Il [video della prova simulata](../data/outputs/clip_sim_300_video/clip_simulation.mp4) dura 46,6 s. La [diagnosi dedicata alla simulazione](prova_clip_simulazione.md) conserva ulteriori dettagli e comandi.

## 1. Componenti e ambienti

| Componente | Ruolo nella prova | Ambiente e stato verificato |
|---|---|---|
| Clip D456 | Fornisce coppie colore/profondità registrate e metadati camera | `execution/clip/`; la camera fisica non era collegata |
| Qwen 2.5 3B | Converte l'istruzione in un piano JSON | Ollama locale, CPU, servizio su `127.0.0.1:11434` |
| SAM 3 | Segmenta nell'immagine l'istanza richiesta dal piano | `.venv`, PyTorch 2.10.0+cu128; provato sia CPU sia RTX 3060 |
| Geometria della repo | Converte pixel della maschera e profondità in punti 3D | `.venv`; modello pinhole e trasformazioni nominali |
| GraspGen | Propone pose di una pinza e le ordina tramite discriminatore | Server Docker separato, PyTorch 2.1.0+cu121, PointNet CUDA, porta 5556 |
| PyBullet | Modella iiwa 7, pinza Robotiq nominale, vassoi e proxy dell'oggetto | Esecuzione headless locale; collisioni e cinematica verificate nel modello |

Il PC usa Ubuntu 24.04 e una NVIDIA GeForce RTX 3060 Laptop da 6 GiB con driver 580.178.04. Il driver è stato verificato sull'host; `nvidia-smi` indica la versione CUDA massima supportata dal driver, non dimostra da solo la presenza del toolkit `nvcc`.

Gli ambienti sono separati perché SAM 3 e il server GraspGen richiedono stack diversi. Nel venv della camera, `.venv-camera`, sono stati installati `pyrealsense2==2.59.2.11152`, NumPy e OpenCV. L'import del runtime librealsense 2.59.2 è riuscito, ma sono stati rilevati **zero dispositivi**: acquisizione, firmware e permessi USB devono essere verificati con la camera collegata. È stata inoltre costruita un'immagine ROS separata, `surgical-ros2-realsense:local`, con `realsense2_camera` 4.59.2 e librealsense 2.59.2. Questo non dimostra che il flusso live funzioni; gli strumenti SDK host come RealSense Viewer non sono stati verificati.

## 2. Dati RGB-D e ipotesi iniziali

La clip contiene **617 coppie** `color_NNNNNN.png` e `depth_NNNNNN.png`, con indice corrispondente e risoluzione 1280×720. Le prove complete di percezione hanno usato i frame **0 e 300**; la simulazione del deposito ha riutilizzato il frame **300**.

`camera.yaml` dichiara che la profondità è allineata al colore (`depth_aligned_to_color: true`) e fornisce le seguenti intrinseche del colore:

| Parametro | Valore | Significato |
|---|---:|---|
| `fx` / `fu` | 647,0326 px | focale lungo l'asse orizzontale |
| `fy` / `fv` | 646,2188 px | focale lungo l'asse verticale |
| `cx` / `ppu` | 646,2775 px | coordinata orizzontale del punto principale |
| `cy` / `ppv` | 378,8295 px | coordinata verticale del punto principale |

Le profondità sono PNG `uint16`. Il file `camera.yaml` **non specifica la scala**: per la prova è stato scelto esplicitamente `0,001 m/unità`. Nel frame 0 la mediana grezza dei pixel non nulli è 764, pari a 0,764 m sotto tale ipotesi; il valore è plausibile, ma non prova la scala effettiva. Questa deve essere confermata dai metadati originali o dal sensore. Il fatto che RGB e profondità abbiano la stessa dimensione non dimostra, da solo, un allineamento geometrico corretto.

La clip dichiara anche una distorsione `inverse_brown_conrady` con coefficienti non nulli. La ricostruzione usata qui è pinhole e **non corregge la distorsione**. Per misure affidabili servono immagini e profondità rettificate con intrinseche coerenti, oppure una deproiezione che gestisca il modello di distorsione. `frames.json` riporta una posa del piano **nominale**, non una calibrazione mano-occhio certificata rispetto al robot.

### Ritaglio usato nella prova

Lo strumento occupa una piccola parte dell'immagine. Per i frame 0 e 300 è stata selezionata la regione `(x=550, y=195, larghezza=400, altezza=300)` in RGB e profondità, senza riscalare i pixel. Il ritaglio contiene più strumenti; la selezione dell'istanza è effettuata da SAM 3.

Quando si ritaglia, la focale in pixel non cambia, mentre il punto principale viene traslato:

```text
fx' = fx                       fy' = fy
cx' = cx - 550 = 96,2775 px   cy' = cy - 195 = 183,8295 px
```

La sorgente `--source clip` della cattura reale verifica dimensioni e tipo della profondità, applica la **stessa regione** a RGB e profondità, moltiplica i valori per la scala scelta e salva una matrice `float32` in metri. Il ritaglio RGB e la conversione della profondità sono stati controllati numericamente sui file originali.

## 3. LLM: dall'istruzione al piano

### Ingresso

Il test riuscito ha usato questa istruzione esplicita:

```text
Pick the surgical scissors from tray 2 and place them in tray 1.
```

### Elaborazione

`pipeline/plan.py` invia il testo a **Qwen 2.5 3B**, tramite l'API locale di Ollama. Il prompt di sistema chiede una risposta JSON con azione, nome preciso in inglese, parola chiave dello strumento, vassoio sorgente e vassoio destinazione. Il codice interpreta e normalizza i campi, poi salva `plan.json`.

Per questa prova il piano ha indicato:

```json
{
  "action": "pick_and_place",
  "target_instrument": "surgical scissors",
  "target_keyword": "scissors",
  "source_tray": "tray_2",
  "destination_tray": "tray_1",
  "planner_backend": "ollama"
}
```

L'LLM **non sceglie da solo** quale oggetto prendere: interpreta il bersaglio richiesto dall'utente. Il nome nel piano diventa il prompt testuale per SAM 3; origine e destinazione sono usate dopo per la scena simulata.

### Controllo del piano e limite osservato

Il planner generale dispone anche di un fallback deterministico. Nel comando diretto della pipeline reale, `--require-llm` richiede il backend `ollama` senza fallback e `--expected-keyword` rifiuta una keyword diversa; la prova precedente usava controlli equivalenti in `scripts/test_clip.py`. Questo controllo non garantisce la correttezza semantica completa.

In una prova precedente Qwen aveva interpretato la frase italiana «Prendi le forbici dal vassoio 2 e mettile nel vassoio 1» come `scalpel`. Per questo il risultato va controllato e il test non va presentato come riconoscimento linguistico infallibile.

## 4. SAM 3: dal testo alla maschera dello strumento

### Ingresso ed elaborazione

`pipeline.real.segment` legge l'immagine RGB acquisita e il nome `target_instrument` dal piano. `Sam3Processor` prepara immagine e testo; `Sam3Model` elabora il prompt e produce istanze candidate, con maschere e punteggi. La pipeline sceglie l'istanza con punteggio massimo per il prompt accettato.

La soglia predefinita per la pipeline reale è **0,30**. Se il nome completo non produce un candidato sufficientemente sicuro, il codice prova la keyword più semplice; se nessun risultato supera la soglia, la segmentazione fallisce. Viene rifiutata anche una maschera che copre oltre il 25% dell'immagine, perché incompatibile con l'ipotesi di un singolo strumento. Il punteggio di SAM è una confidenza del modello, non una garanzia che l'istanza sia quella desiderata.

### Risultato

Sul ritaglio del frame 300, il prompt `surgical scissors` ha dato **0,9429**, con **1.261 pixel attivi**. L'[overlay salvato](../data/outputs/clip_scissors_gpu/frame_000300/real_segmentation_overlay.png) mostra in verde lo strumento ad anelli sulla destra. Il frame 0 ha dato **0,9463** e una nuvola di 1.342 punti.

Sono state eseguite anche prove con il prompt `scalpel`: punteggio massimo 0,033 sul frame intero e 0,021 sul ritaglio. Entrambi sono sotto soglia e la pipeline si è fermata. Questo mostra perché conviene ispezionare l'overlay e non considerare automaticamente valida una maschera debole.

**Output principali:** `real_segmentation_mask.png`, `real_segmentation_overlay.png`, `real_segmentation.json`.

## 5. Dalla maschera alla nuvola 3D

### Deproiezione dei pixel

Per ogni pixel della maschera `(u, v)` con profondità valida `Z`, espressa in metri, la pipeline applica il modello pinhole:

```text
Xcamera = (u - cx) × Z / fx
Ycamera = (v - cy) × Z / fy
Zcamera = Z
```

Il frame ottico usa X verso destra nell'immagine, Y verso il basso e Z lungo la direzione di vista. Il codice esclude profondità non finite o fuori da **0,1–3,0 m**. Un filtro robusto scarta inoltre profondità anomale rispetto alla mediana della maschera, usando una banda basata sulla deviazione mediana assoluta (MAD), con banda minima di 2 cm.

I punti sono poi trasformati nel riferimento base configurato:

```text
p_base = T_base_da_world × T_world_da_camera × p_camera
```

Le `T` sono matrici omogenee 4×4; la posa camera usata nella prima deproiezione è quella **nominale della configurazione**, non una misura mano-occhio. La nuvola risultante è utile per la prova, ma la sua posizione nella base non va interpretata come posizione fisica misurata del KUKA.

### Risultato del frame 300

La maschera ha prodotto **1.261 punti**, salvati in [real_deprojection_cloud.ply](../data/outputs/clip_scissors_gpu/frame_000300/real_deprojection_cloud.ply), con unità dichiarata in metri. L'estensione massima sugli assi della nuvola nel riferimento nominale è circa **0,1224 m**. Il report registra `motion_safe=false` e geometria non verificata per il moto.

## 6. GraspGen: dalla nuvola alle pose candidate

### Preparazione e inferenza

`pipeline.graspgen.backend` verifica il numero di punti, i valori finiti, l'unità dichiarata e l'estensione della nuvola. Se la nuvola supera 4.096 punti, ne seleziona un sottoinsieme deterministico. Sottrae quindi il centroide e invia **solo le coordinate XYZ** al client ufficiale ZeroMQ; non invia né l'immagine né la frase dell'utente. Il server fornisce prima i metadati della rete e dei checkpoint per il controllo di provenienza.

Il server utilizza il modello ufficiale fissato nella repo: un **generatore diffusion**, condizionato dalla nuvola dell'oggetto e dalla pinza del modello, e un **discriminatore PointNet** che assegna un punteggio alle pose proposte. I checkpoint caricati sono quelli **Robotiq 2F-140**. La configurazione di questa prova conserva 200 proposte (`num_grasps: 200`, `topk_num_grasps: 200`); non impone una soglia positiva di accettazione (`grasp_threshold: -1`).

Ogni proposta descrive una posa rigida della base pinza: posizione e orientamento. Nella convenzione del modello, **+Z** è l'asse di avvicinamento e **+X** l'asse di chiusura. Il client ripristina il centroide, ordina globalmente i candidati per punteggio e calcola anche la posa del TCP tramite una trasformazione nominale. Il modello, in questo percorso, **non predice l'apertura necessaria della pinza**.

### Che cosa significa il punteggio

Il punteggio è la valutazione del discriminatore per la proposta originale: non è una probabilità certificata di successo. GraspGen non verifica qui cinematica del KUKA, collisioni con la cella, contatti fisici o zone funzionali chirurgiche. La scelta del candidato effettivamente eseguibile in simulazione avviene nello stadio successivo.

La prima prova sul frame 300 ha salvato 100 candidati. Per la simulazione è stato generato **un nuovo insieme di 200 candidati dalla stessa nuvola**; si trova in `data/outputs/clip_graspgen_200/`. L'inferenza è stocastica: una nuova generazione può produrre pose e risultati diversi.

## 7. Collegamento tra la clip e la cella simulata

Il simulatore ordinario richiede dati coerenti con una propria cattura simulata; non era sufficiente passargli direttamente il report reale. È stato quindi aggiunto l'adattatore separato `scripts/test_clip_sim.py`. Esso legge i risultati salvati, controlla l'hash della nuvola rispetto al report GraspGen e verifica la provenienza del server. Il report originale non viene modificato e l'autorizzazione al moto fisico rimane falsa.

### Trasformazione coerente di punti e prese

La nuvola e le prese sono inizialmente nel frame base nominale usato dalla deproiezione. L'adattatore:

1. applica l'inversa della trasformazione camera→base usata in cattura per tornare al frame ottico;
2. applica `T_world_from_color_nominal` del frame della clip, che orienta la scena secondo il piano nominale della registrazione;
3. applica una traslazione aggiuntiva per posizionare l'oggetto sul vassoio sorgente **ipotetico** della simulazione.

In forma compatta:

```text
T_sim_da_base_originale = T_traslazione_sim × T_clip_nominale_da_camera × inverse(T_base_originale_da_camera)
p_sim = T_sim_da_base_originale × p_base_originale
T_sim_da_pinza = T_sim_da_base_originale × T_base_originale_da_pinza
```

La **stessa trasformazione rigida** è applicata a punti, base pinza e TCP di ogni candidato. La matrice numerica e le ipotesi sono registrate nel [manifest della prova](../data/outputs/clip_sim_300_video/clip_simulation_manifest.json).

### Oggetto e cella usati da PyBullet

L'adattatore costruisce un **involucro convesso** dei punti visibili. Nel riferimento della scena il proxy misura circa **68,3 × 127,6 × 6,6 mm**. Questo involucro chiude i buchi degli anelli, riempie le concavità e non ricostruisce le superfici nascoste: per questo nel video l'oggetto somiglia a un cuneo e non a forbici fedeli.

La cella contiene un modello **KUKA iiwa 7 R800**, la geometria nominale articolata della **Robotiq 2F-140**, tavolo e due vassoi configurati. Altri strumenti, persone e ostacoli osservabili nella clip non sono ricostruiti nella scena. La posizione dei vassoi, il montaggio della pinza e il TCP sono valori di simulazione, non misure della cella fisica.

## 8. Come viene scelta la presa da eseguire in simulazione

I candidati sono esaminati in ordine decrescente di punteggio GraspGen. Per ogni proposta il validatore cerca una geometria di chiusura della pinza rispetto al proxy, poi tenta di pianificare l'intero percorso. I controlli comprendono:

- geometria delle due dita e distanze dai polpastrelli durante la chiusura; ricerca di una correzione geometrica limitata;
- cinematica inversa, limiti dei giunti e continuità del ramo cinematico;
- collisioni e margini campionati rispetto agli oggetti modellati e alle parti del robot;
- traiettorie temporizzate di avvicinamento, presa, sollevamento, trasferimento, deposito e allontanamento.

La validazione seleziona **il primo candidato nell'ordine dei punteggi che supera i controlli**, non necessariamente quello con il punteggio massimo. I controlli sono campionati: non costituiscono una certificazione di assenza continua di collisioni.

### Risultato dei 200 candidati nella prova precedente

Il punteggio più alto dell'insieme era circa **0,9574**, ma la proposta corrispondente non ha superato i controlli del simulatore. Prima della proposta ammessa sono state scartate **194** pose:

| Fase di scarto | Numero |
|---|---:|
| Geometria e chiusura bilaterale della pinza | 189 |
| Cinematica inversa durante il trasferimento | 3 |
| Cinematica inversa alla presa | 1 |
| Corridoio cartesiano presa–prepresa | 1 |

È stato selezionato `server_index=194`, **195º nell'ordine dei punteggi**, con punteggio originale circa **0,0867**. La geometria è stata corretta traslando la posa di **30 mm verso l'alto**; l'apertura stimata della pinza è circa **77,1 mm**. Il punteggio GraspGen appartiene alla posa originale: **la posa corretta non è stata rivalutata dal modello**.

Nella prova che includeva anche il ritorno finale alla posizione di parcheggio, nessuno dei 200 candidati ha superato tutte le fasi: questo candidato è stato rifiutato nella fase `Return` per mancata soluzione IK. Nella prova riuscita è stato omesso **soltanto il ritorno opzionale al parcheggio**, mantenendo pianificazione e verifica dell'allontanamento dopo il deposito. Il percorso e le ragioni degli scarti sono nel [report di validazione](../data/outputs/clip_sim_300_video/graspgen_sim_motion_validation.json).

## 9. Movimento simulato fino all'appoggio nella prova precedente

Dopo la pianificazione, il braccio simulato ha eseguito la traiettoria con controllo di posizione dei giunti. Il modello della pinza segue cinematicamente le articolazioni; all'evento di chiusura, il trasporto dell'oggetto avviene tramite un **vincolo fisso ideale**, non tramite una presa con attrito e forza verificati.

| Fase | Cosa avviene nel modello | Durata simulata |
|---|---|---:|
| Approach | Il braccio raggiunge una posizione di pre-presa | 4,26 s |
| Descend | La pinza segue il corridoio fino alla posa di presa | 5,03 s |
| Close gripper | Le dita si chiudono e viene applicato il vincolo ideale | 0,60 s |
| Lift | Il proxy viene sollevato dal vassoio sorgente | 10,82 s |
| Transfer | Il braccio trasporta il proxy sopra il vassoio destinazione | 6,93 s |
| Place | Il braccio scende alla posizione di deposito | 11,18 s |
| Open gripper | La pinza si apre e il vincolo viene rimosso | 0,60 s |
| Retreat | La pinza si allontana dal punto di rilascio | 5,18 s |

La durata totale del moto è **44,6 s simulati**, con 10.705 passi. Dopo l'esecuzione il proxy è risultato all'interno del vassoio di destinazione, fermo e con tre contatti rilevati nel modello PyBullet. Questo verifica il **deposito del proxy nella scena simulata**, non la stabilità di una presa reale. Il ritorno alla posa di parcheggio non è incluso.

**Output principali:** [video](../data/outputs/clip_sim_300_video/clip_simulation.mp4), [scena iniziale](../data/outputs/clip_sim_300_video/scene_initial.png), [scena finale](../data/outputs/clip_sim_300_video/scene_final.png), [risultato JSON](../data/outputs/clip_sim_300_video/clip_simulation_result.json), [traiettoria articolare NPZ](../data/outputs/clip_sim_300_video/graspgen_sim_joint_path.npz).

## 10. Nuova prova unificata: un comando fino al deposito simulato

**Integrazione del modello nella pipeline reale.** Il KUKA iiwa 7 R800 e la pinza Robotiq 2F-140 sono i modelli confermati per il banco. Prima dell'inferenza GraspGen, la pipeline reale carica e controlla il modello iiwa 7 preparato dalla revisione fissata di LBR Stack, il frame base e il checkpoint Robotiq. Scrive `real_robot_profile.json` e riporta modello, hash, frame, giunti, limiti URDF e convenzione TCP anche in `grasp.json`. Il riferimento GraspGen base-pinza → TCP non è una misura del montaggio flangia → pinza. I limiti URDF non sostituiscono quelli configurati sul controller del banco.

La nuova prova `data/outputs/real_iiwa_robotiq_01/` ha eseguito dall'LLM a GraspGen lo stesso frame 300: 1.261 punti, 100 candidati, checkpoint server `robotiq_2f_140`, profilo iiwa 7 registrato e movimento bloccato. Il numero è 100 perché questa esecuzione usa `config/graspgen.yaml`, che conserva i primi 100 candidati; la prova con deposito simulato sotto usa la configurazione da 200. Non è stata eseguita una presa reale né una nuova prova di deposito con questa run. Montaggio, TCP, calibrazione camera–robot e parametri della cella devono ancora essere misurati o validati; non esiste ancora un esecutore fisico KUKA/pinza validato nella pipeline reale.

Il 26 novembre è stato eseguito `run_pipeline_real.py` con `--source clip --simulate-deposit`: un solo avvio ha prodotto in successione `plan`, `real_capture`, `real_segmentation`, `real_deprojection`, `grasp` e `real_simulation_preview`. Ogni stadio ha scritto il proprio report nella stessa directory di run; la simulazione è uno stadio opzionale della pipeline reale e non è un comando al KUKA fisico.

Sul frame 300 l’LLM ha restituito `scissors`; SAM 3 ha ottenuto **0,9429**, la nuvola contiene **1.261 punti** e GraspGen ha prodotto **200 pose**. Nel primo gruppo della run riuscita sono state rifiutate **188** proposte: 185 al controllo geometrico della chiusura e 3 per IK alla presa. È stato ammesso il candidato con `server_index=188`, punteggio originale circa **0,1289**, dopo una correzione simulata di **35 mm verso l’alto**; la posa corretta non è stata rivalutata da GraspGen. La pinza stimata chiude a circa 68,1 mm.

PyBullet ha completato avvicinamento, discesa, chiusura ideale, sollevamento, trasferimento, deposito, apertura e allontanamento in **47,43 s simulati**. Il proxy è risultato fermo nel vassoio destinazione con tre contatti nel modello. È stato usato `--sim-omit-return`: il ritorno opzionale al parcheggio non è incluso. L’output visivo è il [video della prova unificata](../data/outputs/real_unified_final/simulation/clip_simulation.mp4); i dettagli sono nel [report dello stadio](../data/outputs/real_unified_final/real_simulation_preview.json) e nel [risultato del movimento](../data/outputs/real_unified_final/simulation/clip_simulation_result.json).

Due precedenti tentativi unificati hanno prodotto i candidati ma si sono fermati prima del moto perché nessuna posa aveva superato tutti i controlli: sono conservati in `real_unified_300/` e `real_unified_300_b/`. È stato aggiunto un numero limitato di gruppi indipendenti GraspGen (`--sim-grasp-attempts`, predefinito 3, massimo 10), usati solo se un gruppo viene interamente rifiutato prima del moto. Gli scarti e il report del gruppo selezionato restano tracciati. Anche con più gruppi il successo non è garantito.

La sorgente live ROS 2 usa gli stessi report di cattura, segmentazione, nuvola e prese. È stato predisposto un adattatore simulato che richiede posa camera da TF, estrinseche verificate, dati rettificati senza distorsione residua, base coincidente con il world e nuvola coerente con il vassoio configurato; mantiene le coordinate osservate e non sposta automaticamente la scena live. Con una fixture sintetica in formato ROS, costruita a partire da risultati salvati, è stato eseguito anche lo stadio di simulazione attraverso `run_pipeline_real.py`: il deposito è terminato. Questa prova precede i nuovi controlli sul timestamp descritti sotto e la vecchia fixture non li documentava; non dimostra quindi il deposito live con i controlli aggiornati. La fixture non è un'acquisizione e non verifica LLM, SAM 3 e GraspGen dal vivo: **la D456 live e il percorso completo live non sono stati provati con hardware**. Nell'immagine Docker ROS verificata non era disponibile `ffmpeg`: per registrare il video dalla pipeline eseguita in quell'ambiente occorre installarlo; senza video si può omettere `--sim-video`.

### Verifica dei riferimenti e sincronizzazione live

La trasformazione applicata è `p_base = inverse(T_world_base) · T_world_camera · p_camera`, con profondità in metri e intrinseche della camera. Il centroide sottratto prima di GraspGen viene riaggiunto alle pose restituite; nuvola e candidati sono così espressi nel riferimento della base. Le slide e il report descrivono questa catena nel contesto della loro scena simulata; per una scena reale serve una trasformazione camera–robot misurata e validata.

Il controllo del percorso ROS ha evidenziato che la TF veniva letta all'ultimo istante disponibile: ora viene richiesta **all'istante dell'immagine colore**. Sono inoltre controllati gli header di RGB, profondità e CameraInfo: frame ottico coerente, timestamp validi e scarto entro la tolleranza configurata. Il default è un solo frame di profondità; più frame possono essere usati per diagnostica, ma senza compensazione del moto non sono dichiarati geometricamente verificati. Anche distorsione residua, meno del 50% di profondità valida o TF non disponibile all'istante della cattura impediscono di marcare la geometria live come verificata. Questi controlli non sostituiscono la calibrazione.

È stata eseguita una prova ROS 2 con camera e TF **sintetiche**, senza dispositivi fisici: RGB e profondità hanno avuto lo stesso timestamp, la TF è stata trovata all'istante della cattura e una maschera rettangolare di test ha prodotto 8.800 punti con centroide `[0.5991, -0.4209, 0.102]` m in `lbr_link_0`, coerente con la scena sintetica. Le estrinseche nel file standard rimangono non verificate e `motion_safe=false`. Il controllo sul PC ha rilevato zero RealSense: non si tratta di una validazione della D456.

Separatamente è stato rieseguito l'adattatore live con i controlli aggiornati su una fixture costruita da risultati salvati; la fixture dichiara artificialmente timestamp e calibrazione validi per esercitare il codice. La pianificazione e il deposito simulato del proxy sono terminati. Questa seconda prova non usa il messaggio ROS sintetico precedente e non costituisce una cattura o una misura fisica.

Dopo queste modifiche sono passati **97 test** che coprono i profili robot, gli input reali e sintetici, le trasformazioni, GraspGen e gli adattatori di simulazione. Le prove automatiche verificano il software nei casi esercitati e non sostituiscono la verifica sul banco.

Nella prova della clip rimane una traslazione artificiale verso il vassoio simulato, registrata nel manifest: non equivale a una calibrazione fisica. L'anteprima live conserva le coordinate osservate e controlla i vassoi configurati, ma rappresenta solo tavolo e vassoi, non gli altri strumenti o ostacoli percepiti. Il punto 3 è pertanto verificato per la matematica e il test sintetico; resta aperto per calibrazione fisica e rappresentazione degli ostacoli reali.

## 11. Esiti e limiti da comunicare in presentazione

| Verifica | Esito | Interpretazione corretta |
|---|---|---|
| LLM su istruzione inglese scelta | `surgical scissors`, keyword `scissors` | Piano coerente con questa istruzione; una frase italiana precedente era stata interpretata erroneamente |
| SAM 3 sul frame 300 ritagliato | 0,9429; overlay compatibile con lo strumento a destra | Una singola istanza selezionata; confidenza non equivale a certezza |
| Deproiezione frame 300 | 1.261 punti | Nuvola della superficie visibile, con scala ipotizzata e posa nominale |
| GraspGen | 200 proposte nella run usata per il movimento | Sono ipotesi della pinza Robotiq; punteggi non certificano successo |
| Pianificazione e deposito nella run unificata | 1 proposta ammessa dopo 188 scarti; deposito completato | Valido per proxy e cella simulati, senza ritorno opzionale al parcheggio |
| Camera e robot fisici | Non collegati e non mossi | Nessuna prova di acquisizione live o presa fisica |

La pipeline reale accetta la clip direttamente con `--source clip`: senza `--simulate-deposit` termina ai candidati; con tale opzione richiama l’adattatore di anteprima PyBullet come stadio della stessa esecuzione. Non invia comandi al robot fisico. Gli output locali e i pesi sono esclusi da Git: altri PC o un nuovo clone richiedono dati, asset e ambienti separati.

## 12. Come ripetere la sequenza

Eseguire i comandi dalla radice della repo. Ogni cartella di output indicata deve essere **nuova**. Per il modello Qwen e per GraspGen avviare due servizi separati, solo se non sono già attivi:

```bash
cd /home/alice/Desktop/ROS2Project/ROS2Project
bash scripts/run_ollama.sh
```

```bash
cd /home/alice/Desktop/ROS2Project/ROS2Project
bash scripts/graspgen_demos.sh server
```

### A. Un comando: pipeline reale e deposito simulato

In un terzo terminale, dalla stessa root:

```bash
# La scala 0.001 m/unità è un'ipotesi da verificare alla sorgente.
OLLAMA_HOST=127.0.0.1 .venv/bin/python run_pipeline_real.py \
  --source clip --clip-dir execution/clip --frame-index 300 \
  --crop 550 195 400 300 --depth-scale 0.001 \
  --instruction 'Pick the surgical scissors from tray 2 and place them in tray 1.' \
  --require-llm --expected-keyword scissors \
  --sam3-model data/external/sam3-prepared --sam3-device cuda \
  --grasp-backend graspgen --simulate-deposit --sim-omit-return \
  --sim-grasp-attempts 6 --sim-video --out-dir data/outputs/clip_presentazione
```

Un singolo avvio esegue `plan → real_capture → real_segmentation → real_deprojection → grasp → real_simulation_preview`: produce piano, cattura, overlay, maschera, nuvola, prese, report della simulazione e, se completata, video nella directory indicata. Nella pipeline reale GraspGen è ora il backend predefinito; nel comando rimane esplicito per chiarezza. Se manca Ollama o il bersaglio non corrisponde, la prova si ferma. Anche nel caso in cui la configurazione dichiari estrinseche verificate, per la sorgente clip `motion_safe` resta falso: scala, allineamento, distorsione e posa non sono certificati.

Per il frame 0 cambiare `--frame-index 0` e usare un'altra cartella. La sorgente diretta elabora un indice per comando; `scripts/test_clip.py` rimane disponibile per una lista di indici. Una directory clip non vuota viene rifiutata per evitare di confondere prove vecchie e nuove. Il ritaglio indicato è adatto a questa clip, non a una scena generica. La camera dal vivo resta una sorgente separata, `--source ros2`, non verificata su hardware.

### B. Alternativa manuale: generare 200 nuovi candidati dalla stessa nuvola

```bash
mkdir data/outputs/clip_presentazione_200
cp data/outputs/clip_presentazione/{clip_input.json,plan.json,real_capture.json,real_deprojection.json,real_deprojection_cloud.ply} data/outputs/clip_presentazione_200/
.venv/bin/python -m pipeline.grasp \
  --deprojection data/outputs/clip_presentazione_200/real_deprojection.json \
  --plan data/outputs/clip_presentazione_200/plan.json \
  --out-dir data/outputs/clip_presentazione_200 --grasp-backend graspgen \
  --graspgen-config config/graspgen_simulation.yaml
```

### C. Pianificare ed eseguire soltanto in PyBullet

```bash
MPLCONFIGDIR=/tmp/ros2project-mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/test_clip_sim.py \
  --run-dir data/outputs/clip_presentazione_200 \
  --out-dir data/outputs/clip_presentazione_sim \
  --omit-return --video
```

`--omit-return` termina dopo l'allontanamento controllato e non richiede il rientro al parcheggio; `--video` richiede `ffmpeg`. Il nuovo insieme di prese può non produrre una traiettoria accettabile: in tal caso la simulazione si ferma e conserva le ragioni degli scarti. Per riprodurre le **stesse pose salvate nella prova descritta**, sostituire il `--run-dir` con `data/outputs/clip_graspgen_200` e usare comunque un output nuovo.

## 13. Verifiche e lavoro ancora necessario

- Driver NVIDIA, calcolo PyTorch su GPU, `pip check` e import di pyrealsense2 sono stati verificati. Il server GraspGen ha usato CUDA 12.1 nel proprio container; SAM 3 ha usato CUDA 12.8 nella `.venv`.
- La prova del frame 300 su GPU ha completato gli stadi da cattura a GraspGen in circa 10,8 s, escluso il planner; il tempo dipende dal carico del PC. Le prove complete sui frame 0 e 300 sono state eseguite anche su CPU.
- I test del planner/client GraspGen hanno superato 31 controlli; i test dell'adattatore clip e della simulazione ne hanno superati 12. I server usati nei test automatici del client sono simulati; le prove sui frame hanno usato il server GraspGen effettivo.
- Prima di usare la camera reale: verificare collegamento USB 3, scala, sincronizzazione, allineamento, rettifica e `CameraInfo`; poi validare su punti indipendenti la calibrazione intrinseca e la trasformazione camera–robot.
- Prima di qualsiasi prova fisica: misurare montaggio e TCP della pinza, usare un modello coerente con l'hardware e validare cinematica, collisioni, traiettorie e contatti. Il backend ROS di movimento documentato nella repo è simulato: serve un'integrazione dedicata e collaudata con KUKA e pinza fisici. `motion.allowed=false` e `physical_grasp_validated=false` restano invariati.

### Aggiornamento del 26 novembre: pipeline reale diretta

- `run_pipeline_real.py --source clip` è stato eseguito dall’LLM fino a GraspGen direttamente, senza script di simulazione, sui frame 0 e 300. Il frame 0 ha prodotto 1.342 punti e il frame 300 1.261 punti; entrambi hanno restituito 100 candidati GraspGen con movimento bloccato. Output: `data/outputs/real_direct_clip_000/` e `data/outputs/real_direct_clip_300/`.
- La camera live e la calibrazione camera–robot non sono state provate; le conclusioni sulla simulazione si riferiscono alla prova distinta descritta sopra. I test della sorgente clip, della camera live simulata, delle protezioni e del client GraspGen sono stati eseguiti con 53 test superati.

- Dopo l’unificazione sono stati eseguiti 72 test delle sorgenti reali, validazioni, client GraspGen e adattatori di simulazione. Il percorso live è stato verificato con fixture sintetiche, non con la camera fisica.
