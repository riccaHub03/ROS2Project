# Camera RealSense (D456) su un PC Windows con WSL

Come usare la RealSense quando il progetto gira in WSL/Docker su un PC Windows: immagini **dal vivo** per la
pipeline (ponte di rete), diagnosi, registrazione, calibrazione e riproduzione delle registrazioni.

## Architettura

```
Windows (Python venv o conda, lanciato da WSL)               container (ROS 2)
camera USB ─► pyrealsense2 ─┬─ ponte TCP localhost:50551 ──► realsense_bridge.py ros ─► topic /camera/... ─► pipeline.real
                            └─ registrazione .db3 ─────────► realsense2_camera (rosbag_filename) ─► stessi topic
```

- **Windows possiede la camera.** È l'unica strada supportata da RealSense su questo tipo di PC: driver nativi,
  nessun passaggio intermedio. Gli script si lanciano da WSL col Python di Windows (`scripts/rs_windows.sh`), che
  lavora nella cartella del progetto e ci salva i file.
- **Il container riceve i fotogrammi e pubblica i topic.** Dal vivo li porta il ponte
  (`scripts/realsense_bridge.py`); da una registrazione li ripubblica il driver ROS 2 `realsense2_camera`. In
  entrambi i casi i topic sono quelli di un driver camera, quindi la pipeline non cambia.
- **Perché la camera non va direttamente nel container.** La USB arriverebbe a WSL solo con `usbipd` (protocollo
  USB/IP su una rete virtuale). RealSense non supporta WSL2; nelle prove la camera veniva vista ma non arrivavano
  frame. Mentre è passata a WSL, inoltre, Windows non può usarla. Su un PC con Ubuntu nativo, invece, la camera nel
  container è la soluzione normale.

Le versioni sono allineate a **2.59.2** ovunque (SDK Windows, `pyrealsense2`, `ros-humble-librealsense2`): le
registrazioni si leggono uguali su Windows e nel container.

## Installazione (una volta)

**Windows**

1. Un Python per Windows con i pacchetti RealSense (serve agli script). Va bene **uno** dei due:
   - **venv**: più leggero (circa 200 MB), consigliato su un PC nuovo. Serve un Python installato su Windows
     (3.10–3.14, da python.org oppure `winget install Python.Python.3.12`; il comando `python` del Microsoft Store
     non è un Python installato). In PowerShell:
     ```powershell
     py -3.12 -m venv $env:USERPROFILE\.venvs\realsense
     ```
   - **conda**: se è già installato (ambiente di circa 350 MB). Il canale `conda-forge` evita di dover accettare i
     termini dei canali predefiniti di Anaconda:
     ```bash
     conda create -n realsense --override-channels -c conda-forge python=3.12 pip
     ```

   Poi, in WSL nella radice del progetto, si installano i pacchetti con il Python appena creato:
   ```bash
   bash scripts/rs_windows.sh -m pip install -r requirements-calibration.txt
   ```
   `scripts/rs_windows.sh` cerca prima il venv `%USERPROFILE%\.venvs\realsense`, poi l'ambiente conda `realsense`;
   per un'altra posizione: `RS_WIN_PYTHON=/mnt/c/.../python.exe`. I pacchetti sono gli stessi (wheel di pip): il
   comportamento non cambia.
2. SDK RealSense (facoltativo: RealSense Viewer, Depth Quality Tool, aggiornamento firmware). Installer
   `RealSense.SDK-WIN10-2.59.2.*.exe` dalle release di GitHub `realsenseai/librealsense`. Installarlo in
   `%LOCALAPPDATA%\Programs\RealSense SDK 2.0`, **non** nei Documenti se sono sincronizzati con OneDrive.

**Container** — `docker/Dockerfile` installa `ros-humble-realsense2-camera` (porta con sé `ros-humble-librealsense2`)
e `pyrealsense2`. In un container già creato:
```bash
apt-get update && apt-get install -y ros-humble-realsense2-camera
pip3 install pyrealsense2==2.59.2.11152
```

## Dal vivo: il ponte verso il container

La camera resta aperta su Windows e manda di continuo colore e profondità **allineata al colore** al container, che
li pubblica come un driver camera. Quando la pipeline viene chiamata prende i fotogrammi più recenti.

1. Nel container (ROS 2 caricato), il ricevitore:
   ```bash
   python3 scripts/realsense_bridge.py ros
   ```
2. In WSL, nella radice del progetto, il mittente (Python di Windows, vede la camera):
   ```bash
   bash scripts/rs_windows.sh scripts/realsense_bridge.py send                       # 640x480 a 15 fps
   bash scripts/rs_windows.sh scripts/realsense_bridge.py send --color 1280x720@15   # più risoluzione (USB 3)
   ```
3. Nel container, la pipeline come con qualsiasi camera ROS 2 (serve anche la TF `world → camera_color_optical_frame`
   della calibrazione mano-occhio, vedi il README):
   ```bash
   python3 -m pipeline.real.preflight --source ros2 --ee-frame <frame_end_effector>
   python3 -m pipeline.real.capture --source ros2 --ee-frame <frame_end_effector>
   ```

Pubblica `/camera/color/image_raw` (`bgr8`), `/camera/aligned_depth_to_color/image_raw` (`16UC1`, mm), i rispettivi
`camera_info` e usa il frame `camera_color_optical_frame`: sono i valori predefiniti di `camera_overhead.ros2`.
L'ordine di avvio non conta: il mittente aspetta il ricevitore e si ricollega da solo se cade (e viceversa). Ogni
10 s entrambi stampano i fotogrammi al secondo. Senza camera, `send --fake` manda una scena sintetica per provare la
catena fino alla pipeline. Il ponte non pubblica la TF della camera: viene dalla calibrazione.

## Registrazioni, calibrazione ed esportazione

Dal terminale WSL, nella radice del progetto. La camera deve essere collegata a Windows e **nessun altro
programma deve usarla** (ponte, RealSense Viewer, `usbipd attach`).

1. **Diagnosi**: modello, firmware, tipo di USB e frame al secondo per colore, profondità e i due insieme.
   ```bash
   bash scripts/rs_windows.sh scripts/realsense_probe.py            # --reset: prima riavvia la camera
   ```
   Il collegamento deve essere **USB 3.2**. Con USB 2.1 la profondità non parte o è molto limitata: cambiare porta
   (USB 3, blu o "SS", oppure USB-C) e usare il cavo originale, senza hub.
2. **Registrazione** con anteprima (colore e profondità affiancati):
   ```bash
   bash scripts/rs_windows.sh scripts/realsense_record.py --seconds 60      # conto alla rovescia, poi 60 s
   bash scripts/rs_windows.sh scripts/realsense_record.py                   # manuale: R registra/pausa, Q chiude
   bash scripts/rs_windows.sh scripts/realsense_record.py --no-depth        # solo colore
   ```
   Produce `data/recordings/rs_<data>_<ora>.db3` e un `.json` con camera, USB e profili usati.
3. **Calibrazione intrinseca** con una scacchiera stampata al 100% e incollata su un piano rigido (10×7 quadrati =
   9×6 angoli interni; misurare il lato del quadrato):
   ```bash
   bash scripts/rs_windows.sh scripts/calibrate_camera.py --realsense --size 640x480 --fps 15 --pattern 9x6 --square-mm 25
   ```
   SPAZIO cattura una posa, C calcola, Q esce. Il risultato va in `data/calibration/camera_intrinsics.json` con il
   confronto con le intrinseche di fabbrica. Si può calibrare anche da una registrazione (passo 4, poi
   `--video data/calibration/<nome>/color.mp4`).
4. **Esportazione** di una registrazione (nel container, oppure con `rs_windows.sh`):
   ```bash
   python3 scripts/bag_export.py data/recordings/rs_<data>_<ora>.db3
   ```
   Crea `data/calibration/<nome>/` con `color.mp4`, coppie `frames/color_*.png` + `frames/depth_*.png` (16 bit, mm,
   allineata al colore), `intrinsics_factory.json` e `info.json`.
5. **Riproduzione come camera dal vivo** (nel container):
   ```bash
   ros2 run realsense2_camera realsense2_camera_node --ros-args -r __ns:=/ -r __node:=camera \
       -p rosbag_filename:=/workspace/data/recordings/rs_<data>_<ora>.db3 -p align_depth.enable:=true
   ```
   Pubblica `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw` e `/camera/color/camera_info`
   (frame `camera_color_optical_frame`): sono i nomi predefiniti di `camera_overhead.ros2` nel YAML. Come dal vivo,
   `pipeline.real` vuole anche la TF `world → camera_color_optical_frame` (calibrazione mano-occhio, vedi il README).

## Note

- **Intrinseche.** Con `--source ros2` la pipeline le legge da `CameraInfo`, che la RealSense compila con la
  calibrazione di fabbrica. La calibrazione con la scacchiera serve a **verificarle** (lo script le confronta) e per
  i comandi che leggono le intrinseche dal YAML o da `--intrinsics`. La calibrazione che manca davvero per le prese
  reali è quella **mano-occhio** (posa della camera rispetto al robot).
- **Formato.** Da librealsense 2.59 le registrazioni sono `.db3` (rosbag2), non più `.bag`. Anche il RealSense
  Viewer salva in `.db3`, di default nei Documenti: cambiare la cartella se sono sincronizzati con OneDrive (i file
  sono di alcuni GB).
- **Velocità di scrittura.** Da Windows, la cartella del progetto in WSL si scrive a circa 64 MB/s (il disco `C:` a
  circa 590 MB/s) e SQLite non va d'accordo con le cartelle di rete: `realsense_record.py` registra sul disco locale
  di Windows e sposta il file nel progetto alla fine.
- **Ponte: istanti e rete.** I messaggi portano l'istante di arrivo nel container (colore e profondità arrivano
  insieme, quindi lo scarto è zero e la TF si legge con l'orologio di ROS). Il ricevitore ascolta sulla porta 50551
  (`--port` per cambiarla); con WSL in modalità NAT Windows lo raggiunge come `localhost`. Il container deve usare la
  rete dell'host (`--network host`, come in `scripts/docker_shell.sh`).
- **Dati non versionati.** `data/recordings/` e `data/calibration/` sono esclusi da git (le registrazioni pesano
  GB): il risultato utile della calibrazione si riporta nel YAML in `config/`.
- **usbipd** non serve in questa architettura. Se la camera è stata passata a WSL, restituirla a Windows con
  `usbipd detach --busid <id>` e `usbipd unbind --busid <id>` (PowerShell da amministratore). `usbipd list` deve
  mostrare `Not shared` e nessun dispositivo in `Persisted`.

## Stato (25 novembre 2026)

| Cosa | Stato |
|---|---|
| Ponte dal vivo Windows → container con scena sintetica (`send --fake`) | provato: 15 fotogrammi/s, ritardo ~15–18 ms, dati integri (profondità e colori esatti), riconnessione automatica; `pipeline.real.preflight` tutto OK sulla camera, `pipeline.real.capture` dal vivo in 0,5 s |
| Velocità della rete interna Windows → WSL | misurata: ~620 MB/s (circa 400 coppie colore + profondità 640x480 al secondo) |
| Script avviabili da WSL col Python di Windows (`rs_windows.sh`) | provato, con conda e con un venv (trovati entrambi da soli, pacchetti installati con `rs_windows.sh -m pip`) |
| Calibrazione su un video sintetico con camera nota (fx = fy = 900) | provata: intrinseche ritrovate entro 1 px, RMS 0,19 px, uguale con OpenCV 4.6 e 4.14 |
| Esportazione e riproduzione ROS 2 di un `.db3` sintetico | provate: frame, profondità, intrinseche e topic corretti |
| Trasmissione dei frame dalla D456 su Windows | **da verificare**: nell'ultima prova (USB 2.1) il Viewer non riceveva frame |
| Ponte, registrazione e calibrazione con la camera reale | **da fare** |
