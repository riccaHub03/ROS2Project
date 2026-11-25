# Camera RealSense (D456) su un PC Windows con WSL

Come acquisire con la RealSense quando il progetto gira in WSL/Docker su un PC Windows: diagnosi, registrazione,
calibrazione e riproduzione delle registrazioni come se fossero la camera dal vivo.

## Architettura

```
terminale WSL, nella radice del progetto
 ├─ acquisisce:  Python di Windows (conda `realsense`) ─► camera USB ─► data/recordings/*.db3
 └─ elabora:     container ─► realsense2_camera riproduce il .db3 ─► topic ROS 2 ─► pipeline.real / GraspGen
```

- **Windows possiede la camera.** È l'unica strada supportata da RealSense su questo tipo di PC: driver nativi,
  nessun passaggio intermedio. Gli script si lanciano da WSL col Python di Windows (`scripts/rs_windows.sh`), che
  lavora nella cartella del progetto e ci salva i file.
- **WSL/Docker elabora.** Il driver ROS 2 `realsense2_camera` riproduce una registrazione pubblicando gli stessi
  topic della camera dal vivo, quindi la pipeline non cambia.
- **Perché la camera non va direttamente nel container.** La USB arriverebbe a WSL solo con `usbipd` (protocollo
  USB/IP su una rete virtuale). RealSense non supporta WSL2; nelle prove la camera veniva vista ma non arrivavano
  frame. Mentre è passata a WSL, inoltre, Windows non può usarla. Su un PC con Ubuntu nativo, invece, la camera nel
  container è la soluzione normale.

Le versioni sono allineate a **2.59.2** ovunque (SDK Windows, `pyrealsense2`, `ros-humble-librealsense2`): le
registrazioni si leggono uguali su Windows e nel container.

## Installazione (una volta)

**Windows**

1. Ambiente conda (serve agli script):
   ```bash
   conda create -n realsense --override-channels -c conda-forge python=3.12 pip
   conda activate realsense
   pip install -r requirements-calibration.txt
   ```
   Il canale `conda-forge` evita di dover accettare i termini dei canali predefiniti di Anaconda.
2. SDK RealSense (facoltativo: RealSense Viewer, Depth Quality Tool, aggiornamento firmware). Installer
   `RealSense.SDK-WIN10-2.59.2.*.exe` dalle release di GitHub `realsenseai/librealsense`. Installarlo in
   `%LOCALAPPDATA%\Programs\RealSense SDK 2.0`, **non** nei Documenti se sono sincronizzati con OneDrive.

**Container** — `docker/Dockerfile` installa `ros-humble-realsense2-camera` (porta con sé `ros-humble-librealsense2`)
e `pyrealsense2`. In un container già creato:
```bash
apt-get update && apt-get install -y ros-humble-realsense2-camera
pip3 install pyrealsense2==2.59.2.11152
```

## Uso

Dal terminale WSL, nella radice del progetto. La camera deve essere collegata a Windows e **nessun altro
programma deve usarla** (RealSense Viewer chiuso, nessun `usbipd attach`).

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
- **Dati non versionati.** `data/recordings/` e `data/calibration/` sono esclusi da git (le registrazioni pesano
  GB): il risultato utile della calibrazione si riporta nel YAML in `config/`.
- **usbipd** non serve in questa architettura. Se la camera è stata passata a WSL, restituirla a Windows con
  `usbipd detach --busid <id>` e `usbipd unbind --busid <id>` (PowerShell da amministratore). `usbipd list` deve
  mostrare `Not shared` e nessun dispositivo in `Persisted`.

## Stato (24 novembre 2026)

| Cosa | Stato |
|---|---|
| Script avviabili da WSL col Python di Windows (`rs_windows.sh`) | provato |
| Calibrazione su un video sintetico con camera nota (fx = fy = 900) | provata: intrinseche ritrovate entro 1 px, RMS 0,19 px, uguale con OpenCV 4.6 e 4.14 |
| Esportazione e riproduzione ROS 2 di un `.db3` sintetico | provate: frame, profondità, intrinseche e topic corretti |
| Trasmissione dei frame dalla D456 su Windows | **da verificare**: nell'ultima prova (USB 2.1) il Viewer non riceveva frame |
| Registrazione e calibrazione con la camera reale | **da fare** |
