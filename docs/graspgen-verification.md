# Verifica locale GraspGen — 8 novembre 2026

Ambiente: Mac arm64, Python 3.11.18 nella `.venv` esistente. Installate solo
le nuove dipendenze leggere `pyzmq==27.1.0`, `msgpack==1.1.2` e
`msgpack-numpy==0.4.8`. `pip check`: nessun requisito incompatibile.

## Esito

**65 test passati: 46 preesistenti e 19 nuovi**, con:

```bash
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python -m unittest discover -s tests -v
```

Il log dell'esecuzione è in
`data/outputs/graspgen_mac_verification/unittest.log` (ignorato da Git).
I server dei test ascoltano solo su loopback; nel sandbox Codex è stata
necessaria l'esecuzione con permesso di aprire socket locali. Non vi sono
richieste a un server GPU esterno durante i test.

## Copertura verificata

- Import del client ufficiale senza importare torch o installare GraspGen.
- Protocollo ZMQ/msgpack-numpy reale con un server REP simulato.
- XYZ float32 centrati, ripristino del centro una volta sola e cap deterministico.
- Metri, estensione, finitezza, pose rigide e score validi; rifiuto di mm,
  NaN, matrici improprie e risposte di forma errata.
- Frame camera con extrinseci espliciti, base e world, inclusa base traslata
  e ruotata, rotazioni non banali e composizione dell'offset TCP.
- Ordinamento globale degli score e selezione del massimo, indice server,
  persistenza dei candidati e report StageReport v2.
- Mancanza del server, timeout metadata/inferenza, errori GPU simulati,
  risposta vuota e mismatch pinza/modello/configurazione/provenienza.
- Nessun ripiego su PCA; un tentativo GraspGen fallito lascia l'interfaccia
  `grasp.json` in `FAILED` e conserva `grasp_pca.json`.
- Flag effettivamente inoltrati, ripresa dallo stadio grasp nei launcher
  simulato e reale, import dei dati autori e bundle con PLY relativo al report.
- Pipeline completa `--headless --vision ground-truth` con server simulato,
  STL sintetico e base non identità: `GRASP_VISUALIZED` con `--no-motion`,
  `FAILED` per il movimento incompatibile; nessun report di movimento creato.
- Booleani di configurazione insufficienti a superare una deproiezione
  priva di geometria verificata; ulteriore controllo della pinza nell'executor.
- Suite preesistente su PCA, stadi isolati, base robot, preview IK e
  controller interattivo conservata.

Controllati anche `bash -n scripts/graspgen_demos.sh`, compilazione Python,
help delle CLI, `git diff --check`, preflight e generazione della copia
temporanea corretta del Dockerfile ufficiale. La preview nominale sintetica
`data/outputs/graspgen_mac_verification/mock_preview.png` è stata aperta e
ispezionata: è soltanto una verifica grafica con pose inventate, non inferenza.

Il submodule è rimasto pulito e fissato a
`2dd8852e1be60f5f9d277fafcc621835cdf59110`. Nessun commit o push.

## Da eseguire sul PC NVIDIA

Non effettuati sul Mac: download checkpoint/sample_data, download dataset,
build del container CUDA, import delle estensioni CUDA, inferenza reale,
demo ufficiale e verifica fisica. Le correzioni della ricetta Docker e il
server wrapper sono controllati staticamente; il loro funzionamento GPU va
confermato sull'hardware destinatario con `check-gpu`, `build` e `smoke-gpu`.

Seguire l'ordine in [graspgen.md](graspgen.md): demo ufficiale `diffusion`, stessi dati nel
backend, nuvole dei nostri strumenti, pipeline completa con movimento spento.
Il profilo CUDA 12.1 non copre Blackwell; serve un profilo separato verificato.
Per la presa fisica restano necessarie pinza compatibile, calibrazione TCP e
camera, profondità misurata, controlli di collisione/IK/traiettoria e contatto.
`physical_grasp_validated` rimane `false` in tutti i risultati GraspGen.
