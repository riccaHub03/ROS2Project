# Architettura della pipeline

## 📋 Architettura della Pipeline (`run_pipeline_sim.py`)

La catena di montaggio in simulazione esegue 6 stadi. Ogni stadio è un modulo indipendente
(`python -m <modulo> --help`) che **legge i report JSON degli stadi precedenti** e **scrive un
proprio report JSON** (più gli artefatti) in `data/outputs/` (o `--out-dir` / `$PIPELINE_OUTPUT_DIR`).

```
[Istruzione Clinica]
        │
        ▼
plan            python -m pipeline.plan              ► plan.json
        │
        ▼
capture         python -m pipeline.sim.capture       ► sim_capture.json
        │                                              (+ sim_capture_rgb.png, sim_capture_depth.npy)
        ▼
segment (SAM 3) python -m pipeline.sim.segment       ► sim_segmentation.json
        │                                              (+ sim_segmentation_mask.png, _overlay.png,
        │                                                 e di tutti gli strumenti: _all_mask.png, _all_overlay.png)
        ▼
reference_pose  python -m pipeline.sim.reference_pose► sim_reference_pose.json
        │                                              (+ sim_reference_pose.png)
        ▼
deproject       python -m pipeline.sim.deproject     ► sim_deprojection.json
        │                                              (+ sim_deprojection_cloud.ply, _frames.png)
        ▼
grasp           python -m pipeline.grasp             ► grasp.json
                                                       (+ grasp_zones.ply, grasp_overview.png, grasp_candidates.png)
                                                       10 candidati di presa, ciascuno verificato (IK, limiti,
                                                       collisioni) in un modello PyBullet della cella
```

Di default (`--mode all`) `capture` e `segment` lavorano sull'**intera collezione**: RGB-D e maschera di tutti
gli strumenti. Con `--gui`/`--headless` la finestra mostra sempre lo STL trasportato da PyBullet, e la
collezione viene resa fuori schermo dopo la presa (`python -m pipeline.sim.overview` ► `sim_overview.json`,
`sim_overview_capture_rgb.png`, `sim_overview_capture_depth_vis.png`, `sim_overview_segmentation_all_mask.png`).
`--mode single` ripristina il comportamento precedente (solo lo STL).

La pipeline reale (`run_pipeline_real.py`) usa `pipeline.real.{capture,segment,deproject}` e
produce `real_capture.json`, `real_segmentation.json`, `real_deprojection.json`, poi `grasp.json`.

### Contratto dei report (schema v2)

```json
{
  "schema_version": 2, "stage": "sim_segmentation", "status": "COMPLETED | FAILED",
  "timestamp": "...", "duration_s": 1.2,
  "inputs":    {"plan": ".../plan.json", "capture": ".../sim_capture.json"},
  "artifacts": {"mask_png": "...", "overlay_png": "..."},
  "data":      { "...payload dello stadio..." },
  "warnings":  [], "error": null
}
```

Se uno stadio fallisce scrive comunque il proprio report con `status: FAILED` e `error`
(tipo, messaggio, traceback) e termina con exit code 1: il JSON indica subito dove si è rotto.
Gli stadi a valle rifiutano report `FAILED` o di schema diverso.

### Esecuzione e debug stadio per stadio

```bash
python run_pipeline_sim.py --tool Scalpel                 # intera catena
python run_pipeline_sim.py --tool Scalpel --from-stage segment --stop-after deproject
python -m pipeline.sim.segment --backend ground-truth     # un solo stadio, input dai report su disco
python -m pipeline.grasp --deprojection data/outputs/real_deprojection.json
```

### Pulizia e test degli stadi singoli

```bash
python -m pipeline.clean            # anteprima dei file generati in data/outputs
python -m pipeline.clean --yes      # cancella (.gitkeep e README.md restano)
python -m unittest tests.test_standalone -v   # ogni stadio da solo, da cartella vuota
```

Tutti i report (e gli artefatti) vengono scritti in `data/outputs/` (o `--out-dir`); `--report NOME.json`
è un nome di file creato dentro quella cartella.

Codice condiviso in `pipeline/common/` (`paths`, `report`, `cli`, `scene`, `vision`, `reconstruction`, `ply`).
Test: `python -m unittest discover -s tests`.

---
