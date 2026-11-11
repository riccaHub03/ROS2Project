#!/usr/bin/env bash
# Simulazione grafica KUKA con SAM 3. Esegui dentro il container (o in un ambiente Python con le dipendenze).
#   bash scripts/run_sim_gui.sh                          # SAM 3
#   bash scripts/run_sim_gui.sh --vision ground-truth    # test rapido senza AI
# Ogni argomento viene passato a run_pipeline_sim.py (vedi --help).
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -x .venv/bin/python ]]; then PYTHON_BIN=".venv/bin/python"; else PYTHON_BIN="python3"; fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
export PYTHONUNBUFFERED=1

echo "Avvio simulazione grafica KUKA con SAM3."
echo "Test rapido senza AI: bash scripts/run_sim_gui.sh --vision ground-truth"
echo "Comandi nella finestra: P pausa, R ripeti movimento, Q chiudi."
exec "$PYTHON_BIN" run_pipeline_sim.py --gui --vision sam3 --tool Scissors "$@"
