#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if [[ ! -x .venv/bin/python ]]; then
    echo "Manca .venv. Crea l'ambiente Python e installa requirements.txt." >&2
    exit 1
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-$project_dir/.cache/matplotlib}"
export PYTHONUNBUFFERED=1
echo "Avvio simulazione grafica KUKA con SAM3."
echo "Test rapido senza AI: bash run_graphical.sh --vision ground-truth"
echo "Comandi nella finestra: P pausa, R ripeti movimento, Q chiudi."
exec .venv/bin/python run_pipeline_sim.py --gui --vision sam3 --tool Scissors "$@"
