#!/usr/bin/env bash
# Esegue uno script del progetto col Python di Windows (ambiente conda `realsense`), che vede la camera RealSense
# con i driver nativi di Windows. Si lancia da WSL: lo script lavora nella cartella del progetto e i file che
# produce (registrazioni, calibrazioni) finiscono nel progetto. Guida: docs/realsense.md
#
#   bash scripts/rs_windows.sh scripts/realsense_probe.py
#   bash scripts/rs_windows.sh scripts/realsense_record.py --seconds 60
#   bash scripts/rs_windows.sh scripts/calibrate_camera.py --realsense --size 640x480 --fps 15
#
# Variabile opzionale: RS_WIN_PYTHON = percorso WSL del python.exe da usare (default: ambiente conda `realsense`).
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ $# -eq 0 ]]; then
    sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
fi

PY="${RS_WIN_PYTHON:-}"
if [[ -z "$PY" ]]; then
    # cmd.exe non accetta come cartella corrente un percorso WSL: lo si lancia da C:
    win_user="$(cd /mnt/c && cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')"
    for base in "Users/$win_user/miniconda3" "Users/$win_user/anaconda3" "Users/$win_user/miniforge3" \
                "Users/$win_user/AppData/Local/miniconda3" "ProgramData/miniconda3" "ProgramData/anaconda3"; do
        candidate="/mnt/c/$base/envs/realsense/python.exe"
        if [[ -f "$candidate" ]]; then
            PY="$candidate"
            break
        fi
    done
fi
if [[ -z "$PY" || ! -f "$PY" ]]; then
    echo "Python dell'ambiente conda 'realsense' di Windows non trovato." >&2
    echo "Crealo (docs/realsense.md) oppure indica il percorso: RS_WIN_PYTHON=/mnt/c/.../python.exe" >&2
    exit 1
fi
exec "$PY" "$@"
