#!/usr/bin/env bash
# Esegue uno script del progetto col Python di Windows, che vede la camera RealSense con i driver nativi di Windows.
# Si lancia da WSL: lo script lavora nella cartella del progetto e i file che produce finiscono nel progetto.
# Come creare l'ambiente (venv o conda): docs/realsense.md
#
#   bash scripts/rs_windows.sh scripts/realsense_probe.py
#   bash scripts/rs_windows.sh scripts/realsense_bridge.py send
#   bash scripts/rs_windows.sh scripts/realsense_record.py --seconds 60
#
# Python cercato, in ordine: $RS_WIN_PYTHON, venv %USERPROFILE%\.venvs\realsense, ambiente conda `realsense`.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ $# -eq 0 ]]; then
    sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
fi

PY="${RS_WIN_PYTHON:-}"
if [[ -z "$PY" ]]; then
    # cartella utente di Windows; cmd.exe non accetta come cartella corrente un percorso WSL: lo si lancia da C:
    win_profile="$(cd /mnt/c && cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')"
    profile="$(wslpath -u "$win_profile" 2>/dev/null || true)"
    for candidate in "$profile/.venvs/realsense/Scripts/python.exe" \
                     "$profile/miniconda3/envs/realsense/python.exe" \
                     "$profile/anaconda3/envs/realsense/python.exe" \
                     "$profile/miniforge3/envs/realsense/python.exe" \
                     "$profile/AppData/Local/miniconda3/envs/realsense/python.exe" \
                     /mnt/c/ProgramData/miniconda3/envs/realsense/python.exe \
                     /mnt/c/ProgramData/anaconda3/envs/realsense/python.exe; do
        if [[ -f "$candidate" ]]; then
            PY="$candidate"
            break
        fi
    done
fi
if [[ -z "$PY" || ! -f "$PY" ]]; then
    echo "Python di Windows per la RealSense non trovato (venv %USERPROFILE%\\.venvs\\realsense o ambiente conda 'realsense')." >&2
    echo "Crealo come in docs/realsense.md, oppure indica il percorso: RS_WIN_PYTHON=/mnt/c/.../python.exe" >&2
    exit 1
fi
exec "$PY" "$@"
