#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

# Seleziona l'interprete Python (virtualenv locale o python3 di sistema/container)
if [[ -x .venv/bin/python ]]; then
    PYTHON_BIN=".venv/bin/python"
elif [[ -x ../.venv/bin/python ]]; then
    PYTHON_BIN="../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Errore: Nessun interprete python3 o .venv trovato." >&2
    exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-$project_dir/.cache/matplotlib}"
export PYTHONUNBUFFERED=1

echo "==========================================================================="
echo "Controllore KUKA LBR iiwa: Modalità W-S-A-D & Monitor Chirurgico Endoscopico"
echo "==========================================================================="
echo "Comandi Principali da Tastiera (W-S-A-D):"
echo "  * W / S: Avanti (+X) / Indietro (-X)"
echo "  * A / D: Sinistra (+Y) / Destra (-Y)"
echo "  * Q / E: Quota Z Su / Giu"
echo "  * SHIFT + [Tasto]: MODALITÀ TURBO SCATTO VELOCE (2.2x)"
echo "  * + / -: Regola la velocità di base (fino a 1.2 m/s)"
echo "  * U / O (oppure J / L): Ruota orientamento Yaw pinza"
echo "  * Spazio: Apri / Chiudi pinza"
echo "  * G (oppure Click Destro): Presa / Rilascio dello strumento"
echo "  * F: Mostra / Nascondi finestra Monitor Chirurgico Endoscopico"
echo "  * H: Ripristina posa HOME"
echo "  * V oppure C: Alterna tra la vista telecamera e la vista 3D libera"
echo "Modalità Puntatore Mouse:"
echo "  * Premi la ROTELLA DEL MOUSE (oppure il tasto M) per attivare/disattivare"
echo "    la modalità in cui il robot segue il puntatore del mouse sul vassoio."
echo "==========================================================================="

exec "$PYTHON_BIN" interactive_controller.py "$@"
