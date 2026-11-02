#!/usr/bin/env bash
# =============================================================================
# extract_deps.sh – Analizza i Dockerfile delle repo in src/ ed estrae
#                   le dipendenze APT e Python in file utilizzabili dal
#                   Dockerfile.v2.
#
# Uso:  ./scripts/extract_deps.sh [WORKSPACE_SRC_DIR]
#       Default WORKSPACE_SRC_DIR = ./ROS2_workspace/src
#
# Output:
#   .docker/apt-packages.txt    – pacchetti APT (uno per riga, deduplicati)
#   .docker/pip-packages.txt    – pacchetti pip inline (uno per riga)
#   .docker/requirements.txt    – contenuto aggregato dei requirements.txt
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="${1:-$PROJECT_ROOT/ROS2_workspace/src}"
OUT_DIR="$PROJECT_ROOT/.docker"

mkdir -p "$OUT_DIR"

# File temporanei
APT_TMP=$(mktemp)
PIP_TMP=$(mktemp)
REQ_TMP=$(mktemp)
trap 'rm -f "$APT_TMP" "$PIP_TMP" "$REQ_TMP" /tmp/_processed_reqs' EXIT
: > /tmp/_processed_reqs

echo "🔍 Scansione Dockerfile in: $SRC_DIR"
echo "──────────────────────────────────────────"

# Cerca tutti i Dockerfile nelle sotto-directory di src/
DOCKERFILES=$(find "$SRC_DIR" -name 'Dockerfile' -not -path '*/build/*' -not -path '*/install/*' -not -path '*/log/*' 2>/dev/null)

if [ -z "$DOCKERFILES" ]; then
    echo "⚠️  Nessun Dockerfile trovato in $SRC_DIR"
    exit 1
fi

for df in $DOCKERFILES; do
    REPO_NAME=$(echo "$df" | sed "s|$SRC_DIR/||" | cut -d'/' -f1)
    echo "  📦 $REPO_NAME → $df"

    # ── Estrai pacchetti APT ──
    # Cerca pattern: apt-get install ... (con eventuali backslash-newline)
    # Unisce le righe continuation (\) e poi estrae i nomi pacchetti
    sed ':a;/\\$/N;s/\\\n//;ta' "$df" | \
        grep -E 'apt-get install' | \
        sed 's/.*apt-get install[^\\]*//' | \
        sed 's/--[a-z-]*//g' | \
        sed 's/&&.*//g' | \
        sed 's/\\//g' | \
        tr ' ' '\n' | \
        sed 's/^-y$//' | \
        grep -v '^\s*$' | \
        sed "s/^/# from: $REPO_NAME\n/" \
        >> "$APT_TMP" 2>/dev/null || true

    # Metodo più robusto: cerca le righe con pacchetti apt individuali
    sed ':a;/\\$/N;s/\\\n//;ta' "$df" | \
        grep -oP '(?:ros-humble-[a-z0-9-]+|python3-[a-z0-9-]+|git\b|cmake\b|curl\b|wget\b)' \
        >> "$APT_TMP" 2>/dev/null || true

    # ── Estrai pacchetti pip ──
    # Cerca: pip install / pip3 install
    sed ':a;/\\$/N;s/\\\n//;ta' "$df" | \
        grep -E 'pip[3]?\s+install' | \
        grep -v '\-r ' | \
        sed 's/.*pip[3]\? install//' | \
        sed 's/--[a-z-]*\s*\S*//g' | \
        sed 's/&&.*//g' | \
        tr ' ' '\n' | \
        grep -v '^\s*$' | \
        grep -v '^--' \
        >> "$PIP_TMP" 2>/dev/null || true

    # ── Cerca requirements.txt referenziati ──
    REQ_REFS=$(grep -oP '(?<=-r\s)\S+' "$df" 2>/dev/null || true)
    REPO_DIR=$(dirname "$df")
    for req_ref in $REQ_REFS; do
        # Risolvi il path relativo al contesto del Dockerfile
        REQ_PATH="$REPO_DIR/$req_ref"
        # Prova anche path assoluti comuni nei Dockerfile
        if [ ! -f "$REQ_PATH" ]; then
            REQ_PATH=$(find "$REPO_DIR/.." -name "$(basename "$req_ref")" -maxdepth 2 2>/dev/null | head -1)
        fi
        if [ -f "$REQ_PATH" ]; then
            echo "    📄 Trovato requirements: $REQ_PATH"
            echo "# ── from: $REPO_NAME ──" >> "$REQ_TMP"
            grep -v '^\s*#' "$REQ_PATH" | grep -v '^\s*$' >> "$REQ_TMP" 2>/dev/null || true
            echo "" >> "$REQ_TMP"
        fi
    done

    # Cerca requirements.txt nella root della repo e nella dir del Dockerfile
    for req_file in "$REPO_DIR/requirements.txt" "$(dirname "$REPO_DIR")/requirements.txt"; do
        if [ -f "$req_file" ]; then
            REAL_REQ=$(realpath "$req_file" 2>/dev/null || echo "$req_file")
            # Evita duplicati controllando se il path è già stato processato
            if ! grep -qF "$REAL_REQ" /tmp/_processed_reqs 2>/dev/null; then
                echo "$REAL_REQ" >> /tmp/_processed_reqs
                echo "    📄 Trovato requirements (auto): $req_file"
                echo "# ── from: $REPO_NAME (auto-detected) ──" >> "$REQ_TMP"
                grep -v '^\s*#' "$req_file" | grep -v '^\s*$' >> "$REQ_TMP" 2>/dev/null || true
                echo "" >> "$REQ_TMP"
            fi
        fi
    done
done

echo ""
echo "──────────────────────────────────────────"
echo "📝 Generazione file dipendenze..."

# ── Deduplicazione e ordinamento APT ──
sort -u "$APT_TMP" | grep -v '^#' | grep -v '^\s*$' | sort > "$OUT_DIR/apt-packages.txt"
APT_COUNT=$(wc -l < "$OUT_DIR/apt-packages.txt" 2>/dev/null || echo 0)
echo "  ✅ .docker/apt-packages.txt ($APT_COUNT pacchetti)"

# ── Requirements aggregati ──
if [ -s "$REQ_TMP" ]; then
    cp "$REQ_TMP" "$OUT_DIR/requirements.txt"
    REQ_COUNT=$(grep -cvE '^\s*$|^\s*#' "$OUT_DIR/requirements.txt" 2>/dev/null || echo 0)
    echo "  ✅ .docker/requirements.txt ($REQ_COUNT dipendenze)"
else
    echo "# Nessun requirements.txt trovato nelle repo" > "$OUT_DIR/requirements.txt"
    echo "  ℹ️  .docker/requirements.txt (vuoto)"
fi

# ── Deduplicazione pip ──
if [ -s "$PIP_TMP" ]; then
    sort -u "$PIP_TMP" | grep -v '^\s*$' > "$OUT_DIR/pip-packages.txt"
else
    : > "$OUT_DIR/pip-packages.txt"
fi
PIP_COUNT=$(wc -l < "$OUT_DIR/pip-packages.txt" 2>/dev/null || echo 0)
echo "  ✅ .docker/pip-packages.txt ($PIP_COUNT pacchetti)"

echo ""
echo "🎯 Fatto! I file sono in: $OUT_DIR/"
echo "   Puoi ora buildare con: docker build -f Dockerfile.v2 -t ambiente_robotica_v2 ."
