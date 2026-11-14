#!/usr/bin/env bash
# Foreground project-local Ollama service. Keep this terminal open during a full run.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="$ROOT/.local/ollama/bin/ollama"
if [[ ! -x "$BINARY" ]]; then
  echo "Project-local Ollama is missing: complete the local service installation first." >&2
  exit 1
fi
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS="$ROOT/.local/ollama/models"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=0  # Release LLM RAM before SAM3 CPU loads its weights.
mkdir -p "$OLLAMA_MODELS"
exec "$BINARY" serve
