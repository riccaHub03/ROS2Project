#!/usr/bin/env bash
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$project_dir/run_graphical.sh" "$@"
