#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${QUIZ3_PYTHON:-}" ]]; then
    PYTHON_BIN="$QUIZ3_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/python.exe" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/python.exe"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    printf '%s\n' 'Activate MMIP_LLM or set QUIZ3_PYTHON to its Python executable.' >&2
    exit 1
fi
exec "$PYTHON_BIN" -X utf8 "$SCRIPT_DIR/Code/run_all.py"
