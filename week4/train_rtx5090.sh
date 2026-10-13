#!/usr/bin/env bash
# Git Bash / Linux Bash; activate MMIP first. No secrets are read from files.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${MMIP_PYTHON:-}"
CONFIG="${MMIP_CONFIG:-$SCRIPT_DIR/config.rtx5090.json}"
STAGE=all
DRY_RUN=0
SKIP_GEMINI=0
INSTALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      [[ $# -ge 2 ]] || { printf 'Missing --stage value\n' >&2; exit 2; }
      STAGE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-gemini) SKIP_GEMINI=1; shift ;;
    --install) INSTALL=1; shift ;;
    -h|--help)
      printf 'Usage: bash train_rtx5090.sh [--stage all|prepare|sentiment|vision|caption|gemini|report] [--install] [--skip-gemini] [--dry-run]\n'
      printf 'Variables: MMIP_PYTHON, MMIP_CONFIG, GEMINI_API_KEY, GEMINI_MODEL.\n'
      printf 'Missing/skipped Gemini: local work runs, but full workflow exits 3 (incomplete).\n'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
case "$STAGE" in
  all|prepare|sentiment|vision|caption|gemini|report) ;;
  *) printf 'Invalid stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac
if [[ -z "$PYTHON_BIN" && -n "${CONDA_PREFIX:-}" ]]; then
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
      PREFIX="$CONDA_PREFIX"
      if command -v cygpath >/dev/null 2>&1; then PREFIX="$(cygpath -u "$PREFIX")"; fi
      if [[ -f "$PREFIX/python.exe" ]]; then PYTHON_BIN="$PREFIX/python.exe"; fi ;;
    *)
      if [[ -x "$CONDA_PREFIX/bin/python" ]]; then PYTHON_BIN="$CONDA_PREFIX/bin/python"; fi ;;
  esac
fi
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3)"
  fi
fi
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    if [[ -n "$PYTHON_BIN" ]] && command -v cygpath >/dev/null 2>&1; then
      PYTHON_BIN="$(cygpath -u "$PYTHON_BIN")"
    fi ;;
esac
if [[ -z "$PYTHON_BIN" ]] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python not found. Use Git Bash with conda activate MMIP.\n' >&2
  printf 'PowerShell alternative: .\\week4_homework\\train_rtx5090.ps1\n' >&2
  printf 'PowerShell bash.exe may start WSL; Windows Conda does not activate inside WSL.\n' >&2
  exit 127
fi
[[ -f "$CONFIG" ]] || { printf 'Missing config: %s\n' "$CONFIG" >&2; exit 2; }
cd -- "$SCRIPT_DIR"
ARGS=("$PYTHON_BIN" -u "$SCRIPT_DIR/run_all.py" --config "$CONFIG" --device cuda --stage "$STAGE")
if [[ "$SKIP_GEMINI" -eq 1 ]]; then ARGS+=(--skip-gemini); fi
printf 'Interpreter: %s\nStage: %s\nCommand:' "$PYTHON_BIN" "$STAGE"
printf ' %q' "${ARGS[@]}"
printf '\n'
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'Dry run: no installation, downloads, API calls or training.\n'
  exit 0
fi
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
if [[ "$INSTALL" -eq 1 ]]; then
  "$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi
"$PYTHON_BIN" "$SCRIPT_DIR/check_gpu.py"
mkdir -p -- "$SCRIPT_DIR/logs"
LOG="$SCRIPT_DIR/logs/${STAGE}_$(date +%Y%m%d_%H%M%S).log"
# pipefail retains Python's exit status. Do not run two workflows concurrently.
set +e
"${ARGS[@]}" 2>&1 | tee -- "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
if [[ "$STATUS" -eq 3 ]]; then
  printf 'Local results saved; some assignment requirements remain pending. See completion.json.\n'
elif [[ "$STATUS" -ne 0 ]]; then
  printf 'Run failed (exit %s). Fix the cause and rerun to resume checkpoints. Log: %s\n' "$STATUS" "$LOG" >&2
else
  printf 'Requested stage completed. Log: %s\n' "$LOG"
fi
exit "$STATUS"
