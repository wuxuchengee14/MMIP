#!/usr/bin/env bash
set -euo pipefail

# Accept an explicit executable or resolve Conda's Windows Python before PATH.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${QUIZ2_PYTHON:-}" ]]; then
    PYTHON_BIN="$QUIZ2_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/python.exe" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/python.exe"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    printf '%s\n' 'Python not found. Activate MMIP_LLM or set QUIZ2_PYTHON to its python.exe.' >&2
    exit 1
fi
printf 'Python: %s\n' "$PYTHON_BIN"
"$PYTHON_BIN" "$SCRIPT_DIR/Code/prepare_benchmark.py"
"$PYTHON_BIN" "$SCRIPT_DIR/Code/setup_vlm.py" --check-processors
for MODEL in qwen smol; do
    if [[ -f "$SCRIPT_DIR/Output/full/$MODEL.json" ]]; then
        if "$PYTHON_BIN" -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); sys.exit(0 if r.get("complete") is True and len(r.get("results",[]))==20 else 1)' "$SCRIPT_DIR/Output/full/$MODEL.json"; then
            printf 'Keeping existing completed run: %s (will validate before scoring)\n' "$MODEL"
        else
            printf 'Incomplete %s run. Inspect the error, then rerun run_vlm.py --model %s --overwrite.\n' "$MODEL" "$MODEL" >&2
            exit 1
        fi
    else
        "$PYTHON_BIN" "$SCRIPT_DIR/Code/run_vlm.py" --model "$MODEL"
    fi
done
"$PYTHON_BIN" "$SCRIPT_DIR/Code/quiz2_tasks.py" status
"$PYTHON_BIN" "$SCRIPT_DIR/Code/quiz2_tasks.py" score
"$PYTHON_BIN" "$SCRIPT_DIR/Code/quiz2_tasks.py" export-judge
printf '%s\n' 'Local inference and objective scoring complete. Next: upload each cases/vision_XX/image.png with prompt.txt to a commercial vision model.'
printf '%s\n' 'Read Output/Commercial_manual/README.md, save all 20 review JSON files, then run import-judge and report.'
