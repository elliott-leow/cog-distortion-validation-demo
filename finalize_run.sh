#!/bin/bash
# Run after `python reference.py ...` completes. Substitutes paper placeholders
# from the new results.json, generates side-by-side behavioural completions,
# regenerates figures, and rebuilds the Colab notebook.
#
# Usage: ./finalize_run.sh [device]
#   device defaults to mps; pass "cpu" or "cuda" to override.

set -euo pipefail
cd "$(dirname "$0")"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

DEVICE="${1:-mps}"

echo "==> fill_paper.py"
python3 fill_paper.py

echo "==> regenerate_figures.py"
python3 regenerate_figures.py

echo "==> behavioral_demo.py (--device $DEVICE)"
python3 behavioral_demo.py --device "$DEVICE" || echo "behavioral_demo failed (non-blocking)"

echo "==> build_notebook.py"
python3 build_notebook.py

echo
echo "Done. See:"
echo "  paper/paper.md            (placeholders filled)"
echo "  results/behavioral_demo.md (qualitative side-by-side)"
echo "  notebook.ipynb            (refreshed Colab notebook)"
echo "  figures/*.png             (refreshed)"
