#!/bin/bash
# Fix missing httpx dependency in janus environment
set -e

CONDA_BASE="/mnt/mmlab2024nas/ldtuan/miniconda3"
source "${CONDA_BASE}/bin/activate"

echo "Installing httpx in janus environment..."
PYTHONNOUSERSITE=1 conda run -n janus python -m pip install httpx

echo "Verifying installation..."
PYTHONNOUSERSITE=1 conda run -n janus python -c "import httpx; print(f'httpx version: {httpx.__version__}')"

echo "Testing transformers import (should work now)..."
PYTHONNOUSERSITE=1 conda run -n janus python -c "from transformers import AutoModelForCausalLM; print('Success: transformers import works')"

echo "Fix complete! You can now re-run './run_all.sh'"