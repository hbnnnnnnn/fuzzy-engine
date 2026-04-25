#!/bin/bash
#SBATCH --job-name=setup-nexus-uv
#SBATCH --output=setup_nexus_uv_%j.out
#SBATCH --error=setup_nexus_uv_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch

# =============================================================================
# Setup script for the "fuzzy-engine" uv virtual environment
#
# Uses uv for fast, reproducible environment management.
#
# Key considerations:
#   - PyTorch 2.4.0 + CUDA 12.1 is served from the pytorch CUDA index,
#     configured in pyproject.toml via [tool.uv.sources].
#   - flash-attn 2.8.3 compiles from source and needs no-build-isolation
#     (so it can see the already-installed torch headers). This is declared
#     in pyproject.toml via [tool.uv] no-build-isolation-package.
#   - faiss-gpu-cu12 provides CUDA-12-native binaries (replaces faiss-gpu
#     from PyPI which was built against older CUDA).
#   - DiffSynth-Studio is installed from the pinned git commit declared in
#     [tool.uv.sources]. If you have a local checkout at DIFFSYNTH_LOCAL,
#     set that env var to override and avoid a network clone.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFSYNTH_LOCAL="${DIFFSYNTH_LOCAL:-}"

# ---------------------------------------------------------------------------
# Check uv is available
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "uv not found — installing via the official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "============================================================"
echo "  Nexus Environment Setup (uv)"
echo "  Node:    $(hostname)"
echo "  Date:    $(date)"
echo "  uv:      $(uv --version)"
echo "============================================================"

cd "${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# Phase 1 — Sync environment (installs everything from pyproject.toml)
#
# uv creates .venv/ in the project root by default.
# flash-attn will be compiled in-tree (no-build-isolation is set in
# pyproject.toml), so torch must be installed before it. uv resolves the
# dependency order automatically.
# ---------------------------------------------------------------------------
echo ""
echo ">>> [Phase 1] Creating / syncing virtual environment"
echo "------------------------------------------------------------"

# If a local DiffSynth checkout exists, override the git source temporarily.
if [ -n "${DIFFSYNTH_LOCAL}" ] && [ -d "${DIFFSYNTH_LOCAL}" ]; then
    echo "  Using local DiffSynth-Studio at: ${DIFFSYNTH_LOCAL}"
    # Temporarily patch [tool.uv.sources] to use the local path.
    # uv path sources are declared as: diffsynth = { path = "..." }
    # We do a one-shot uv add to register the local path, then sync.
    uv add "diffsynth @ file://${DIFFSYNTH_LOCAL}" --no-build-isolation
else
    echo "  Using DiffSynth-Studio from git (as declared in pyproject.toml)"
fi

MAX_JOBS=4 uv sync --no-build-isolation

# ---------------------------------------------------------------------------
# Phase 2 — Verification
# ---------------------------------------------------------------------------
echo ""
echo ">>> [Phase 2] Verification"
echo "------------------------------------------------------------"

uv run python3 - <<'PYEOF'
import sys
print(f"Python: {sys.version}")

errors = []

try:
    import torch
    print(f"  torch:        {torch.__version__} | CUDA: {torch.version.cuda} | GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
except Exception as e:
    errors.append(f"torch: {e}")

try:
    import flash_attn
    print(f"  flash_attn:   {flash_attn.__version__}")
except Exception as e:
    errors.append(f"flash_attn: {e}")

try:
    import deepspeed
    print(f"  deepspeed:    {deepspeed.__version__}")
except Exception as e:
    errors.append(f"deepspeed: {e}")

try:
    import faiss
    ngpus = faiss.get_num_gpus()
    print(f"  faiss-gpu:    OK | GPUs detected: {ngpus}")
except Exception as e:
    errors.append(f"faiss: {e}")

try:
    import diffsynth
    print(f"  diffsynth:    OK")
except Exception as e:
    errors.append(f"diffsynth: {e}")

try:
    import transformers
    print(f"  transformers: {transformers.__version__}")
except Exception as e:
    errors.append(f"transformers: {e}")

try:
    import bitsandbytes
    print(f"  bitsandbytes: {bitsandbytes.__version__}")
except Exception as e:
    errors.append(f"bitsandbytes: {e}")

try:
    import accelerate
    print(f"  accelerate:   {accelerate.__version__}")
except Exception as e:
    errors.append(f"accelerate: {e}")

try:
    import peft
    print(f"  peft:         {peft.__version__}")
except Exception as e:
    errors.append(f"peft: {e}")

try:
    import gradio
    print(f"  gradio:       {gradio.__version__}")
except Exception as e:
    errors.append(f"gradio: {e}")

if errors:
    print("\n  ERRORS:")
    for err in errors:
        print(f"    - {err}")
    sys.exit(1)
else:
    print("\n  All critical packages verified successfully.")
PYEOF

VERIFY_EXIT=$?

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
if [ $VERIFY_EXIT -eq 0 ]; then
    echo "  Setup COMPLETE — environment at .venv/ is ready."
else
    echo "  Setup FINISHED with WARNINGS — check verification output above."
fi
echo ""
echo "  To activate:"
echo "    source ${SCRIPT_DIR}/.venv/bin/activate"
echo "  Or run commands directly with uv:"
echo "    uv run python your_script.py"
echo ""
echo "  Date: $(date)"
echo "============================================================"
