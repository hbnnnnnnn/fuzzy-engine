#!/bin/bash
# =============================================================================
# Fix all missing dependencies discovered from run_all.sh failures
# Run this ONCE before re-running run_all.sh
# =============================================================================
set -o pipefail
CONDA_BASE="/mnt/mmlab2024nas/ldtuan/miniconda3"
source "${CONDA_BASE}/bin/activate"

# Timeout (seconds) for pip install commands to prevent hangs on NFS / source builds
PIP_TIMEOUT=180

# Helper: install packages into the correct conda env
# PYTHONNOUSERSITE=1 prevents bleeding of ~/.local/lib/pythonX.Y/site-packages
# NOTE: Do NOT use --target — it bypasses Python version/ABI checks and installs
#       wrong-ABI wheels (e.g. cp310 .so into a Python 3.9 env).
env_pip_install() {
    local envname="$1"; shift
    PYTHONNOUSERSITE=1 timeout "$PIP_TIMEOUT" conda run -n "$envname" \
        python -m pip install --upgrade "$@"
}
env_python() {
    local envname="$1"; shift
    PYTHONNOUSERSITE=1 conda run -n "$envname" python "$@"
}

echo "=========================================="
echo "  Fixing all environment dependencies"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. geneval env: mmcv CUDA extensions not compiled
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Fixing geneval env: mmcv._ext missing"
GENEVAL_TORCH=$(env_python geneval -c "import torch; print(torch.__version__)" 2>/dev/null)
GENEVAL_CUDA=$(env_python geneval -c "import torch; print(torch.version.cuda)" 2>/dev/null)
echo "  geneval torch=$GENEVAL_TORCH cuda=$GENEVAL_CUDA"

# Determine CUDA short version for the URL
CUDA_SHORT=""
case "$GENEVAL_CUDA" in
    11.6*) CUDA_SHORT="cu116";;
    11.7*) CUDA_SHORT="cu117";;
    11.8*) CUDA_SHORT="cu118";;
    12.1*) CUDA_SHORT="cu121";;
    12.4*) CUDA_SHORT="cu124";;
    12.6*) CUDA_SHORT="cu126";;
    12.8*) CUDA_SHORT="cu128";;
    *)     CUDA_SHORT="cu121";; # fallback
esac

# Determine torch version for URL (strip +cuXXX suffix, then major.minor.0)
TORCH_CLEAN="${GENEVAL_TORCH%%+*}"   # e.g. "2.8.0"
TORCH_URL_VER=""
if [[ -n "$TORCH_CLEAN" ]]; then
    TMAJ="${TORCH_CLEAN%%.*}"
    TREST="${TORCH_CLEAN#*.}"
    TMIN="${TREST%%.*}"
    TORCH_URL_VER="torch${TMAJ}.${TMIN}.0"
fi

echo "  Will try: $CUDA_SHORT / $TORCH_URL_VER"

# mmcv-full 1.7.2 only has pre-built wheels up to ~torch 2.1 / cu121.
# For newer torch (>=2.2), install mmcv>=2.0 (the successor) instead.
INSTALLED_MMCV=false
TMAJ="${TORCH_CLEAN%%.*}"
TREST="${TORCH_CLEAN#*.}"
TMIN="${TREST%%.*}"

if [[ "$TMAJ" -ge 2 && "$TMIN" -ge 2 ]] || [[ "$TMAJ" -ge 3 ]]; then
    echo "  torch >= 2.2 detected — installing mmcv>=2.0 (mmcv-full is obsolete for this torch)"
    # System nvcc may not match torch's CUDA. Find best available CUDA toolkit.
    for cuda_dir in /usr/local/cuda-${GENEVAL_CUDA%%.*}.* /usr/local/cuda-${GENEVAL_CUDA%%.*} /usr/local/cuda; do
        if [[ -x "$cuda_dir/bin/nvcc" ]]; then
            BEST_CUDA_HOME="$cuda_dir"
            break
        fi
    done
    if [[ -n "$BEST_CUDA_HOME" ]]; then
        echo "  Using CUDA_HOME=$BEST_CUDA_HOME"
    fi
    # Try pre-built wheel first (fastest)
    if env_pip_install geneval "mmcv>=2.0" --only-binary=:all: \
            -f "https://download.openmmlab.com/mmcv/dist/$CUDA_SHORT/$TORCH_URL_VER/index.html" 2>&1; then
        echo "  SUCCESS: mmcv pre-built wheel installed"
        INSTALLED_MMCV=true
    else
        echo "  No pre-built wheel; trying source build with CUDA_HOME=$BEST_CUDA_HOME ..."
        # MAX_JOBS=2 to prevent OOM on NFS; CUDA_HOME to match torch's cuda
        if CUDA_HOME="${BEST_CUDA_HOME:-/usr/local/cuda}" MAX_JOBS=2 \
                env_pip_install geneval "mmcv>=2.0" --no-build-isolation 2>&1; then
            echo "  SUCCESS: mmcv>=2.0 built from source"
            INSTALLED_MMCV=true
        else
            echo "  FAILED or timed out installing mmcv>=2.0"
        fi
    fi
else
    # Try pre-built mmcv-full wheels (only-binary to prevent source builds that hang)
    for cu in "$CUDA_SHORT" cu121 cu118 cu117; do
        for tv in "$TORCH_URL_VER" "torch2.0.0" "torch1.13.0" "torch2.1.0"; do
            [[ -z "$tv" ]] && continue
            if ! $INSTALLED_MMCV; then
                echo "  Trying mmcv-full 1.7.2 for $cu / $tv ..."
                if env_pip_install geneval mmcv-full==1.7.2 --only-binary=:all: \
                        -f "https://download.openmmlab.com/mmcv/dist/$cu/$tv/index.html" 2>&1; then
                    echo "  SUCCESS: mmcv-full installed"
                    INSTALLED_MMCV=true
                fi
            fi
        done
    done
fi
if ! $INSTALLED_MMCV; then
    echo "  WARNING: Could not auto-install mmcv. Install manually."
fi

# ---------------------------------------------------------------------------
# 2. tifa env: missing clip (for DrawBench CLIPScore eval)
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Fixing tifa env: missing 'clip' module"
# First fix regex — previous --target install put a cp310 .so into this cp39 env
echo "  Fixing broken regex (wrong ABI from prior --target install)..."
PYTHONNOUSERSITE=1 conda run -n tifa python -m pip install --upgrade --force-reinstall regex 2>&1
env_pip_install tifa openai-clip
env_python tifa -c "import clip; print('clip OK')" 2>&1

# ---------------------------------------------------------------------------
# 3. t2icomp env: missing spacy, ruamel.yaml, word2number
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Fixing t2icomp env: missing spacy, ruamel.yaml, word2number"
# First fix numpy — previous --target install put numpy 2.x into this Python 3.9 env
# numpy>=2.0 requires Python>=3.10, so pin to <2
echo "  Fixing broken numpy (numpy 2.x incompatible with Python 3.9)..."
PYTHONNOUSERSITE=1 conda run -n t2icomp python -m pip install --upgrade --force-reinstall "numpy<2" 2>&1
# Force-reinstall srsly first to ensure C extensions (ujson) are built for the correct Python
PYTHONNOUSERSITE=1 timeout "$PIP_TIMEOUT" conda run -n t2icomp \
    python -m pip install --upgrade --force-reinstall --no-deps srsly 2>&1
env_pip_install t2icomp spacy "ruamel.yaml" word2number
# Download spacy English model into the env
env_python t2icomp -m spacy download en_core_web_sm 2>&1
echo "  Verifying..."
env_python t2icomp -c "import spacy; print('spacy OK:', spacy.__version__)" 2>&1
env_python t2icomp -c "import ruamel.yaml; print('ruamel.yaml OK')" 2>&1
env_python t2icomp -c "from word2number import w2n; print('word2number OK')" 2>&1

# ---------------------------------------------------------------------------
# 4. showo env: transformers too old (missing cache_utils)
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Fixing showo env: transformers too old (need >=4.36)"
# Fix numpy ABI clash — previous --target install put numpy 2.x which crashes with old torch
echo "  Fixing numpy for showo..."
PYTHONNOUSERSITE=1 conda run -n showo python -m pip install --upgrade --force-reinstall "numpy<2" 2>&1
# Install transformers with its tokenizers dependency (pin to <5.0 for torch 1.13 compat)
env_pip_install showo "transformers>=4.36,<5" "tokenizers>=0.22,<=0.23"
echo "  Verifying..."
env_python showo -c "from transformers.cache_utils import Cache, DynamicCache; print('transformers cache_utils OK')" 2>&1

# ---------------------------------------------------------------------------
# 5. emu3 env: already fixed urllib3, verify requests chain works
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Fixing emu3 env: urllib3+requests+charset_normalizer"
# Install charset_normalizer (needed by requests) and ensure requests is in the env
env_pip_install emu3 urllib3 requests charset-normalizer chardet 2>&1
env_python emu3 -c "import urllib3; import requests; print('emu3 OK: urllib3', urllib3.__version__)" 2>&1
# If still broken:
if ! env_python emu3 -c "import urllib3" 2>/dev/null; then
    echo "  Reinstalling urllib3+requests with force..."
    PYTHONNOUSERSITE=1 timeout "$PIP_TIMEOUT" conda run -n emu3 \
        python -m pip install --upgrade --force-reinstall urllib3 requests charset-normalizer 2>&1
fi

# ---------------------------------------------------------------------------
# 6. nexusgen env: flash_attn ABI mismatch
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Fixing nexusgen env: flash_attn ABI mismatch"
NEXUS_TORCH=$(env_python nexusgen -c "import torch; print(torch.__version__)" 2>/dev/null)
NEXUS_CUDA=$(env_python nexusgen -c "import torch; print(torch.version.cuda)" 2>/dev/null)
echo "  nexusgen torch=$NEXUS_TORCH cuda=$NEXUS_CUDA"
echo "  Uninstalling old flash-attn..."
PYTHONNOUSERSITE=1 timeout 60 conda run -n nexusgen python -m pip uninstall -y flash-attn 2>/dev/null
echo "  Reinstalling flash-attn (trying pre-built wheel first, then source build with ${PIP_TIMEOUT}s timeout)..."
# flash-attn >=2.0 requires torch >=2.0; for torch 1.13.x use flash-attn <2.0
FLASH_ATTN_SPEC="flash-attn"
NEXUS_TMAJ="${NEXUS_TORCH%%.*}"
NEXUS_TREST="${NEXUS_TORCH#*.}"
NEXUS_TMIN="${NEXUS_TREST%%.*}"
if [[ "$NEXUS_TMAJ" -le 1 ]]; then
    echo "  torch $NEXUS_TORCH detected — pinning flash-attn<2.0 for compatibility"
    FLASH_ATTN_SPEC="flash-attn<2.0"
fi
# Check if flash_attn already works before trying to reinstall
if env_python nexusgen -c "import flash_attn; print('flash_attn already OK:', flash_attn.__version__)" 2>&1; then
    echo "  flash_attn is working — skipping reinstall"
else
    # Try pre-built wheel first to avoid long source compile on NFS
    if ! PYTHONNOUSERSITE=1 timeout "$PIP_TIMEOUT" conda run -n nexusgen python -m pip install --upgrade "$FLASH_ATTN_SPEC" --only-binary=:all: 2>&1; then
        echo "  No pre-built wheel found; building from source (this may take a while)..."
        PYTHONNOUSERSITE=1 MAX_JOBS=2 timeout 600 conda run -n nexusgen python -m pip install --upgrade "$FLASH_ATTN_SPEC" --no-build-isolation 2>&1 | tail -10
    fi
    echo "  Verifying..."
    env_python nexusgen -c "import flash_attn; print('flash_attn OK:', flash_attn.__version__)" 2>&1
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Fix script complete."
echo "=========================================="
echo ""
echo "Before re-running, clean empty output dirs:"
echo "  find /mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/eval_models/outputs -type d -empty -delete"
echo ""
echo "Then re-run:"
echo "  bash /mnt/mmlab2024nas/ldtuan/code/ndbao_hbngoc/eval_models/run_all.sh"
