#!/bin/bash
#SBATCH --job-name=setup-nexus
#SBATCH --output=setup_nexus_%j.out
#SBATCH --error=setup_nexus_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch

# =============================================================================
# Setup script for the "nexus" conda environment
#
# Installs all libraries from requirements.txt into the pre-existing "nexus"
# conda environment (Python 3.10).
#
# Key considerations:
#   - PyTorch 2.4.0 + CUDA 12.1 must be installed FIRST (many packages
#     depend on it at build time).
#   - flash-attn 2.8.3 compiles from source: needs nvcc, ninja, and
#     --no-build-isolation (so it can see the already-installed torch).
#   - faiss-gpu 1.7.2 on PyPI was built against older CUDA. We install
#     faiss-gpu-cu12 (CUDA 12.x native) instead, which is API-compatible.
#   - DiffSynth-Studio is already cloned locally at the exact commit
#     (afd101f) referenced in requirements.txt — we install it from the
#     local path rather than re-cloning from GitHub.
#   - deepspeed JIT-compiles CUDA ops at runtime by default, but we
#     still need nvcc available.
# =============================================================================

set -e  # exit on error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INSTALL_DIR="${INSTALL_DIR:-/media02/nthuy/ndbao}"
CONDA_ENV="nexus"
DIFFSYNTH_LOCAL="${INSTALL_DIR}/Nexus-Gen/DiffSynth-Studio"

# ---------------------------------------------------------------------------
# Fix cross-device link error (Errno 18)
# ---------------------------------------------------------------------------
# The compute node's /tmp is on a different filesystem than /media02.
# pip's build/download uses TMPDIR, but the wheel cache is on /media02.
# os.rename() across filesystems fails with "Invalid cross-device link".
# Fix: point TMPDIR to a directory on the same filesystem as pip's cache.
export TMPDIR="/media02/nthuy/.pip_tmp"
mkdir -p "$TMPDIR"

# ---------------------------------------------------------------------------
# Initialise conda
# ---------------------------------------------------------------------------
module purge
source /media02/nthuy/miniconda3/bin/activate

echo "============================================================"
echo "  Nexus Environment Setup"
echo "  Node:       $(hostname)"
echo "  Date:       $(date)"
echo "  Conda env:  ${CONDA_ENV}"
echo "  Python:     $(/media02/nthuy/miniconda3/envs/${CONDA_ENV}/bin/python --version 2>&1)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Activate the environment
# ---------------------------------------------------------------------------
conda activate "${CONDA_ENV}"

# ============================================================
# Phase 0 — Build tools (needed by flash-attn, deepspeed)
# ============================================================
echo ""
echo ">>> [Phase 0] Installing build tools"
echo "------------------------------------------------------------"
pip install --quiet packaging psutil ninja setuptools wheel

# ============================================================
# Phase 1 — PyTorch + CUDA 12.1 (must come first)
# ============================================================
echo ""
echo ">>> [Phase 1] Installing PyTorch 2.4.0 + CUDA 12.1"
echo "------------------------------------------------------------"
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# triton==3.0.0 is pulled in automatically by torch 2.4.0

# Verify CUDA is visible
python3 -c "import torch; print('  torch', torch.__version__, '| cuda', torch.version.cuda, '| available:', torch.cuda.is_available())"

# ============================================================
# Phase 2 — Packages that compile CUDA kernels
#            (need torch + nvcc already present)
# ============================================================
echo ""
echo ">>> [Phase 2] Installing CUDA-compiled packages"
echo "------------------------------------------------------------"

# --- flash-attn (compiles from source, ~5-10 min with ninja) ---
echo "  Installing flash-attn==2.8.3 (building from source)..."
MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation

# --- deepspeed (JIT-compiles ops at runtime, pip install is quick) ---
echo "  Installing deepspeed==0.18.2..."
pip install deepspeed==0.18.2

# ============================================================
# Phase 3 — faiss-gpu (CUDA 12 native build)
# ============================================================
echo ""
echo ">>> [Phase 3] Installing faiss-gpu (CUDA 12)"
echo "------------------------------------------------------------"
# faiss-gpu==1.7.2 on PyPI was built against old CUDA.
# faiss-gpu-cu12 provides CUDA-12-native binaries and is API-compatible.
pip install faiss-gpu-cu12

# ============================================================
# Phase 4 — DiffSynth-Studio (from local checkout)
# ============================================================
echo ""
echo ">>> [Phase 4] Installing DiffSynth-Studio (local)"
echo "------------------------------------------------------------"
# The local repo at Nexus-Gen/DiffSynth-Studio is already at commit
# afd101f3452c — the exact commit specified in requirements.txt line 32.
#
# DiffSynth's setup.py uses "import pkg_resources" which was removed in
# newer setuptools (>=74). Using --no-build-isolation avoids pip creating
# a fresh build env with the latest setuptools that lacks pkg_resources.
# We ensure setuptools<74 is present in the env for this to work.
pip install --quiet "setuptools<74"
if [ -d "${DIFFSYNTH_LOCAL}" ]; then
    echo "  Installing from: ${DIFFSYNTH_LOCAL}"
    pip install -e "${DIFFSYNTH_LOCAL}" --no-build-isolation
else
    echo "  WARNING: ${DIFFSYNTH_LOCAL} not found!"
    echo "  Falling back to git install from GitHub..."
    pip install "git+https://github.com/modelscope/DiffSynth-Studio.git@afd101f3452c9ecae0c87b79adfa2e22d65ffdc3#egg=diffsynth" --no-build-isolation
fi

# ============================================================
# Phase 5 — Remaining pip packages (bulk install)
# ============================================================
echo ""
echo ">>> [Phase 5] Installing remaining packages"
echo "------------------------------------------------------------"

# Phase 4 (DiffSynth) already installed latest versions of several packages
# (transformers, accelerate, peft, huggingface-hub, safetensors, etc.).
# We install datasets FIRST without pinning fsspec, because:
#   datasets==4.4.1 requires fsspec<=2025.10.0
#   but the requirements.txt pins fsspec==2026.2.0 (captured by pip freeze
#   AFTER torch upgraded it — the old env had a compatible older version).
# Letting datasets pull its own fsspec is safe; torch only needs "fsspec".
echo "  Installing datasets (needs fsspec<=2025.10.0)..."
pip install --quiet datasets==4.4.1

# Now install the remaining packages. We drop exact pins for packages that:
#   - are already installed by earlier phases (torch, DiffSynth deps)
#   - would conflict with what DiffSynth pulled in
# Using flexible pins where the exact version from freeze is incompatible.
pip install --quiet \
    absl-py==2.3.1 \
    accelerate==1.11.0 \
    addict==2.4.0 \
    aiofiles==24.1.0 \
    aiohappyeyeballs==2.6.1 \
    aiohttp==3.13.2 \
    aiosignal==1.4.0 \
    aliyun-python-sdk-core==2.16.0 \
    aliyun-python-sdk-kms==2.16.5 \
    annotated-doc==0.0.4 \
    annotated-types==0.7.0 \
    anyio==4.12.1 \
    async-timeout==5.0.1 \
    attrdict==2.0.1 \
    attrs==25.4.0 \
    av==16.0.1 \
    binpacking==1.5.2 \
    bitsandbytes==0.48.2 \
    brotli==1.2.0 \
    cffi==2.0.0 \
    click==8.3.1 \
    contourpy==1.3.2 \
    cpm-kernels==1.0.11 \
    crcmod==1.7 \
    cryptography==46.0.3 \
    cycler==0.12.1 \
    dacite==1.9.2 \
    dill==0.4.0 \
    distro==1.9.0 \
    fastapi==0.121.2 \
    ffmpy==1.0.0 \
    fonttools==4.60.1 \
    future==1.0.0 \
    gradio==5.49.1 \
    gradio_client==1.13.3 \
    groovy==0.1.2 \
    grpcio==1.76.0 \
    importlib_metadata==8.7.0 \
    jieba==0.42.1 \
    jiter==0.12.0 \
    jmespath==0.10.0 \
    joblib==1.5.2 \
    kiwisolver==1.4.9 \
    lightning==2.6.1 \
    lightning-utilities==0.15.2 \
    Markdown==3.10 \
    matplotlib==3.10.7 \
    modelscope==1.19.1 \
    ms_swift==3.3.0 \
    multidict==6.7.0 \
    multiprocess==0.70.18 \
    nltk==3.9.2 \
    numpy==1.26.4 \
    openai==2.8.1 \
    orjson==3.11.4 \
    oss2==2.19.1 \
    peft==0.15.2 \
    pillow==11.3.0 \
    propcache==0.4.1 \
    pyarrow==22.0.0 \
    pycparser==2.23 \
    pycryptodome==3.23.0 \
    pydub==0.25.1 \
    pyparsing==3.2.5 \
    python-multipart==0.0.20 \
    pytorch-lightning==2.6.1 \
    qwen-vl-utils==0.0.6 \
    rouge==1.0.1 \
    ruff==0.14.5 \
    safehttpx==0.1.7 \
    scikit-learn==1.7.2 \
    scipy==1.15.3 \
    semantic-version==2.10.0 \
    simplejson==3.20.2 \
    sniffio==1.3.1 \
    sortedcontainers==2.4.0 \
    starlette==0.49.3 \
    tensorboard==2.20.0 \
    tensorboard-data-server==0.7.2 \
    threadpoolctl==3.6.0 \
    tiktoken==0.12.0 \
    tokenizers==0.21.4 \
    tomlkit==0.13.3 \
    torchmetrics==1.8.2 \
    transformers==4.49.0 \
    transformers-stream-generator==0.0.5 \
    trl==0.16.1 \
    typer==0.24.1 \
    typer-slim==0.24.0 \
    uvicorn==0.38.0 \
    websockets==15.0.1 \
    Werkzeug==3.1.3 \
    xxhash==3.6.0 \
    yarl==1.22.0 \
    zipp==3.23.0 \
    zstandard==0.25.0

# Note: nvidia-cublas-cu12, nvidia-cuda-cupti-cu12, nvidia-cuda-nvrtc-cu12,
# nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, nvidia-cufft-cu12,
# nvidia-curand-cu12, nvidia-cusolver-cu12, nvidia-cusparse-cu12,
# nvidia-nccl-cu12, nvidia-nvjitlink-cu12, nvidia-nvtx-cu12
# are already installed as dependencies of torch==2.4.0 (cu121).
# No need to install them separately.

# ============================================================
# Phase 6 — Verification
# ============================================================
echo ""
echo ">>> [Phase 6] Verification"
echo "------------------------------------------------------------"

python3 - <<'PYEOF'
import sys
print(f"Python: {sys.version}")

errors = []

# Core
try:
    import torch
    print(f"  torch:        {torch.__version__} | CUDA: {torch.version.cuda} | GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
except Exception as e:
    errors.append(f"torch: {e}")

# flash-attn
try:
    import flash_attn
    print(f"  flash_attn:   {flash_attn.__version__}")
except Exception as e:
    errors.append(f"flash_attn: {e}")

# deepspeed
try:
    import deepspeed
    print(f"  deepspeed:    {deepspeed.__version__}")
except Exception as e:
    errors.append(f"deepspeed: {e}")

# faiss
try:
    import faiss
    ngpus = faiss.get_num_gpus()
    print(f"  faiss-gpu:    OK | GPUs detected: {ngpus}")
except Exception as e:
    errors.append(f"faiss: {e}")

# diffsynth
try:
    import diffsynth
    print(f"  diffsynth:    OK")
except Exception as e:
    errors.append(f"diffsynth: {e}")

# transformers
try:
    import transformers
    print(f"  transformers: {transformers.__version__}")
except Exception as e:
    errors.append(f"transformers: {e}")

# bitsandbytes
try:
    import bitsandbytes
    print(f"  bitsandbytes: {bitsandbytes.__version__}")
except Exception as e:
    errors.append(f"bitsandbytes: {e}")

# accelerate
try:
    import accelerate
    print(f"  accelerate:   {accelerate.__version__}")
except Exception as e:
    errors.append(f"accelerate: {e}")

# peft
try:
    import peft
    print(f"  peft:         {peft.__version__}")
except Exception as e:
    errors.append(f"peft: {e}")

# gradio
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

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
if [ $VERIFY_EXIT -eq 0 ]; then
    echo "  Setup COMPLETE — all packages installed and verified."
else
    echo "  Setup FINISHED with WARNINGS — check verification output above."
fi
echo ""
echo "  To use:"
echo "    source /media02/nthuy/miniconda3/bin/activate"
echo "    conda activate ${CONDA_ENV}"
echo "    cd ${INSTALL_DIR}/Nexus-Gen"
echo ""
echo "  Date: $(date)"
echo "============================================================"

# Cleanup temp directory
rm -rf "$TMPDIR"
