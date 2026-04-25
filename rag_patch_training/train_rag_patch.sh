#!/bin/bash
#SBATCH --job-name=rag-patch
#SBATCH --partition=002-partition-default
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/rag-patch_%j.out
#SBATCH --error=logs/rag-patch_%j.err

# =============================================================================
# RAG Patch Training — LoRA correction stream for FLUX DiT
# =============================================================================
# Trains a LoRA-injected FLUX DiT (Stream 2) that outputs an additive
# correction signal conditioned on VGG-19 style features from retrieved
# reference images.  The base FLUX DiT (Stream 1) is frozen.
#
# Resource budget  :  2 × GPU,  64 GB RAM,  16 CPUs
# Strategy         :  DDP across 2 GPUs
# =============================================================================

set -euo pipefail

# ---- Paths ---------------------------------------------------------------
PROJECT_DIR="/lustre/users/vmduc/Projects/fuzzy-engine"
NEXUS_DIR="${PROJECT_DIR}/Nexus-Gen"
DIFFSYNTH_DIR="${NEXUS_DIR}/DiffSynth-Studio"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"

cd "${PROJECT_DIR}"

echo "============================================================"
echo "  RAG Patch Training"
echo "  Node  : $(hostname)"
echo "  GPUs  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd', ')"
echo "  Date  : $(date)"
echo "  Python: ${VENV_PYTHON}"
echo "============================================================"

# ---- Verify Nexus-GenV2 weights exist ------------------------------------
NEXGEN_DIT="${NEXUS_DIR}/models/Nexus-GenV2/generation_decoder.bin"
if [[ ! -f "${NEXGEN_DIT}" ]]; then
    echo "ERROR: Nexus-GenV2 generation_decoder.bin not found at:"
    echo "       ${NEXGEN_DIT}"
    echo "Please ensure the Nexus-GenV2 model is downloaded."
    exit 1
fi
echo "Base DiT weights: ${NEXGEN_DIT} ($(du -h "${NEXGEN_DIT}" | cut -f1))"

# ---- Set PYTHONPATH so all imports resolve --------------------------------
export PYTHONPATH="${PROJECT_DIR}:${NEXUS_DIR}:${DIFFSYNTH_DIR}:${PYTHONPATH:-}"

# ---- Offline mode: prevent HuggingFace/Torch from hitting the internet ----
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCH_HOME="${HOME}/.cache/torch"

# ---- Prevent DeepSpeed from JIT-compiling CUDA extensions at import time --
# DS_BUILD_OPS=0 skips extension compilation entirely (ops are not needed for
# DDP training; they are only required when using DeepSpeed ZeRO strategies).
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1

# ---- Redirect Triton autotune cache off NFS (avoids hang on exit) ---------
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
mkdir -p "${TRITON_CACHE_DIR}"

# ---- Launch training (srun required for Lightning SLURM integration) -------
echo "Launching training..."
srun "${VENV_PYTHON}" rag_patch_training/train.py \
    --config rag_patch_training/config.yaml \
    "$@"

echo "Training complete."
