#!/bin/bash
#SBATCH --job-name=rag-patch
#SBATCH --partition=batch
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_rag_patch_%j.out
#SBATCH --error=slurm_rag_patch_%j.err

# =============================================================================
# RAG Patch Training — LoRA correction stream for FLUX DiT
# =============================================================================
# Trains a LoRA-injected FLUX DiT (Stream 2) that outputs an additive
# correction signal conditioned on VGG-19 style features from retrieved
# reference images.  The base FLUX DiT (Stream 1) is frozen.
#
# Resource budget  :  2 × GPU,  64 GB RAM,  16 CPUs
# Strategy         :  DeepSpeed Stage 2 (no param offload, 2-GPU sharding)
# =============================================================================

set -euo pipefail

module purge
source ~/miniconda3/bin/activate

# ---- Create / reuse conda environment ------------------------------------
ENV_NAME="nexus"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda env '${ENV_NAME}' (clone of nexus)..."
    conda create --name "${ENV_NAME}" --clone nexus -y
    conda activate "${ENV_NAME}"
    echo "Installing extra dependencies..."
    pip install peft lightning deepspeed torchvision --quiet
else
    echo "Conda env '${ENV_NAME}' already exists."
    conda activate "${ENV_NAME}"
fi

echo "============================================================"
echo "  RAG Patch Training"
echo "  Node  : $(hostname)"
echo "  GPUs  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd', ')"
echo "  Date  : $(date)"
echo "  Env   : ${CONDA_DEFAULT_ENV}"
echo "============================================================"

# ---- Paths ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NDBAO_DIR="/media02/nthuy/ndbao"
NEXUS_DIR="${NDBAO_DIR}/Nexus-Gen"
DIFFSYNTH_DIR="${NEXUS_DIR}/DiffSynth-Studio"

cd "${NDBAO_DIR}"

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
export PYTHONPATH="${NDBAO_DIR}:${NEXUS_DIR}:${DIFFSYNTH_DIR}:${PYTHONPATH:-}"

# ---- Redirect Triton autotune cache off NFS (avoids hang on exit) ---------
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
mkdir -p "${TRITON_CACHE_DIR}"

# ---- Launch training (srun required for Lightning SLURM integration) -------
echo "Launching training..."
srun python rag_patch_training/train.py \
    --config rag_patch_training/config.yaml \
    "$@"

echo "Training complete."
