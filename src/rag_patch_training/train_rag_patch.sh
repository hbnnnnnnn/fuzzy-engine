#!/bin/bash
#SBATCH --job-name=rag-patch
#SBATCH --partition=002-partition-default
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=/lustre/users/vmduc/Projects/fuzzy-engine/logs/rag_patch_training/train_%j.out
#SBATCH --error=/lustre/users/vmduc/Projects/fuzzy-engine/logs/rag_patch_training/train_%j.err

# =============================================================================
# RAG Patch Training — LoRA correction stream for FLUX DiT
# =============================================================================
# Trains a LoRA-injected FLUX DiT (Stream 2) that outputs an additive
# correction signal conditioned on VGG-19 style features from retrieved
# reference images.  The base FLUX DiT (Stream 1) is frozen.
#
# Resource budget  :  2 × GPU,  64 GB RAM,  16 CPUs
# Strategy         :  DeepSpeed Stage 2 (no param offload, 2-GPU sharding)
# Container        :  /lustre/users/vmduc/container_cache/image_generation_pipeline:latest.sqsh
# Environment      :  uv (pyproject.toml at project root)
# =============================================================================

set -euo pipefail

# ---- Paths ---------------------------------------------------------------
PROJECT_DIR="/lustre/users/vmduc/Projects/fuzzy-engine"
SRC_DIR="${PROJECT_DIR}/src"
NEXUS_DIR="${SRC_DIR}/Nexus-Gen"
DIFFSYNTH_DIR="${NEXUS_DIR}/DiffSynth-Studio"
CONTAINER_IMAGE="/lustre/users/vmduc/container_cache/image_generation_pipeline:latest.sqsh"

# ---- Ensure log directory exists -----------------------------------------
mkdir -p "${PROJECT_DIR}/logs/rag_patch_training"

cd "${PROJECT_DIR}"

# ---- Verify Nexus-GenV2 weights exist ------------------------------------
NEXGEN_DIT="${NEXUS_DIR}/models/Nexus-GenV2/generation_decoder.bin"
if [[ ! -f "${NEXGEN_DIT}" ]]; then
    echo "ERROR: Nexus-GenV2 generation_decoder.bin not found at:"
    echo "       ${NEXGEN_DIT}"
    echo "Please ensure the Nexus-GenV2 model is downloaded."
    exit 1
fi
echo "Base DiT weights: ${NEXGEN_DIT} ($(du -h "${NEXGEN_DIT}" | cut -f1))"

echo "============================================================"
echo "  RAG Patch Training"
echo "  Node      : $(hostname)"
echo "  Date      : $(date)"
echo "  Project   : ${PROJECT_DIR}"
echo "  Container : ${CONTAINER_IMAGE}"
echo "============================================================"

# ---- Environment variables passed into the container ---------------------
# PYTHONPATH: SRC_DIR for rag_patch_training; NEXUS_DIR + DIFFSYNTH_DIR for
#             in-tree Nexus-Gen + DiffSynth modules.
export PYTHONPATH="${SRC_DIR}:${NEXUS_DIR}:${DIFFSYNTH_DIR}:${PYTHONPATH:-}"

# Redirect Triton autotune cache off NFS (avoids hang on exit)
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"

# Offline mode: compute nodes have no outbound network
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCH_HOME="${HOME}/.cache/torch"

# Skip DeepSpeed JIT CUDA op compilation (ZeRO-2 works without it)
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ---- Launch training inside container via srun + uv ----------------------
echo "Launching training..."
srun -l -K1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-remap-root \
    --container-mounts /lustre:/lustre,/home:/home \
    bash -c "
        set -euo pipefail
        mkdir -p '${TRITON_CACHE_DIR}'
        cd '${PROJECT_DIR}'
        export PATH=\"\${HOME}/.local/bin:\${PATH}\"
        uv venv --system-site-packages .venv --quiet
        uv sync --extra train --frozen --quiet
        uv run python src/rag_patch_training/train.py \
            --config src/rag_patch_training/config.yaml \
            $@
    "

echo "Training complete."
