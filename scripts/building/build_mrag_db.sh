#!/bin/bash
#SBATCH --job-name=mrag-db-build
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_mrag_db_%j.out
#SBATCH --error=slurm_mrag_db_%j.err

# =============================================================================
# MRAG Database Build — Qwen2.5-VL captioning + SigLIP embedding + FAISS index
# =============================================================================
# Loads images from laion/laion2B-en-aesthetic, generates fresh captions with
# Qwen2.5-VL-7B-Instruct (≤50 words), embeds both images and captions with
# SigLIP, then stores them in FAISS indexes (image.faiss + text.faiss).
# =============================================================================

module purge
source ~/miniconda3/bin/activate

# Clone nexus env and upgrade transformers (only once — skipped if already exists)
if ! conda env list | grep -q "^nexus_mrag "; then
    echo "Creating nexus_mrag env (clone of nexus)..."
    conda create --name nexus_mrag --clone nexus -y
    echo "Upgrading transformers in nexus_mrag..."
    conda run -n nexus_mrag pip install "transformers" -U
    echo "nexus_mrag env ready."
else
    echo "nexus_mrag env already exists, skipping clone."
fi

conda activate nexus_mrag

echo "============================================================"
echo "  MRAG DB Build"
echo "  Node:   $(hostname)"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Date:   $(date)"
echo "============================================================"

cd ~/ndbao/Nexus-Gen

export CUDA_LAUNCH_BLOCKING=1
# Allow HuggingFace datasets to use more connections for streaming
export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"
export MRAG_OUTPUT_DIR="${MRAG_OUTPUT_DIR:-mrag-db}"

# HuggingFace authentication — token is read from ~/.cache/huggingface/token automatically.
# Optionally override with HF_TOKEN env var.

echo ""
echo "Starting DB build..."
echo ""

python3 -u mrag_db_build.py

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Build COMPLETE — exit code 0"
    echo "  Output dir: ~/ndbao/Nexus-Gen/${MRAG_OUTPUT_DIR}/"
    ls -lh ~/ndbao/Nexus-Gen/${MRAG_OUTPUT_DIR}/ 2>/dev/null
else
    echo "  Build FAILED — exit code $EXIT_CODE"
fi
echo "  Date: $(date)"
echo "============================================================"

exit $EXIT_CODE
