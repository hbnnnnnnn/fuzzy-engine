#!/bin/bash
#SBATCH --job-name=jdb-build
#SBATCH --partition=batch
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_journeydb_%j.out
#SBATCH --error=slurm_journeydb_%j.err

# =============================================================================
# JourneyDB Dataset Preparation
# Streams JourneyDB/JourneyDB (train split) from HuggingFace, decodes images,
# saves as JPEG (VGG-19 compatible), and writes metadata.jsonl + train.csv.
# Stops after collecting exactly 100k valid (image, prompt) pairs.
# =============================================================================

module purge
source ~/miniconda3/bin/activate
conda activate nexus

echo "============================================================"
echo "  JourneyDB Dataset Build"
echo "  Node:   $(hostname)"
echo "  Date:   $(date)"
echo "============================================================"

cd ~/ndbao

# Output directory — override with env var if needed
export JOURNEYDB_OUT="${JOURNEYDB_OUT:-./journeydb_dataset}"

# HuggingFace cache (reuse existing cache)
export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"

# HuggingFace token is read automatically from ~/.cache/huggingface/token
# If your token is elsewhere, uncomment and set:
# export HF_TOKEN="hf_..."

echo ""
echo "Output directory : ${JOURNEYDB_OUT}"
echo "HF datasets cache: ${HF_DATASETS_CACHE}"
echo ""
echo "Starting dataset preparation..."
echo ""

python3 -u build_journeydb_dataset.py

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Build COMPLETE — exit code 0"
    echo "  Output dir: ${JOURNEYDB_OUT}"
    echo ""
    echo "  File listing:"
    ls -lh "${JOURNEYDB_OUT}/" 2>/dev/null
    echo ""
    echo "  Sample count check:"
    wc -l "${JOURNEYDB_OUT}/metadata.jsonl" 2>/dev/null
elif [ $EXIT_CODE -eq 2 ]; then
    echo "  Build INCOMPLETE — fewer than 100k samples collected"
    echo "  Check logs above for details (gated dataset / network issues?)"
else
    echo "  Build FAILED — exit code $EXIT_CODE"
fi
echo "  Date: $(date)"
echo "============================================================"

exit $EXIT_CODE
