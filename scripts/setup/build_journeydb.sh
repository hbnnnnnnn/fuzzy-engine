#!/bin/bash
#SBATCH --job-name=build-journeydb
#SBATCH --partition=002-partition-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/lustre/users/vmduc/Projects/fuzzy-engine/logs/rag_patch_training/journeydb_%j.out
#SBATCH --error=/lustre/users/vmduc/Projects/fuzzy-engine/logs/rag_patch_training/journeydb_%j.err

set -euo pipefail

source ~/sbai-env-script/set-proxy.sh

PROJECT_DIR="/lustre/users/vmduc/Projects/fuzzy-engine"
export JOURNEYDB_OUT="${PROJECT_DIR}/data/journeydb_dataset"
export SHARD_CACHE="${PROJECT_DIR}/data/journeydb_shards"
export HF_HUB_DISABLE_XET=1

mkdir -p "${JOURNEYDB_OUT}" "${SHARD_CACHE}"

echo "============================================================"
echo "  Building JourneyDB dataset"
echo "  Output : ${JOURNEYDB_OUT}"
echo "  Node   : $(hostname)"
echo "  Date   : $(date)"
echo "============================================================"

cd "${PROJECT_DIR}"
export PATH="${HOME}/.local/bin:${PATH}"

# Ensure uv venv exists with all base deps
uv sync --frozen

uv run python build_journeydb_dataset.py

echo ""
echo "============================================================"
echo "  JourneyDB build COMPLETE at $(date)"
ls -lh "${JOURNEYDB_OUT}/"
echo "============================================================"
