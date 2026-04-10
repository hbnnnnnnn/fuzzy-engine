#!/bin/bash
#SBATCH --job-name=eval_setup_models
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Submit example:
# sbatch submit_setup_all_models.slurm
# sbatch submit_setup_all_models.slurm --export=ALL,INSTALL_DIR=/path/to/ndbao_hbngoc

REPO_ROOT="/media02/nthuy/ndbao/ndbao_hbngoc"
SCRIPT="${REPO_ROOT}/eval_models/setup_all_models.sh"

# Override if your conda base differs.
export CONDA_BASE="${CONDA_BASE:-/media02/nthuy/miniconda3}"

cd "${REPO_ROOT}/eval_models"
bash "${SCRIPT}" "$@"
