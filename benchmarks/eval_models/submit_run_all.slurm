#!/bin/bash
#SBATCH --job-name=eval-run-all
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_eval_run_all_%j.out
#SBATCH --error=slurm_eval_run_all_%j.err

set -euo pipefail

# Submit example:
# sbatch submit_run_all.slurm
# sbatch submit_run_all.slurm --gen-only --model sdxl

REPO_ROOT="/media02/nthuy/ndbao/ndbao_hbngoc"
SCRIPT="${REPO_ROOT}/eval_models/run_all.sh"

# Override if your conda base differs.
export CONDA_BASE="${CONDA_BASE:-/media02/nthuy/miniconda3}"

cd "${REPO_ROOT}/eval_models"
bash "${SCRIPT}" "$@"
