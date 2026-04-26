#!/bin/bash
#SBATCH --job-name=test_nexus
#SBATCH --output=test_nexus_%j.out
#SBATCH --error=test_nexus_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch

module purge
source /media02/nthuy/miniconda3/bin/activate

conda activate nexus

cd ~/ndbao/Nexus-Gen

python image_generation.py --prompt "A cute cat" --width 512 --height 512