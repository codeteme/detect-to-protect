#!/bin/bash
#SBATCH --job-name=dtp-baseline-train
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_baseline_%j.out
#SBATCH --error=logs/train_baseline_%j.err

set -euo pipefail

source /hpc/group/coursess26/ids705/team-project/detect-to-protect/activate.sh

cd /hpc/group/coursess26/ids705/team-project/detect-to-protect
mkdir -p logs outputs

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python src/train_baseline.py

echo "Job finished: $(date)"
