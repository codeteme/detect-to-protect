#!/bin/bash
#SBATCH --job-name=dtp-videomae
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

source /hpc/group/coursess26/ids705/team-project/detect-to-protect/activate.sh

cd /hpc/group/coursess26/ids705/team-project/detect-to-protect

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python src/train.py

echo "Job finished: $(date)"
