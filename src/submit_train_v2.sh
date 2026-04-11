#!/bin/bash
#SBATCH --job-name=dtp-vmae-v2
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_v2_%j.out
#SBATCH --error=logs/train_v2_%j.err

source /hpc/group/coursess26/ids705/team-project/detect-to-protect/activate.sh

cd /hpc/group/coursess26/ids705/team-project/detect-to-protect

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python src/train_v2.py

echo "Job finished: $(date)"
