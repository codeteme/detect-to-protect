#!/bin/bash
#SBATCH --job-name=dtp-train
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

source /hpc/group/coursess26/ids705/team-project/detect-to-protect/activate.sh

python src/train.py