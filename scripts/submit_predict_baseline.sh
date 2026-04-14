#!/bin/bash
#SBATCH --job-name=dtp-baseline-predict
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/predict_baseline_%j.out
#SBATCH --error=logs/predict_baseline_%j.err

set -euo pipefail

source /hpc/group/coursess26/ids705/team-project/detect-to-protect/activate.sh

cd /hpc/group/coursess26/ids705/team-project/detect-to-protect
mkdir -p logs outputs

PYTHON_BIN=/hpc/group/coursess26/ids705/team-project/detect-to-protect/envs/dtp/bin/python

echo "Job started: $(date)"
echo "Node: $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
	echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
else
	echo "GPU: nvidia-smi not available"
fi
echo "Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "CUDA: $($PYTHON_BIN -c 'import torch; print(torch.cuda.is_available())')"

if [[ -z "${CLIP_LEN:-}" ]]; then
  CLIP_LEN=32
fi
echo "CLIP_LEN: ${CLIP_LEN}"

$PYTHON_BIN -u src/predict_baseline.py

echo "Job finished: $(date)"