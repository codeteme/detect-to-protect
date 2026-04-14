#!/bin/bash
#SBATCH --job-name=dtp-videomae-depth-train
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_videomae_depth_%j.out
#SBATCH --error=logs/train_videomae_depth_%j.err

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

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ANCHOR_OFFSET_SEC=${ANCHOR_OFFSET_SEC:-0.0}
RUN_NAME=${RUN_NAME:-videomae-depth-ofs${ANCHOR_OFFSET_SEC}}

echo "ANCHOR_OFFSET_SEC: ${ANCHOR_OFFSET_SEC}"
echo "RUN_NAME: ${RUN_NAME}"

$PYTHON_BIN -u src/train_videomae_depth.py \
	--anchor-offset-sec "${ANCHOR_OFFSET_SEC}" \
	--run-name "${RUN_NAME}"

echo "Job finished: $(date)"