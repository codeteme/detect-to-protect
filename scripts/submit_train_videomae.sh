#!/bin/bash
#SBATCH --job-name=dtp-videomae-train
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_videomae_%j.out
#SBATCH --error=logs/train_videomae_%j.err

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

# # Compute nodes have no internet — run W&B and HF offline
# export WANDB_MODE=offline
# export WANDB_API_KEY=${WANDB_API_KEY:-$(grep -A2 'wandb.ai' ~/.netrc | grep password | awk '{print $2}')}
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

CLIP_LEN=${CLIP_LEN:-16}
ANCHOR_OFFSET_SEC=${ANCHOR_OFFSET_SEC:-0.0}
RUN_NAME=${RUN_NAME:-videomae-clip${CLIP_LEN}-ofs${ANCHOR_OFFSET_SEC}}
echo "CLIP_LEN: ${CLIP_LEN}"
echo "ANCHOR_OFFSET_SEC: ${ANCHOR_OFFSET_SEC}"
echo "RUN_NAME: ${RUN_NAME}"

$PYTHON_BIN -u src/train_videomae.py \
	--clip-len "${CLIP_LEN}" \
	--anchor-offset-sec "${ANCHOR_OFFSET_SEC}" \
	--run-name "${RUN_NAME}"

echo "Job finished: $(date)"

# Sync offline W&B runs to cloud from the login node after job completes
echo "Syncing W&B offline runs..."
wandb sync outputs/wandb/offline-run-* \
	&& echo "W&B sync done." \
	|| echo "W&B sync failed — run manually: wandb sync outputs/wandb/offline-run-*"