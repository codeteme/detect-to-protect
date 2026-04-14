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

CLIP_LEN=${CLIP_LEN:-32}
ANCHOR_OFFSET_SEC=${ANCHOR_OFFSET_SEC:-0.0}
RUN_NAME=${RUN_NAME:-baseline-clip${CLIP_LEN}-ofs${ANCHOR_OFFSET_SEC}}
OFFSET_TAG=${ANCHOR_OFFSET_SEC//./p}
OFFSET_TAG=${OFFSET_TAG//-/m}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/best_baseline_scratch_clip${CLIP_LEN}_ofs${OFFSET_TAG}.pt}
SUBMISSION_PATH=${SUBMISSION_PATH:-outputs/submission_${RUN_NAME}.csv}

echo "CLIP_LEN: ${CLIP_LEN}"
echo "ANCHOR_OFFSET_SEC: ${ANCHOR_OFFSET_SEC}"
echo "RUN_NAME: ${RUN_NAME}"
echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
echo "SUBMISSION_PATH: ${SUBMISSION_PATH}"

$PYTHON_BIN -u src/predict_baseline.py \
	--checkpoint-path "${CHECKPOINT_PATH}" \
	--submission-path "${SUBMISSION_PATH}"

echo "Job finished: $(date)"