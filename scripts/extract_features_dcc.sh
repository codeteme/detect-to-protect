#!/bin/bash
# =============================================================================
# extract_features_dcc.sh
#
# SLURM array job for feature extraction on the Duke Compute Cluster (DCC).
# Runs one job per video in parallel — each job processes a single video
# through the full pipeline: frame extraction → detection → depth → features.
#
# Submit with:
#   sbatch scripts/extract_features_dcc.sh
#
# Monitor with:
#   squeue -u $USER
#   sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS
# =============================================================================

#SBATCH --job-name=d2p-features
#SBATCH --output=logs/slurm/%A_%a.out     # stdout: logs/slurm/<job_id>_<array_idx>.out
#SBATCH --error=logs/slurm/%A_%a.err      # stderr: logs/slurm/<job_id>_<array_idx>.err
#SBATCH --partition=gpu-common            # DCC GPU partition
#SBATCH --gres=gpu:1                      # 1 GPU per job (for YOLO + Depth Anything V2)
#SBATCH --mem=16G                         # RAM per job
#SBATCH --cpus-per-task=4                 # CPU cores per job
#SBATCH --time=01:00:00                   # Max 1 hour per video (adjust if needed)
#SBATCH --array=0-1499                    # One job per video (1500 videos total)

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Load conda and activate your project environment
module load Miniconda3
conda activate detect-to-protect

# Move to project root (adjust this path to where you cloned the repo on DCC)
cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Map array index → video ID
# ---------------------------------------------------------------------------
# train.csv lists all video IDs. We use the array index to pick one video.
# awk skips the header (NR>1) and selects the row matching this array task.

VIDEO_ID=$(awk -F',' -v idx=$SLURM_ARRAY_TASK_ID 'NR>1 { if (NR-2 == idx) print $1 }' data/train.csv)

if [ -z "$VIDEO_ID" ]; then
    echo "ERROR: No video ID found for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Array task : $SLURM_ARRAY_TASK_ID"
echo "Video ID   : $VIDEO_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $CUDA_VISIBLE_DEVICES"

# ---------------------------------------------------------------------------
# Step 1: Extract frames from raw video
# ---------------------------------------------------------------------------
# Skip if frames already exist (safe to re-run)

FRAMES_DIR="data/frames/${VIDEO_ID}"

if [ -d "$FRAMES_DIR" ] && [ "$(ls -A $FRAMES_DIR)" ]; then
    echo "Frames already exist for $VIDEO_ID — skipping extraction"
else
    echo "Extracting frames for $VIDEO_ID..."
    python src/pipeline/extract_frames.py --video_path data/raw/${VIDEO_ID}.mp4 --output_dir $FRAMES_DIR
fi

# ---------------------------------------------------------------------------
# Step 2: Extract features (detection + depth + temporal)
# ---------------------------------------------------------------------------
# Skip if output already exists (safe to re-run after partial failures)

FEATURES_OUT="data/features/${VIDEO_ID}.parquet"

if [ -f "$FEATURES_OUT" ]; then
    echo "Features already exist for $VIDEO_ID — skipping"
else
    echo "Extracting features for $VIDEO_ID..."
    python src/pipeline/feature_extractor.py \
        --video_dir $FRAMES_DIR \
        --output_dir data/features
fi

echo "Done: $VIDEO_ID"
