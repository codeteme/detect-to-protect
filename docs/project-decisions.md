# Detect-to-Protect Project Decisions

## Problem

Predict collision risk from video clips. Each sample is a video with frame data, depth, and segmentation data.

## Goal

Build a clear baseline first, then test small changes one at a time. Keep the pipeline simple and easy to compare.

## Data Types

- RGB frames
- Depth maps
- Segmentation masks
- Binary target label: 0 or 1

## Model Types Being Built

- TinyVideoCNN baseline from scratch
- VideoMAE fine-tuning next

## Preprocessing Decision

The `.npy` files were created before training by preprocessing the raw video data and saving the frame sequences to disk. Each file stores one video as an array of RGB frames, with matching depth and segmentation arrays saved alongside it. The training script then reads these `.npy` files from the shared DCC folder. For each video, it loads the RGB frames, depth, and segmentation arrays. The clip is taken as a fixed window near the event or from the end of the video if there is no event. In this project, the videos are used at `10 fps`, and the baseline keeps a short clip length of `32` frames by default. The model input is a tensor shaped like `[T, C, H, W]` for one clip, then stacked into batches during training.

Simple flow:

```text
video files -> fixed clip window -> tensor -> TinyVideoCNN -> binary score
```

This flow is for the baseline only.

## First Baseline Model

The first model is a small 3D CNN trained from scratch. It sees the whole clip at once, not frame by frame.

```text
32 frames @ 10 fps
      |
      v
[T, C, H, W]
      |
      v
TinyVideoCNN
      |
      v
score: probability between 0 and 1
```

## Baseline Result

- Best validation AUC: `0.6787`
- Best epoch: `8`
- W&B run: `baseline-scratch`

## Ablation Plan

Test one change at a time.

Train and prediction can be queued together using Slurm dependency:

```bash
# Example: clip64 / offset0.0
jid=$(CLIP_LEN=64 ANCHOR_OFFSET_SEC=0.0 RUN_NAME=baseline-clip64-ofs0.0 sbatch scripts/submit_train_baseline.sh | awk '{print $4}')
CLIP_LEN=64 ANCHOR_OFFSET_SEC=0.0 RUN_NAME=baseline-clip64-ofs0.0 sbatch --dependency=afterok:${jid} scripts/submit_predict_baseline.sh
```

### Clip-length ablations

```bash
CLIP_LEN=64  ANCHOR_OFFSET_SEC=0.0 RUN_NAME=baseline-clip64-ofs0.0  sbatch scripts/submit_train_baseline.sh
CLIP_LEN=100 ANCHOR_OFFSET_SEC=0.0 RUN_NAME=baseline-clip100-ofs0.0 sbatch scripts/submit_train_baseline.sh
```

Why this test:

- Tests how much temporal context helps prediction quality.
- Checks whether longer clips add signal or mostly add noise.

### Event-anchor ablation

```bash
CLIP_LEN=32  ANCHOR_OFFSET_SEC=0.5 RUN_NAME=baseline-clip32-ofs0.5  sbatch scripts/submit_train_baseline.sh
```

Why this test:

- Tests if the model can predict from pre-collision cues.
- Reduces chance of relying on impact-moment frames only.

## W&B Notes

- Use W&B for run tracking, metrics, and config comparison.
- Log clip length, anchor offset, loss, AUC, and best checkpoint.
- Main project page: `https://wandb.ai/teme/detect-to-protect`

## Experiment Tracking Table

| run_name | clip_len | anchor_offset_sec | best_val_auc | checkpoint | submission_file | kaggle_score | notes |
|---|---:|---:|---:|---|---|---:|---|
| baseline-scratch | 32 | 0.0 | 0.6787 | outputs/best_baseline_scratch_clip32_ofs0p0.pt | outputs/submission_baseline-scratch.csv |  | first baseline |
| baseline-clip64-ofs0.0 | 64 | 0.0 | 0.6331 | outputs/best_baseline_scratch_clip64_ofs0p0.pt | outputs/submission_baseline-clip64-ofs0.0.csv |  | lower than baseline |
| baseline-clip100-ofs0.0 | 100 | 0.0 | 0.6289 | outputs/best_baseline_scratch_clip100_ofs0p0.pt | outputs/submission_baseline-clip100-ofs0.0.csv |  | lower than baseline |
| baseline-clip32-ofs0.5 | 32 | 0.5 | 0.5894 | outputs/best_baseline_scratch_clip32_ofs0p5.pt | outputs/submission_baseline-clip32-ofs0.5.csv |  | lower than baseline |

## VideoMAE Ablation Plan

Fine-tuning MCG-NJU/videomae-base with binary classification head. Clip length is fixed at 16 frames — the model was pretrained with fixed position embeddings for 16 frames and cannot handle other lengths without retraining the embeddings.

Each modality combination is tested at both anchor offsets (0.0 and 0.5).

### Modality ablations

| stage | modalities | script |
|---|---|---|
| 1 | rgb | train_videomae.py |
| 2 | rgb + depth | train_videomae_depth.py |
| 3 | rgb + seg | train_videomae_seg.py |
| 4 | rgb + depth + seg | train_videomae_full.py |

### Anchor offset

Both 0.0 and 0.5 tested for each modality combination.

## VideoMAE Experiment Tracking Table

| run_name | modalities | anchor_offset_sec | best_val_auc | checkpoint | submission_file | kaggle_score | notes |
|---|---|---:|---:|---|---|---:|---|
| videomae-clip16-ofs0.0 | rgb | 0.0 | 0.7690 | outputs/best_videomae_clip16_ofs0p0.pt | outputs/submission_videomae-clip16-ofs0.0.csv | | rgb baseline |
| videomae-clip16-ofs0.5 | rgb | 0.5 | 0.7724 | outputs/best_videomae_clip16_ofs0p5.pt | outputs/submission_videomae-clip16-ofs0.5.csv | | rgb anchor ablation |
| videomae-depth-ofs0.0 | rgb+depth | 0.0 | | outputs/best_videomae_depth_ofs0p0.pt | outputs/submission_videomae-depth-ofs0.0.csv | | |
| videomae-depth-ofs0.5 | rgb+depth | 0.5 | | outputs/best_videomae_depth_ofs0p5.pt | outputs/submission_videomae-depth-ofs0.5.csv | | |
| videomae-seg-ofs0.0 | rgb+seg | 0.0 | 0.6823 | outputs/best_videomae_seg_ofs0p0.pt | outputs/submission_videomae-seg-ofs0.0.csv | | lower than rgb only |
| videomae-seg-ofs0.5 | rgb+seg | 0.5 | 0.6657 | outputs/best_videomae_seg_ofs0p5.pt | outputs/submission_videomae-seg-ofs0.5.csv | | lower than rgb only |
| videomae-full-ofs0.0 | rgb+depth+seg | 0.0 | | outputs/best_videomae_full_ofs0p0.pt | outputs/submission_videomae-full-ofs0.0.csv | | running |
| videomae-full-ofs0.5 | rgb+depth+seg | 0.5 | | outputs/best_videomae_full_ofs0p5.pt | outputs/submission_videomae-full-ofs0.5.csv | | running |
| videomae-full-ofs0.5 | rgb+depth+seg | 0.0 | | outputs/best_videomae_full_ofs0p5.pt | outputs/submission_videomae-full-ofs0.5.csv | | running | variable ..