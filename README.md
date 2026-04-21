# Detect to Protect

Video-based collision risk prediction using deep learning on dashcam footage.  
---

## Overview

Every year, tens of thousands of people are killed in vehicle collisions in the United States. Many crashes are preceded by detectable visual cues in the seconds before impact. This project builds and evaluates a pipeline that watches 1.6 seconds of dashcam video and outputs a collision probability, enabling faster emergency braking or pre-crash seat belt tensioning.

We fine-tuned [VideoMAE](https://huggingface.co/MCG-NJU/videomae-base) on the [Nexar Detect to Protect](https://www.kaggle.com/competitions/nexar-collision-prediction) dataset (1,500 labeled clips, balanced between collision and non-collision) and systematically ablated three input modalities — RGB frames, depth maps (DepthAnything v2), and segmentation masks (YOLO) — across two clip timing offsets.

**Best result:** three-stream VideoMAE (RGB + Depth + Seg) — validation AUC **0.918** (95% CI: 0.884–0.945).

---

## Results

| Model | Modalities | Offset (s) | Val AUC | 95% CI |
|---|---|---|---|---|
| TinyVideoCNN (scratch) | RGB | 0.0 | 0.679 | (0.618–0.735) |
| TinyVideoCNN (scratch) | RGB | 0.5 | 0.589 | (0.527–0.651) |
| TinyVideoCNN (scratch) | RGB, clip=64 | 0.0 | 0.633 | (0.571–0.693) |
| TinyVideoCNN (scratch) | RGB, clip=100 | 0.0 | 0.629 | (0.567–0.692) |
| VideoMAE fine-tuned | RGB | 0.0 | 0.769 | (0.712–0.818) |
| VideoMAE fine-tuned | RGB | 0.5 | 0.772 | (0.718–0.824) |
| VideoMAE fine-tuned | RGB + Depth | 0.0 | 0.814 | (0.766–0.861) |
| VideoMAE fine-tuned | RGB + Depth | 0.5 | 0.712 | (0.650–0.768) |
| VideoMAE fine-tuned | RGB + Seg | 0.0 | 0.682 | (0.625–0.740) |
| VideoMAE fine-tuned | RGB + Seg | 0.5 | 0.666 | (0.606–0.725) |
| VideoMAE fine-tuned | **RGB + Depth + Seg** | **0.0** | **0.918** | **(0.884–0.945)** |

95% CIs computed via bootstrap resampling (2,000 iterations) on the held-out 20% validation set.

---

## Repository Structure

```
detect-to-protect/
├── activate.sh                  # source this every DCC session
├── requirements.txt
├── docs/
│   ├── setup.md                 # DCC cluster setup guide
│   └── project-decisions.md    # design decisions log
├── notebooks/
│   ├── preprocess.ipynb         # frame extraction exploration
│   └── train.ipynb              # interactive training
├── scripts/                     # SLURM batch job scripts
│   ├── submit_train_baseline.sh
│   ├── submit_train_videomae.sh
│   ├── submit_train_videomae_depth.sh
│   ├── submit_train_videomae_seg.sh
│   ├── submit_train_videomae_full.sh
│   ├── submit_predict_baseline.sh
│   ├── submit_predict_videomae.sh
│   ├── submit_predict_videomae_depth.sh
│   ├── submit_predict_videomae_seg.sh
│   └── submit_predict_videomae_full.sh
├── src/
│   ├── train_baseline.py        # TinyVideoCNN trained from scratch
│   ├── train_videomae.py        # VideoMAE RGB fine-tuning
│   ├── train_videomae_depth.py  # two-stream RGB + Depth
│   ├── train_videomae_seg.py    # two-stream RGB + Seg
│   ├── train_videomae_full.py   # three-stream RGB + Depth + Seg (best)
│   ├── predict_baseline.py      # Kaggle submission — baseline
│   ├── predict_videomae.py      # Kaggle submission — RGB
│   ├── predict_videomae_depth.py
│   ├── predict_videomae_seg.py
│   ├── predict_videomae_full.py
│   ├── eval_save_preds.py       # save val predictions for bootstrap CI
│   └── visualize_pipeline.py   # generate pipeline figure
└── data/                        # not tracked in git — see Data section
    ├── train.csv
    ├── test.csv
    ├── frames/                  # RGB frames at 10 fps (.npy)
    ├── depth/                   # DepthAnything v2 depth maps (.npy)
    └── segmentation/            # YOLO segmentation masks (.npy)
```

---

## Setup

All training was run on the Duke Computing Cluster (DCC). See [`docs/setup.md`](docs/setup.md) for the full environment setup guide.

**Quick start (each session):**

```bash
ssh <netid>@dcc-login.oit.duke.edu
source /hpc/group/coursess26/ids705/team-project/detect-to-protect/activate.sh
```

---

## Data

Download from the [Kaggle competition page](https://www.kaggle.com/competitions/nexar-collision-prediction). Place files under `data/` matching the structure above. Pre-extracted `.npy` arrays for frames, depth, and segmentation are stored on the DCC shared filesystem and are not committed to this repository.

---

## Training

Submit any model as a SLURM batch job from the project root:

```bash
# Best model — three-stream RGB + Depth + Seg
sbatch scripts/submit_train_videomae_full.sh

# RGB + Depth two-stream
ANCHOR_OFFSET_SEC=0.5 sbatch scripts/submit_train_videomae_depth.sh

# Baseline from scratch
sbatch scripts/submit_train_baseline.sh
```

All runs log to [Weights & Biases](https://wandb.ai/teme/detect-to-protect).

---

## Evaluation

To recompute validation predictions and bootstrap confidence intervals for a saved checkpoint:

```bash
PYTHON=envs/dtp/bin/python

# Example: three-stream best model
$PYTHON src/eval_save_preds.py \
    --type full \
    --checkpoint outputs/best_videomae_full_ofs0p0.pt \
    --out outputs/preds_videomae_full_ofs0p0.npz
```

`--type` choices: `rgb`, `depth`, `seg`, `full`, `baseline`

---

## Key Findings

- **Pretraining matters.** VideoMAE fine-tuned on RGB (AUC 0.769) substantially outperformed a 3D CNN trained from scratch (AUC 0.679) on the same data.
- **All three modalities together are best.** The three-stream model (RGB + Depth + Seg) with freeze-then-finetune training reached AUC 0.918, the highest across all configurations.
- **Segmentation alone doesn't help, but combined with depth it does.** RGB+Seg scored 0.682 (below the RGB baseline), but RGB+Depth+Seg scored 0.918, suggesting depth and segmentation carry complementary information the model can exploit when fused together.
- **Only the final 1–2 seconds before impact matter.** Longer clips consistently hurt TinyVideoCNN performance; the most predictive signal is concentrated immediately before the collision event.
