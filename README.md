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
| TinyVideoCNN (scratch) | RGB, clip=16 | 0.0 | 0.709 | (0.648–0.765) |
| TinyVideoCNN (scratch) | RGB, clip=16 | 0.5 | 0.645 | (0.584–0.703) |
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
| VideoMAE fine-tuned | RGB + Depth + Seg | 0.5 | — | — |
| VideoMAE fine-tuned | RGB + Depth + Seg | 0.8 | 0.801 | (0.749–0.850) |
| VideoMAE fine-tuned | RGB + Depth + Seg | 1.0 | 0.771 | (0.716–0.823) |

95% CIs computed via bootstrap resampling (2,000 iterations) on the held-out 20% validation set.

### Classification Metrics (F1-Optimal Threshold)

Precision, recall, and F1 computed at the threshold that maximises F1 for each model. Run `python src/compute_metrics.py` to reproduce.

| Model | Modalities | Offset (s) | AUC | 95% CI | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| TinyVideoCNN (scratch) | RGB, clip=32 | 0.0 | 0.679 | (0.618–0.735) | 0.559 | 0.953 | 0.704 |
| TinyVideoCNN (scratch) | RGB, clip=32 | 0.5 | 0.589 | (0.527–0.651) | 0.538 | 0.940 | 0.684 |
| TinyVideoCNN (scratch) | RGB, clip=64 | 0.0 | 0.633 | (0.571–0.693) | 0.574 | 0.853 | 0.686 |
| TinyVideoCNN (scratch) | RGB, clip=100 | 0.0 | 0.629 | (0.567–0.692) | 0.503 | 1.000 | 0.670 |
| TinyVideoCNN (scratch) | RGB, clip=16 | 0.0 | 0.709 | (0.648–0.765) | 0.619 | 0.887 | 0.729 |
| TinyVideoCNN (scratch) | RGB, clip=16 | 0.5 | 0.645 | (0.584–0.703) | 0.557 | 0.913 | 0.692 |
| VideoMAE fine-tuned | RGB | 0.0 | 0.769 | (0.712–0.818) | 0.626 | 0.860 | 0.725 |
| VideoMAE fine-tuned | RGB | 0.5 | 0.772 | (0.718–0.824) | 0.689 | 0.767 | 0.726 |
| VideoMAE fine-tuned | RGB + Depth | 0.0 | 0.814 | (0.765–0.861) | 0.733 | 0.787 | 0.759 |
| VideoMAE fine-tuned | RGB + Depth | 0.5 | 0.712 | (0.650–0.768) | 0.608 | 0.860 | 0.713 |
| VideoMAE fine-tuned | RGB + Seg | 0.0 | 0.682 | (0.625–0.740) | 0.627 | 0.773 | 0.693 |
| VideoMAE fine-tuned | RGB + Seg | 0.5 | 0.666 | (0.605–0.725) | 0.537 | 0.973 | 0.692 |
| **VideoMAE fine-tuned** | **RGB + Depth + Seg** | **0.0** | **0.918** | **(0.884–0.945)** | **0.781** | **0.927** | **0.848** |
| VideoMAE fine-tuned | RGB + Depth + Seg | 0.5 | — | — | — | — | — |
| VideoMAE fine-tuned | RGB + Depth + Seg | 0.8 | 0.801 | (0.749–0.850) | 0.685 | 0.900 | 0.778 |
| VideoMAE fine-tuned | RGB + Depth + Seg | 1.0 | 0.771 | (0.716–0.823) | 0.667 | 0.840 | 0.743 |

### Confusion Matrices (F1-Optimal Threshold)

Out of 150 collision clips and 150 non-collision clips in the validation set.

| Model | Offset (s) | Collisions Caught | Collisions Missed | False Alarms | Correct Negatives | Recall | False Alarm Rate |
|---|---|---|---|---|---|---|---|
| TinyVideoCNN, clip=32 | 0.0 | 143 | 7 | 113 | 37 | 95.3% | 75.3% |
| TinyVideoCNN, clip=32 | 0.5 | 141 | 9 | 121 | 29 | 94.0% | 80.7% |
| TinyVideoCNN, clip=64 | 0.0 | 128 | 22 | 95 | 55 | 85.3% | 63.3% |
| TinyVideoCNN, clip=100 | 0.0 | 150 | 0 | 148 | 2 | 100.0% | 98.7% |
| VideoMAE RGB | 0.0 | 129 | 21 | 77 | 73 | 86.0% | 51.3% |
| VideoMAE RGB | 0.5 | 115 | 35 | 52 | 98 | 76.7% | 34.7% |
| VideoMAE RGB + Depth | 0.0 | 118 | 32 | 43 | 107 | 78.7% | 28.7% |
| VideoMAE RGB + Depth | 0.5 | 129 | 21 | 83 | 67 | 86.0% | 55.3% |
| VideoMAE RGB + Seg | 0.0 | 116 | 34 | 69 | 81 | 77.3% | 46.0% |
| VideoMAE RGB + Seg | 0.5 | 146 | 4 | 126 | 24 | 97.3% | 84.0% |
| **VideoMAE Full** | **0.0** | **139** | **11** | **39** | **111** | **92.7%** | **26.0%** |
| VideoMAE Full | 0.5 | — | — | — | — | — | — |
| VideoMAE Full | 0.8 | 135 | 15 | 62 | 88 | 90.0% | 41.3% |
| VideoMAE Full | 1.0 | 126 | 24 | 63 | 87 | 84.0% | 42.0% |
| TinyVideoCNN, clip=16 | 0.0 | 133 | 17 | 82 | 68 | 88.7% | 54.7% |
| TinyVideoCNN, clip=16 | 0.5 | 137 | 13 | 109 | 41 | 91.3% | 72.7% |

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
│   ├── compute_metrics.py       # precision, recall, F1, confusion matrix from .npz files
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
- **All three modalities together are best.** The three-stream model (RGB + Depth + Seg) with freeze-then-finetune training reached AUC 0.918, the highest across all configurations. At its optimal threshold it catches 139/150 collisions (92.7% recall) with a 26% false alarm rate.
- **Segmentation alone doesn't help, but combined with depth it does.** RGB+Seg scored 0.682 (below the RGB baseline), but RGB+Depth+Seg scored 0.918, suggesting depth and segmentation carry complementary information the model can exploit when fused together.
- **The final moments before impact are the most predictive.** For the best model, shifting the clip window back by 0.8s drops AUC from 0.918 to 0.801; shifting back 1.0s drops it further to 0.771. Recall falls from 92.7% to 84.0% and false alarms nearly double. Earlier footage adds noise rather than signal. This is reinforced by the baseline clip-length ablation: clip=16 (AUC 0.709) outperforms clip=32 (0.679), clip=64 (0.633), and clip=100 (0.629) — shorter windows consistently perform better.
- **Depth is the most time-sensitive modality.** Shifting the clip back 0.5s hurts the depth model sharply (AUC 0.814 → 0.712) but barely affects the RGB-only model (0.769 → 0.772), confirming that proximity cues change most rapidly in the final half-second before a collision.
- **Night is a precision problem, not a recall problem.** The model achieves 100% recall on dark clips (0 missed collisions at night) but over-triggers on dark non-collision scenes — 5 of the 10 worst false positives are dark clips with near-certain confidence (scores ≥ 0.990). All 11 missed collisions are bright daytime clips. This is the inverse of object-detection-based systems (e.g. V-CAS), which fail at night because bounding-box pipelines break in the dark. See [`docs/lighting-analysis.md`](docs/lighting-analysis.md) for the full analysis.

---

## Lighting Analysis

A brightness-based proxy analysis was run on the best model's validation predictions to assess day vs. night performance. Clips were classified as dark (mean last-frame brightness < 60) or bright otherwise.

| Condition | n | AUC | Recall | False Alarm Rate |
|---|---|---|---|---|
| All clips | 300 | 0.918 | 92.7% | 26.0% |
| Bright (day proxy) | 216 | 0.915 | 89.7% | 26.6% |
| Dark (night proxy) | 84 | 0.922 | **100.0%** | 24.4% |

The model misses zero collisions at night but fires with near-certainty on dark non-collision clips, suggesting a spurious correlation between visual darkness and collision score. All 11 missed collisions are visually bright daytime events where the pre-collision signal is subtle.

See [`docs/lighting-analysis.md`](docs/lighting-analysis.md) for methodology, worst-case clip analysis, comparison with V-CAS, and implications for future work.
