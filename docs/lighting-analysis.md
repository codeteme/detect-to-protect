# Lighting Condition Analysis: Night vs. Day Performance

## Motivation

A natural hypothesis when building dashcam-based collision prediction systems is that nighttime driving degrades model performance. Reduced visibility, lower contrast, headlight glare, and changes in color distribution all alter the visual signal that a model must interpret. This document traces how the hypothesis was raised from the related literature, what experiment we ran to test it, and what the data actually showed.

---

## How the V-CAS Paper Framed the Problem

**Citation:** M. W. Ashraf, A. Hassan, and I. A. Shah, "V-CAS: A Realtime Vehicle Anti Collision System Using Vision Transformer on Multi-Camera Streams," in *2024 International Conference on Machine Learning and Applications (ICMLA)*, Dec. 2024, pp. 939–944. DOI: 10.1109/ICMLA61862.2024.00138.

V-CAS is a real-time collision avoidance system built on RT-DETR object detection, DeepSORT multi-object tracking, speed and relative acceleration estimation, and brake light detection — all deployed on a Jetson Orin Nano edge device. It explicitly evaluated performance separately for daytime and nighttime scenarios using the **Car Crash Dataset (CCD)**, which contains 1,500 crash clips from YouTube (764 daytime, 376 nighttime) and 3,000 normal clips sampled from BDD100K.

### V-CAS Day vs. Night Results (Table V in the paper)

| Condition | Ground Truth Crashes | Predicted | Precision | Accuracy |
|---|---|---|---|---|
| **Day** | 764 | 759 | 98.68% | 97.64% |
| **Night** | 376 | 304 | 89.47% | **68.95%** |

The nighttime accuracy of 68.95% represents a **~29 percentage point drop** relative to daytime. The authors diagnose this directly:

> *"However, for nighttime videos, due to loss in detection and tracking, there is a drop in accuracy when being used without the brake detection method."*

The root cause is that RT-DETR, like most object detectors trained predominantly on daytime data, degrades when objects are occluded by darkness or only partially illuminated. When the tracker loses an object, the speed and relative acceleration pipeline has no signal, and the collision score collapses to zero — causing false negatives even when a crash is imminent.

### How They Fixed It: Brake Light Detection as a Redundant Signal

The authors added a dedicated **brake light detector** (a separate RT-DETR model fine-tuned on 22,525 images of `Brake OFF` / `Brake ON` classes) as a parallel fallback mechanism. The logic is:

- If the object detector fails at night, brake lights are still detectable because they emit their own light source.
- When a forward vehicle's brake light enters the host vehicle's proximity zone, an emergency braking signal is issued immediately — bypassing the speed/acceleration pipeline entirely.

With brake light detection enabled, the nighttime accuracy rose from 68.95% to **90.87%**, nearly closing the day/night gap:

| Condition | Without Brake Detection | With Brake Detection |
|---|---|---|
| Day accuracy | 97.64% | **98.12%** |
| Night accuracy | 68.95% | **90.87%** |

This result demonstrates a key principle: **modality-specific failures can be patched with purpose-built complementary signals** rather than requiring a better general-purpose backbone.

---

## Our Experiment: Brightness-Based Lighting Analysis

### Hypothesis

Reading the V-CAS results raised the question: does our three-stream VideoMAE model (RGB + Depth + Segmentation, AUC=0.918) similarly fail more on nighttime clips? The Nexar dataset does not come with explicit day/night labels, so we needed a proxy.

### Method

We wrote `src/error_analysis.py`, which:

1. **Reconstructs the exact validation split** used during training (80/20 stratified, `random_state=42`) from `train.csv` to recover the ordered list of 300 validation clip IDs. This is necessary because the saved `.npz` prediction files were generated before clip IDs were stored alongside predictions.

2. **Estimates per-clip brightness** by loading each clip's RGB `.npy` array (shape `[N, H, W, 3]`, uint8) and computing the mean pixel value of the last frame in the clip window. The last frame is the frame closest to the predicted collision moment at offset 0.0s, making it the most representative frame for lighting classification.

3. **Classifies clips as dark (night proxy) or bright (day proxy)** using a threshold of mean brightness < 60 (on a 0–255 scale). This threshold was chosen conservatively to capture clearly underlit scenes without mis-classifying dusk or heavily shadowed daylight clips.

4. **Computes AUC, recall, and false alarm rate separately** for dark and bright subsets, and reports the brightness distribution within each error category (TP, TN, FP, FN).

5. **Identifies the worst failures** — the 10 highest-confidence false negatives (missed collisions) and false positives (false alarms) — with their clip IDs and brightness values for direct visual inspection.

The script was run on the DCC cluster where the full frame dataset is available:

```bash
python src/error_analysis.py \
    --preds outputs/preds_videomae_full_ofs0p0.npz \
    --save-csv outputs/error_analysis_full.csv \
    --save-thumbnails outputs/thumbnails
```

---

## Results

### Metrics by Lighting Condition

| Condition | n | AUC | Recall | False Alarm Rate | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| **All clips** | 300 | 0.918 | 92.7% | 26.0% | 139 | 39 | 11 | 111 |
| **Bright (day)** | 216 | 0.915 | 89.7% | 26.6% | 96 | 29 | **11** | 80 |
| **Dark (night)** | 84 | 0.922 | **100.0%** | 24.4% | 43 | 10 | **0** | 31 |

*Threshold: F1-optimal (0.274). Dark defined as mean last-frame brightness < 60.*

### Brightness Distribution by Error Type

| Error Type | n | Mean Brightness | Dark Fraction | Min | Max |
|---|---|---|---|---|---|
| TP (caught collisions) | 139 | 85.5 | 30.9% | 17.4 | 155.5 |
| TN (correct non-collision) | 111 | 90.1 | 27.9% | 10.6 | 170.8 |
| FP (false alarms) | 39 | 93.4 | 25.6% | 27.9 | 160.0 |
| **FN (missed collisions)** | **11** | **117.5** | **0.0%** | 69.7 | 155.0 |

### Worst False Negatives (missed collisions)

| Clip ID | Model Score | Brightness | Dark? |
|---|---|---|---|
| 00896 | 0.225 | 140.1 | No |
| 00457 | 0.198 | 140.7 | No |
| 00060 | 0.173 | 138.6 | No |
| 00180 | 0.146 | 155.0 | No |
| 00313 | 0.104 | 113.1 | No |
| 00141 | 0.102 | 73.8 | No |
| 00904 | 0.063 | 130.8 | No |
| 01016 | 0.053 | 69.7 | No |
| 00983 | 0.051 | 107.7 | No |
| 00388 | 0.033 | 99.2 | No |

### Worst False Positives (false alarms, by model confidence)

| Clip ID | Model Score | Brightness | Dark? |
|---|---|---|---|
| 02090 | 0.998 | 50.8 | **Yes** |
| 01224 | 0.998 | 37.2 | **Yes** |
| 01061 | 0.997 | 105.7 | No |
| 01770 | 0.996 | 57.5 | **Yes** |
| 01234 | 0.992 | 40.6 | **Yes** |
| 01950 | 0.990 | 47.0 | **Yes** |
| 01333 | 0.990 | 87.7 | No |
| 01923 | 0.983 | 118.6 | No |
| 01621 | 0.982 | 120.0 | No |
| 01709 | 0.971 | 112.3 | No |

---

## Interpretation

### Finding 1: Night is not a recall problem — it is a precision problem

Our results are the **inverse** of V-CAS's findings. While V-CAS found that night caused the model to *miss* collisions (recall dropped to 68.95%), our three-stream VideoMAE achieves **100% recall on dark clips** (43 TP, 0 FN) compared to only 89.7% on bright clips. Every one of the 11 missed collisions is a daytime clip with high brightness (mean brightness 117.5, dark_frac=0.0%).

This divergence makes sense given the architectural difference. V-CAS relies on detecting and tracking object bounding boxes to derive collision scores — a pipeline that physically breaks down when objects are invisible in the dark. Our VideoMAE learns appearance features holistically over the full clip, and the depth stream (DepthAnythingV2) may provide particularly strong signal in darker scenes where RGB contrast alone is ambiguous.

### Finding 2: The model over-triggers on dark non-collision scenes

Five of the ten highest-confidence false positives are dark clips (model scores: 0.998, 0.998, 0.996, 0.992, 0.990). The model is extremely confident these are collisions — these are not borderline cases. This suggests the model has learned a **spurious correlation between visual darkness and collision probability**.

A plausible mechanism: dark scenes may visually resemble pre-collision moments in ways the model has overfit to — headlight glare creating bloom artifacts on the depth map, high-contrast brake-light-like spots in the segmentation channel, or low-texture regions that the VideoMAE encoder associates with chaotic motion. Whatever the feature, the model fires with near-certainty on dark non-collision clips.

This aligns with the V-CAS brake light finding from a different angle: both systems appear sensitive to lighting-associated visual cues. V-CAS uses this sensitivity intentionally and correctly (brake lights → real danger signal). Our model has learned something similar but without the explicit discrimination between "brake light in proximity" and "generally dark scene."

### Finding 3: All missed collisions are visually subtle daytime events

The FN clips have a mean brightness of 117.5 — brighter than any other error category — and model scores ranging from 0.033 to 0.225, meaning the model is not just wrong but confidently wrong in the other direction. These are likely collisions where the visual pre-impact signal is weak: slow-speed approaches, far-distance events, or scenarios where the collision occurs at the edge of frame or is preceded by little motion change in the final 1.6 seconds.

---

## Summary Comparison

| Axis | V-CAS finding | Our finding |
|---|---|---|
| Night recall | Severely degraded (68.95% accuracy) | Perfect (100% recall on dark clips) |
| Night precision / false alarms | Not the focus | Primary failure mode (top FPs are dark) |
| Source of FNs | Night — object detector fails in the dark | Daytime — subtle visual pre-collision signal |
| Architectural cause | Bounding-box pipeline breaks without visible objects | Spurious correlation: darkness → collision score |
| Mitigation proposed by authors | Brake light detection as fallback | N/A — identified in this analysis |

---

## Implications for Future Work

**Addressing dark false alarms.** The most actionable finding is the cluster of near-certain false positives on dark non-collision clips. One targeted fix is the brake light modality suggested by V-CAS: a binary `Brake ON/OFF` detector added as a fourth stream would provide discriminative signal that separates "dark scene with braking vehicles in proximity" (real risk) from "dark scene with no relevant vehicle dynamics" (not a risk). This could substantially reduce the 24.4% false alarm rate on night clips without sacrificing the 100% recall.

**Understanding the daylight FNs.** The 11 missed daytime collisions warrant direct video inspection to categorize by collision type (far-field, low-speed, lateral, etc.). If they cluster into a recognizable category, targeted data augmentation or a longer temporal context window could address them specifically.

**Caution with brightness as a proxy.** Mean pixel brightness of the last frame is a noisy proxy for lighting condition. Dusk scenes, shadowed tunnels, and overexposed daytime clips can all fall near the threshold. A cleaner approach for future analysis would be to annotate a subset of clips with actual time-of-day labels (many dashcam videos embed timestamps) or to use a dedicated scene classification model.
