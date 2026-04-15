"""
Pipeline walkthrough visualization for video 00208.
Generates a figure showing RGB frames, depth maps, seg masks,
tubelet patch grid, and model prediction.

Usage:
    python src/visualize_pipeline.py

Output:
    outputs/pipeline_walkthrough_00208.png
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from PIL import Image
import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

VIDEO_ID = "00208"
TIME_OF_EVENT = 19.8
FPS = 10
CLIP_LEN = 16
MODEL_NAME = "MCG-NJU/videomae-base"
CKPT_PATH = OUT_DIR / "best_videomae_depth_ofs0p0.pt"

N_SHOW = 8
PATCH_SIZE = 16        # VideoMAE patch size
PROC_SIZE = 224        # processor resizes to 224x224
N_PATCHES = PROC_SIZE // PATCH_SIZE   # 14x14 grid


# ── Load data ──────────────────────────────────────────────────────────────────
frames = np.load(DATA_DIR / "frames" / "train" / f"{VIDEO_ID}.npy")
depth  = np.load(DATA_DIR / "depth"  / "train" / f"{VIDEO_ID}.npy")
seg    = np.load(DATA_DIR / "segmentation" / "train" / f"{VIDEO_ID}.npy")

n = len(frames)
anchor_frame = int(TIME_OF_EVENT * FPS)
end   = min(max(anchor_frame, 1), n)
start = max(end - CLIP_LEN, 0)

rgb_clip = frames[start:end]
dep_clip = depth[start:end]
seg_clip = seg[start:end]

def pad(clip, dtype):
    t = len(clip)
    if t < CLIP_LEN:
        pad_arr = np.zeros((CLIP_LEN - t, *clip.shape[1:]), dtype=dtype)
        clip = np.concatenate([pad_arr, clip], axis=0)
    return clip

rgb_clip = pad(rgb_clip, frames.dtype)
dep_clip = pad(dep_clip, depth.dtype)
seg_clip = pad(seg_clip, seg.dtype)

display_idx = np.linspace(0, CLIP_LEN - 1, N_SHOW, dtype=int)
timestamps  = [(i - (CLIP_LEN - 1)) / FPS for i in display_idx]


# ── Model inference ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoStreamVideoMAE(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.rgb_encoder = VideoMAEForVideoClassification.from_pretrained(
            model_name, num_labels=1, ignore_mismatched_sizes=True
        )
        self.dep_encoder = VideoMAEForVideoClassification.from_pretrained(
            model_name, num_labels=1, ignore_mismatched_sizes=True
        )
        hidden = self.rgb_encoder.config.hidden_size
        self.rgb_encoder.classifier = nn.Identity()
        self.dep_encoder.classifier = nn.Identity()
        self.fusion_head = nn.Linear(hidden * 2, 1)

    def forward(self, rgb_pixels, dep_pixels):
        rgb_feat = self.rgb_encoder(pixel_values=rgb_pixels).logits
        dep_feat = self.dep_encoder(pixel_values=dep_pixels).logits
        return self.fusion_head(torch.cat([rgb_feat, dep_feat], dim=-1)).squeeze(-1)

processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
model = TwoStreamVideoMAE(MODEL_NAME).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

rgb_list   = [rgb_clip[i] for i in range(CLIP_LEN)]
rgb_pixels = processor(rgb_list, return_tensors="pt")["pixel_values"].to(device)

dep = dep_clip.astype(np.float32)
dmin, dmax = dep.min(), dep.max()
if dmax > dmin:
    dep = (dep - dmin) / (dmax - dmin) * 255.0
dep_uint8  = dep.astype(np.uint8)
dep_rgb    = np.stack([dep_uint8] * 3, axis=-1)
dep_list   = [dep_rgb[i] for i in range(CLIP_LEN)]
dep_pixels = processor(dep_list, return_tensors="pt")["pixel_values"].to(device)

with torch.no_grad():
    logit = model(rgb_pixels, dep_pixels)
    prob  = torch.sigmoid(logit).item()

print(f"Collision probability: {prob:.3f}")


# ── Prepare resized impact frame for tubelet panel ─────────────────────────────
impact_frame = rgb_clip[-1]   # last frame = impact
impact_pil   = Image.fromarray(impact_frame).resize((PROC_SIZE, PROC_SIZE), Image.BILINEAR)
impact_arr   = np.array(impact_pil)

# pair frame: second-to-last (tubelets span 2 frames)
pair_frame   = rgb_clip[-2]
pair_pil     = Image.fromarray(pair_frame).resize((PROC_SIZE, PROC_SIZE), Image.BILINEAR)
pair_arr     = np.array(pair_pil)


# ── Figure — 5 rows ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(
    5, N_SHOW + 1,
    figure=fig,
    hspace=0.5,
    wspace=0.08,
    left=0.04, right=0.96,
    top=0.93, bottom=0.04,
)

# ── Rows 0-2: RGB, Depth, Seg ──────────────────────────────────────────────────
row_labels = ["RGB frames", "Depth maps", "Seg masks"]
cmaps      = [None, "plasma", "tab20"]

for row, (label, cmap) in enumerate(zip(row_labels, cmaps)):
    ax_label = fig.add_subplot(gs[row, 0])
    ax_label.axis("off")
    ax_label.text(1.0, 0.5, label, ha="right", va="center",
                  fontsize=9, color="#555", transform=ax_label.transAxes)

    for col, fi in enumerate(display_idx):
        ax = fig.add_subplot(gs[row, col + 1])
        ax.set_xticks([])
        ax.set_yticks([])

        if row == 0:
            ax.imshow(rgb_clip[fi])
            t = timestamps[col]
            color = "#c0392b" if fi == display_idx[-1] else "#333"
            ax.set_xlabel(f"{t:+.1f}s", fontsize=7, color=color, labelpad=2)
            for spine in ax.spines.values():
                spine.set_edgecolor("#c0392b" if fi == display_idx[-1] else "#ccc")
                spine.set_linewidth(1.5 if fi == display_idx[-1] else 0.5)
        elif row == 1:
            ax.imshow(dep_clip[fi].astype(np.float32), cmap=cmap)
            for spine in ax.spines.values():
                spine.set_edgecolor("#ccc"); spine.set_linewidth(0.5)
        else:
            ax.imshow(seg_clip[fi], cmap=cmap, vmin=0, vmax=20)
            for spine in ax.spines.values():
                spine.set_edgecolor("#ccc"); spine.set_linewidth(0.5)


# ── Row 3: Tubelet patch grid ──────────────────────────────────────────────────
ax_tub_label = fig.add_subplot(gs[3, 0])
ax_tub_label.axis("off")
ax_tub_label.text(1.0, 0.5, "Tubelet\npatch grid", ha="right", va="center",
                  fontsize=9, color="#555", transform=ax_tub_label.transAxes)

# Left: pair frame (t-0.2s) with grid
ax_pair = fig.add_subplot(gs[3, 1:4])
ax_pair.imshow(pair_arr)
ax_pair.set_xticks([]); ax_pair.set_yticks([])
ax_pair.set_title("frame t-0.2s", fontsize=8, color="#555", pad=3)
for i in range(N_PATCHES + 1):
    ax_pair.axhline(i * PATCH_SIZE - 0.5, color="cyan", linewidth=0.4, alpha=0.7)
    ax_pair.axvline(i * PATCH_SIZE - 0.5, color="cyan", linewidth=0.4, alpha=0.7)
# Highlight one tubelet pair
ax_pair.add_patch(patches.Rectangle(
    (4 * PATCH_SIZE, 5 * PATCH_SIZE), PATCH_SIZE, PATCH_SIZE,
    linewidth=2, edgecolor="yellow", facecolor="yellow", alpha=0.3
))
for spine in ax_pair.spines.values():
    spine.set_edgecolor("#aaa"); spine.set_linewidth(0.5)

# Right: impact frame (t=0) with same grid and same highlighted patch
ax_impact = fig.add_subplot(gs[3, 4:7])
ax_impact.imshow(impact_arr)
ax_impact.set_xticks([]); ax_impact.set_yticks([])
ax_impact.set_title("frame t=0 (impact)", fontsize=8, color="#c0392b", pad=3)
for i in range(N_PATCHES + 1):
    ax_impact.axhline(i * PATCH_SIZE - 0.5, color="cyan", linewidth=0.4, alpha=0.7)
    ax_impact.axvline(i * PATCH_SIZE - 0.5, color="cyan", linewidth=0.4, alpha=0.7)
ax_impact.add_patch(patches.Rectangle(
    (4 * PATCH_SIZE, 5 * PATCH_SIZE), PATCH_SIZE, PATCH_SIZE,
    linewidth=2, edgecolor="yellow", facecolor="yellow", alpha=0.3
))
for spine in ax_impact.spines.values():
    spine.set_edgecolor("#c0392b"); spine.set_linewidth(1.5)

# Annotation panel
ax_ann = fig.add_subplot(gs[3, 7:])
ax_ann.axis("off")
ax_ann.text(0.05, 0.80,
    f"Each cell = 16×16 px patch\n"
    f"Grid: {N_PATCHES}×{N_PATCHES} = {N_PATCHES*N_PATCHES} patches per frame\n"
    f"One tubelet = same patch from 2 consecutive frames\n"
    f"Total tubelets: {N_PATCHES*N_PATCHES * (CLIP_LEN//2)} = {N_PATCHES}×{N_PATCHES} × 8 pairs\n\n"
    f"Yellow = one example tubelet\n"
    f"(same spatial position, both frames)",
    transform=ax_ann.transAxes,
    fontsize=8, color="#333", va="top", linespacing=1.6
)


# ── Row 4: Prediction bar ──────────────────────────────────────────────────────
ax_pred_label = fig.add_subplot(gs[4, 0])
ax_pred_label.axis("off")
ax_pred_label.text(1.0, 0.5, "Prediction", ha="right", va="center",
                   fontsize=9, color="#555", transform=ax_pred_label.transAxes)

ax_pred = fig.add_subplot(gs[4, 1:])
ax_pred.axis("off")
bar_color = "#c0392b" if prob > 0.5 else "#2980b9"
ax_pred.barh(0, prob, height=0.35, color=bar_color, alpha=0.85)
ax_pred.barh(0, 1.0,  height=0.35, color="#eee", zorder=0)
ax_pred.set_xlim(0, 1)
ax_pred.set_ylim(-0.5, 0.5)
ax_pred.axvline(0.5, color="#aaa", linewidth=0.8, linestyle="--")
ax_pred.text(prob + 0.01, 0, f"{prob:.2f}", va="center", fontsize=11,
             fontweight="bold", color=bar_color)
ax_pred.text(0.5, -0.42, "Decision threshold (0.5)", ha="center",
             fontsize=8, color="#aaa", transform=ax_pred.transData)
verdict = "COLLISION PREDICTED" if prob > 0.5 else "NO COLLISION PREDICTED"
ax_pred.text(-0.01, 0, "Model output:", va="center", ha="right",
             fontsize=9, color="#555")
ax_pred.set_title(f"{verdict}  (VideoMAE RGB+Depth, val AUC = 0.814)",
                  fontsize=10, color=bar_color, pad=6)

fig.suptitle(
    f"Pipeline walkthrough — Video {VIDEO_ID}  |  Collision at t={TIME_OF_EVENT}s  |  "
    f"16 frames at 10fps → {N_PATCHES*N_PATCHES*(CLIP_LEN//2)} tubelets → collision probability",
    fontsize=11, y=0.97, color="#222"
)

out_path = OUT_DIR / "pipeline_walkthrough_00208.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()