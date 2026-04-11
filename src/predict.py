"""
src/predict.py — Run inference on test set and generate submission CSV
Usage: python src/predict.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = "/hpc/group/coursess26/ids705/team-project/detect-to-protect"
DATA = f"{BASE}/data"

PATHS = {
    "test_csv":    f"{DATA}/test.csv",
    "frames_test": f"{DATA}/frames/test",
    "seg_test":    f"{DATA}/segmentation/test",
    "depth_test":  f"{DATA}/depth/test",
    "output_dir":  f"{BASE}/outputs",
}

CKPT_PATH       = f"{PATHS['output_dir']}/best_videomae.pt"
SUBMISSION_PATH = f"{PATHS['output_dir']}/submission.csv"
BATCH_SIZE      = 4
NUM_WORKERS     = 4
CLIP_LEN        = 100
FPS             = 10
MODEL_ID        = "MCG-NJU/videomae-base-finetuned-kinetics"


# ── Dataset ────────────────────────────────────────────────────────────────────
class NexarTestDataset(Dataset):
    def __init__(self, csv_path, frames_dir, seg_dir, depth_dir,
                 fps=10, clip_len=100):
        self.frames_dir = frames_dir
        self.seg_dir    = seg_dir
        self.depth_dir  = depth_dir
        self.fps        = fps
        self.clip_len   = clip_len
        df = pd.read_csv(csv_path)
        df["id"] = df["id"].astype(str).str.zfill(5)
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_id = self.df.iloc[idx]["id"]
        frames = np.load(os.path.join(self.frames_dir, f"{video_id}.npy"))
        seg    = np.load(os.path.join(self.seg_dir,    f"{video_id}.npy"))
        depth  = np.load(os.path.join(self.depth_dir,  f"{video_id}.npy"))

        N     = len(frames)
        start = max(N - self.clip_len, 0)
        end   = N

        frames = frames[start:end]
        seg    = seg[start:end]
        depth  = depth[start:end]

        T = len(frames)
        if T < self.clip_len:
            pad    = self.clip_len - T
            frames = np.concatenate([np.zeros((pad, *frames.shape[1:]), dtype=frames.dtype), frames])
            seg    = np.concatenate([np.zeros((pad, *seg.shape[1:]),    dtype=seg.dtype),    seg])
            depth  = np.concatenate([np.zeros((pad, *depth.shape[1:]),  dtype=depth.dtype),  depth])

        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        seg_t    = torch.from_numpy(seg.astype(np.float32)).unsqueeze(1)
        depth_t  = torch.from_numpy(depth.astype(np.float32)).unsqueeze(1)
        video    = torch.cat([frames_t, depth_t, seg_t], dim=1)  # [T, 5, H, W]
        return video, video_id


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_video(batch_video, num_frames, img_size, mean, std):
    x = batch_video[:, :, :3]
    b, t, c, h, w = x.shape
    idx = torch.linspace(0, t - 1, steps=num_frames, device=x.device).long()
    x   = x.index_select(1, idx).reshape(b * num_frames, c, h, w)
    x   = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    x   = x.reshape(b, num_frames, c, img_size, img_size)
    return (x - mean) / std


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} | val_auc={ckpt['val_auc']:.4f}")

    # Build model
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_ID, num_labels=1, ignore_mismatched_sizes=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    cfg        = ckpt.get("config", {})
    NUM_FRAMES = int(cfg.get("NUM_FRAMES", model.config.num_frames))
    IMG_SIZE   = int(cfg.get("IMG_SIZE",   model.config.image_size))
    mean       = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std        = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    print(f"NUM_FRAMES={NUM_FRAMES}, IMG_SIZE={IMG_SIZE}")

    # Test dataset
    test_ds = NexarTestDataset(
        csv_path=PATHS["test_csv"], frames_dir=PATHS["frames_test"],
        seg_dir=PATHS["seg_test"],  depth_dir=PATHS["depth_test"],
        fps=FPS, clip_len=CLIP_LEN,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=use_amp,
    )
    print(f"Test set: {len(test_ds)} videos ({len(test_loader)} batches)")

    # Inference
    all_ids, all_probs = [], []
    with torch.no_grad():
        for batch_video, batch_ids in tqdm(test_loader, desc="Inference"):
            batch_video  = batch_video.to(device, non_blocking=True)
            pixel_values = preprocess_video(batch_video, NUM_FRAMES, IMG_SIZE, mean, std)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(pixel_values=pixel_values).logits.squeeze(-1)
                probs  = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_ids.extend(list(batch_ids))

    # Save submission
    sub_df = pd.DataFrame({"id": all_ids, "target": all_probs})
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved: {SUBMISSION_PATH}")
    print(sub_df.head(10))
    print(f"Score distribution: min={sub_df.target.min():.4f} "
          f"mean={sub_df.target.mean():.4f} max={sub_df.target.max():.4f}")


if __name__ == "__main__":
    main()
