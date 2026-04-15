"""
Three-stream VideoMAE RGB + Depth + Segmentation late fusion inference script.

Usage:
    python src/predict_videomae_full.py
    python src/predict_videomae_full.py \
        --checkpoint-path outputs/best_videomae_full_ofs0p0.pt \
        --submission-path outputs/submission_videomae-full-ofs0.0.csv
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TEST_CSV = DATA_DIR / "test.csv"
FRAMES_DIR = DATA_DIR / "frames" / "test"
DEPTH_DIR = DATA_DIR / "depth" / "test"
SEG_DIR = DATA_DIR / "segmentation" / "test"
CKPT_PATH = OUT_DIR / "best_videomae_full_ofs0p0.pt"
SUBMISSION_PATH = OUT_DIR / "submission_videomae-full.csv"

BATCH_SIZE = 4
NUM_WORKERS = 4
DEFAULT_CLIP_LEN = 16
MODEL_NAME = "MCG-NJU/videomae-base"


class ThreeStreamTestDataset(Dataset):
    def __init__(self, csv_path, frames_dir, depth_dir, seg_dir, processor, clip_len):
        df = pd.read_csv(csv_path)
        df["id"] = df["id"].astype(str).str.zfill(5)
        available_ids = {p.stem for p in frames_dir.glob("*.npy")}
        df = df[df["id"].isin(available_ids)].reset_index(drop=True)
        if len(df) == 0:
            raise FileNotFoundError(f"No matching frame files found in {frames_dir}.")
        self.df = df
        self.frames_dir = frames_dir
        self.depth_dir = depth_dir
        self.seg_dir = seg_dir
        self.processor = processor
        self.clip_len = clip_len

    def __len__(self):
        return len(self.df)

    def _pad(self, clip, dtype):
        t = len(clip)
        if t < self.clip_len:
            pad = self.clip_len - t
            clip = np.concatenate(
                [np.zeros((pad, *clip.shape[1:]), dtype=dtype), clip], axis=0
            )
        return clip

    def _to_pixels(self, clip_hw, normalize=False):
        arr = clip_hw.astype(np.float32)
        if normalize:
            dmin, dmax = arr.min(), arr.max()
            if dmax > dmin:
                arr = (arr - dmin) / (dmax - dmin) * 255.0
        arr_uint8 = arr.astype(np.uint8)
        tiled = np.stack([arr_uint8] * 3, axis=-1)
        frames_list = [tiled[i] for i in range(self.clip_len)]
        return self.processor(frames_list, return_tensors="pt")["pixel_values"].squeeze(0)

    def __getitem__(self, idx):
        video_id = self.df.iloc[idx]["id"]

        frames = np.load(self.frames_dir / f"{video_id}.npy")
        depth = np.load(self.depth_dir / f"{video_id}.npy")
        seg = np.load(self.seg_dir / f"{video_id}.npy")

        n = len(frames)
        start = max(n - self.clip_len, 0)

        rgb_clip = self._pad(frames[start:n], frames.dtype)
        dep_clip = self._pad(depth[start:n], depth.dtype)
        seg_clip = self._pad(seg[start:n], seg.dtype)

        rgb_list = [rgb_clip[i] for i in range(self.clip_len)]
        rgb_pixels = self.processor(rgb_list, return_tensors="pt")["pixel_values"].squeeze(0)
        dep_pixels = self._to_pixels(dep_clip, normalize=True)
        seg_pixels = self._to_pixels(seg_clip, normalize=False)

        return rgb_pixels, dep_pixels, seg_pixels, video_id


class ThreeStreamVideoMAE(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.rgb_encoder = VideoMAEForVideoClassification.from_pretrained(
            model_name, num_labels=1, ignore_mismatched_sizes=True
        )
        self.dep_encoder = VideoMAEForVideoClassification.from_pretrained(
            model_name, num_labels=1, ignore_mismatched_sizes=True
        )
        self.seg_encoder = VideoMAEForVideoClassification.from_pretrained(
            model_name, num_labels=1, ignore_mismatched_sizes=True
        )
        hidden = self.rgb_encoder.config.hidden_size
        self.rgb_encoder.classifier = nn.Identity()
        self.dep_encoder.classifier = nn.Identity()
        self.seg_encoder.classifier = nn.Identity()
        
        # Upgraded MLP Fusion Head to match training script
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, 1)
        )

    def forward(self, rgb_pixels, dep_pixels, seg_pixels):
        rgb_feat = self.rgb_encoder(pixel_values=rgb_pixels).logits
        dep_feat = self.dep_encoder(pixel_values=dep_pixels).logits
        seg_feat = self.seg_encoder(pixel_values=seg_pixels).logits
        return self.fusion_head(torch.cat([rgb_feat, dep_feat, seg_feat], dim=-1)).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run three-stream VideoMAE RGB+Depth+Seg prediction")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--submission-path", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = torch.cuda.is_available()
    print(f"Device: {device}", flush=True)

    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else CKPT_PATH
    submission_path = Path(args.submission_path) if args.submission_path else SUBMISSION_PATH

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    clip_len = int(cfg.get("clip_len", DEFAULT_CLIP_LEN))
    model_name = cfg.get("model_name", MODEL_NAME)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} | val_auc={ckpt['val_auc']:.4f}", flush=True)
    print(f"clip_len={clip_len}, model={model_name}", flush=True)

    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = ThreeStreamVideoMAE(model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = ThreeStreamTestDataset(TEST_CSV, FRAMES_DIR, DEPTH_DIR, SEG_DIR, processor, clip_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin_mem)
    print(f"Test set: {len(test_ds)} videos ({len(test_loader)} batches)", flush=True)

    all_ids, all_probs = [], []
    with torch.no_grad():
        for rgb_pixels, dep_pixels, seg_pixels, batch_ids in tqdm(test_loader, desc="Inference"):
            rgb_pixels = rgb_pixels.to(device, non_blocking=True)
            dep_pixels = dep_pixels.to(device, non_blocking=True)
            seg_pixels = seg_pixels.to(device, non_blocking=True)
            probs = torch.sigmoid(model(rgb_pixels, dep_pixels, seg_pixels))
            all_probs.extend(probs.cpu().numpy().tolist())
            all_ids.extend(list(batch_ids))

    sub_df = pd.DataFrame({"id": all_ids, "target": all_probs})
    sub_df.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}", flush=True)
    print(sub_df.head(10))
    print(f"Score distribution: min={sub_df.target.min():.4f} "
          f"mean={sub_df.target.mean():.4f} max={sub_df.target.max():.4f}")


if __name__ == "__main__":
    main()