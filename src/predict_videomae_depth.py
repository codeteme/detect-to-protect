"""
Two-stream VideoMAE RGB + Depth late fusion inference script.

Usage:
    python src/predict_videomae_depth.py
    python src/predict_videomae_depth.py \
        --checkpoint-path outputs/best_videomae_depth_ofs0p0.pt \
        --submission-path outputs/submission_videomae-depth-ofs0.0.csv
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
CKPT_PATH = OUT_DIR / "best_videomae_depth_ofs0p0.pt"
SUBMISSION_PATH = OUT_DIR / "submission_videomae-depth.csv"

BATCH_SIZE = 4
NUM_WORKERS = 4
DEFAULT_CLIP_LEN = 16
MODEL_NAME = "MCG-NJU/videomae-base"


class TwoStreamTestDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        frames_dir: Path,
        depth_dir: Path,
        processor: VideoMAEImageProcessor,
        clip_len: int,
    ):
        df = pd.read_csv(csv_path)
        df["id"] = df["id"].astype(str).str.zfill(5)
        available_ids = {p.stem for p in frames_dir.glob("*.npy")}
        df = df[df["id"].isin(available_ids)].reset_index(drop=True)
        if len(df) == 0:
            raise FileNotFoundError(f"No matching frame files found in {frames_dir}.")
        self.df = df
        self.frames_dir = frames_dir
        self.depth_dir = depth_dir
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

    def __getitem__(self, idx: int):
        video_id = self.df.iloc[idx]["id"]

        frames = np.load(self.frames_dir / f"{video_id}.npy")
        depth = np.load(self.depth_dir / f"{video_id}.npy")

        n = len(frames)
        start = max(n - self.clip_len, 0)

        rgb_clip = self._pad(frames[start:n], frames.dtype)
        dep_clip = self._pad(depth[start:n], depth.dtype)

        rgb_list = [rgb_clip[i] for i in range(self.clip_len)]
        rgb_pixels = self.processor(rgb_list, return_tensors="pt")["pixel_values"].squeeze(0)

        dep = dep_clip.astype(np.float32)
        dmin, dmax = dep.min(), dep.max()
        if dmax > dmin:
            dep = (dep - dmin) / (dmax - dmin) * 255.0
        dep_uint8 = dep.astype(np.uint8)
        dep_rgb = np.stack([dep_uint8] * 3, axis=-1)
        dep_list = [dep_rgb[i] for i in range(self.clip_len)]
        dep_pixels = self.processor(dep_list, return_tensors="pt")["pixel_values"].squeeze(0)

        return rgb_pixels, dep_pixels, video_id


class TwoStreamVideoMAE(nn.Module):
    def __init__(self, model_name: str):
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
        fused = torch.cat([rgb_feat, dep_feat], dim=-1)
        return self.fusion_head(fused).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run two-stream VideoMAE RGB+Depth prediction")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--submission-path", type=str, default="")
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
    model = TwoStreamVideoMAE(model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = TwoStreamTestDataset(TEST_CSV, FRAMES_DIR, DEPTH_DIR, processor, clip_len)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_mem,
    )
    print(f"Test set: {len(test_ds)} videos ({len(test_loader)} batches)", flush=True)

    all_ids, all_probs = [], []
    with torch.no_grad():
        for rgb_pixels, dep_pixels, batch_ids in tqdm(test_loader, desc="Inference"):
            rgb_pixels = rgb_pixels.to(device, non_blocking=True)
            dep_pixels = dep_pixels.to(device, non_blocking=True)
            probs = torch.sigmoid(model(rgb_pixels, dep_pixels))
            all_probs.extend(probs.cpu().numpy().tolist())
            all_ids.extend(list(batch_ids))

    sub_df = pd.DataFrame({"id": all_ids, "target": all_probs})
    sub_df.to_csv(submission_path, index=False)

    print(f"Saved: {submission_path}", flush=True)
    print(sub_df.head(10))
    print(
        f"Score distribution: min={sub_df.target.min():.4f} "
        f"mean={sub_df.target.mean():.4f} max={sub_df.target.max():.4f}"
    )


if __name__ == "__main__":
    main()