"""
Minimal baseline inference script (for TinyVideoCNN checkpoint).

Usage:
    python src/predict_baseline.py
    python src/predict_baseline.py --checkpoint-path outputs/best_baseline_scratch_clip64_ofs0p0.pt --submission-path outputs/submission_baseline-clip64-ofs0.0.csv
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TEST_CSV = DATA_DIR / "test.csv"
FRAMES_DIR = DATA_DIR / "frames" / "test"
CKPT_PATH = OUT_DIR / "best_baseline_scratch.pt"
SUBMISSION_PATH = OUT_DIR / "submission_baseline.csv"

BATCH_SIZE = 8
NUM_WORKERS = 4
DEFAULT_CLIP_LEN = 32


class NexarFramesTestDataset(Dataset):
    def __init__(self, csv_path: Path, frames_dir: Path, clip_len: int):
        df = pd.read_csv(csv_path)
        df["id"] = df["id"].astype(str).str.zfill(5)
        available_ids = {p.stem for p in frames_dir.glob("*.npy")}
        df = df[df["id"].isin(available_ids)].reset_index(drop=True)
        if len(df) == 0:
            raise FileNotFoundError(
                f"No matching frame files found in {frames_dir}. "
                "Check that your local data folder is populated."
            )
        self.df = df
        self.frames_dir = frames_dir
        self.clip_len = clip_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        video_id = self.df.iloc[idx]["id"]
        frames = np.load(self.frames_dir / f"{video_id}.npy")  # [T, H, W, 3]

        n = len(frames)
        start = max(n - self.clip_len, 0)
        clip = frames[start:n]

        t = len(clip)
        if t < self.clip_len:
            pad = self.clip_len - t
            zeros = np.zeros((pad, *clip.shape[1:]), dtype=clip.dtype)
            clip = np.concatenate([zeros, clip], axis=0)

        x = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 255.0  # [T, 3, H, W]
        return x, video_id


class TinyVideoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline prediction")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--submission-path", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = torch.cuda.is_available()
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else CKPT_PATH
    submission_path = Path(args.submission_path) if args.submission_path else SUBMISSION_PATH

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    clip_len = int(ckpt.get("config", {}).get("clip_len", DEFAULT_CLIP_LEN))
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} | val_auc={ckpt['val_auc']:.4f}")
    print(f"Checkpoint path: {ckpt_path}")
    print(f"Using clip_len={clip_len}")

    model = TinyVideoCNN().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = NexarFramesTestDataset(TEST_CSV, FRAMES_DIR, clip_len)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_mem,
    )
    print(f"Test set: {len(test_ds)} videos ({len(test_loader)} batches)")

    all_ids, all_probs = [], []
    with torch.no_grad():
        for x, batch_ids in tqdm(test_loader, desc="Inference"):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_ids.extend(list(batch_ids))

    sub_df = pd.DataFrame({"id": all_ids, "target": all_probs})
    sub_df.to_csv(submission_path, index=False)

    print(f"Saved: {submission_path}")
    print(sub_df.head(10))
    print(
        f"Score distribution: min={sub_df.target.min():.4f} "
        f"mean={sub_df.target.mean():.4f} max={sub_df.target.max():.4f}"
    )


if __name__ == "__main__":
    main()
