"""
VideoMAE RGB baseline inference script.

Usage:
    python src/predict_videomae.py
    python src/predict_videomae.py --checkpoint-path outputs/best_videomae_clip16_ofs0p0.pt \
                                   --submission-path outputs/submission_videomae-clip16-ofs0.0.csv
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TEST_CSV = DATA_DIR / "test.csv"
FRAMES_DIR = DATA_DIR / "frames" / "test"
CKPT_PATH = OUT_DIR / "best_videomae_clip16_ofs0p0.pt"
SUBMISSION_PATH = OUT_DIR / "submission_videomae.csv"

BATCH_SIZE = 4
NUM_WORKERS = 4
DEFAULT_CLIP_LEN = 16
MODEL_NAME = "MCG-NJU/videomae-base"


class NexarFramesTestDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        frames_dir: Path,
        processor: VideoMAEImageProcessor,
        clip_len: int,
    ):
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
        self.processor = processor
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
            clip = np.concatenate(
                [np.zeros((pad, *clip.shape[1:]), dtype=clip.dtype), clip], axis=0
            )

        frames_list = [clip[i] for i in range(self.clip_len)]
        inputs = self.processor(frames_list, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)   # [T, C, H, W]
        return pixel_values, video_id


def parse_args():
    parser = argparse.ArgumentParser(description="Run VideoMAE prediction")
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
    cfg = ckpt.get("config", {})
    clip_len = int(cfg.get("clip_len", DEFAULT_CLIP_LEN))
    model_name = cfg.get("model_name", MODEL_NAME)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} | val_auc={ckpt['val_auc']:.4f}")
    print(f"Checkpoint path: {ckpt_path}")
    print(f"Using clip_len={clip_len}, model={model_name}")

    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = NexarFramesTestDataset(TEST_CSV, FRAMES_DIR, processor, clip_len)
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
        for pixel_values, batch_ids in tqdm(test_loader, desc="Inference"):
            pixel_values = pixel_values.to(device, non_blocking=True)
            outputs = model(pixel_values=pixel_values)
            probs = torch.sigmoid(outputs.logits.squeeze(-1))
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