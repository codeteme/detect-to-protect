"""
Two-stream VideoMAE RGB + Depth late fusion training script.

Usage:
    python src/train_videomae_depth.py
    python src/train_videomae_depth.py --anchor-offset-sec 0.5 --run-name videomae-depth-ofs0.5
"""

from pathlib import Path
import os
import argparse
import importlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from tqdm import tqdm

try:
    wandb = importlib.import_module("wandb")
except ModuleNotFoundError:
    wandb = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
FRAMES_DIR = DATA_DIR / "frames" / "train"
DEPTH_DIR = DATA_DIR / "depth" / "train"

SEED = 42
BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
FPS = 10
CLIP_LEN = 16
ANCHOR_OFFSET_SEC = 0.0

MODEL_NAME = "MCG-NJU/videomae-base"


class TwoStreamDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        frames_dir: Path,
        depth_dir: Path,
        processor: VideoMAEImageProcessor,
        fps: int,
        clip_len: int,
        anchor_offset_sec: float,
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
        self.fps = fps
        self.clip_len = clip_len
        self.anchor_offset_sec = anchor_offset_sec

    def __len__(self):
        return len(self.df)

    def _get_clip_indices(self, n, toe):
        if pd.notna(toe):
            anchor_frame = int((float(toe) - self.anchor_offset_sec) * self.fps)
            end = min(max(anchor_frame, 1), n)
            start = max(end - self.clip_len, 0)
        else:
            end = n
            start = max(end - self.clip_len, 0)
        return start, end

    def _pad(self, clip, dtype):
        t = len(clip)
        if t < self.clip_len:
            pad = self.clip_len - t
            clip = np.concatenate(
                [np.zeros((pad, *clip.shape[1:]), dtype=dtype), clip], axis=0
            )
        return clip

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_id = row["id"]

        frames = np.load(self.frames_dir / f"{video_id}.npy")   # [T, H, W, 3] uint8
        depth = np.load(self.depth_dir / f"{video_id}.npy")     # [T, H, W] float16

        n = len(frames)
        start, end = self._get_clip_indices(n, row.get("time_of_event"))

        rgb_clip = self._pad(frames[start:end], frames.dtype)    # [clip_len, H, W, 3]
        dep_clip = self._pad(depth[start:end], depth.dtype)      # [clip_len, H, W]

        # RGB stream — standard processor
        rgb_list = [rgb_clip[i] for i in range(self.clip_len)]
        rgb_pixels = self.processor(rgb_list, return_tensors="pt")["pixel_values"].squeeze(0)

        # Depth stream — normalize to [0, 255] uint8 and tile to 3 channels
        dep = dep_clip.astype(np.float32)
        dmin, dmax = dep.min(), dep.max()
        if dmax > dmin:
            dep = (dep - dmin) / (dmax - dmin) * 255.0
        dep_uint8 = dep.astype(np.uint8)                         # [clip_len, H, W]
        dep_rgb = np.stack([dep_uint8] * 3, axis=-1)             # [clip_len, H, W, 3]
        dep_list = [dep_rgb[i] for i in range(self.clip_len)]
        dep_pixels = self.processor(dep_list, return_tensors="pt")["pixel_values"].squeeze(0)

        y = torch.tensor(row["target"], dtype=torch.float32)
        return rgb_pixels, dep_pixels, y


class TwoStreamVideoMAE(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.rgb_encoder = VideoMAEForVideoClassification.from_pretrained(
            model_name, num_labels=1, ignore_mismatched_sizes=True
        )
        self.dep_encoder = VideoMAEForVideoClassification.from_pretrained(
            model_name, num_labels=1, ignore_mismatched_sizes=True
        )
        # Replace both classifiers with identity — we fuse before classifying
        hidden = self.rgb_encoder.config.hidden_size
        self.rgb_encoder.classifier = nn.Identity()
        self.dep_encoder.classifier = nn.Identity()
        self.fusion_head = nn.Linear(hidden * 2, 1)

    def forward(self, rgb_pixels, dep_pixels):
        rgb_feat = self.rgb_encoder(pixel_values=rgb_pixels).logits   # [B, hidden]
        dep_feat = self.dep_encoder(pixel_values=dep_pixels).logits   # [B, hidden]
        fused = torch.cat([rgb_feat, dep_feat], dim=-1)               # [B, hidden*2]
        return self.fusion_head(fused).squeeze(-1)                    # [B]


def run_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true, y_score = [], []

    for rgb_pixels, dep_pixels, y in tqdm(loader, leave=False):
        rgb_pixels = rgb_pixels.to(device, non_blocking=True)
        dep_pixels = dep_pixels.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(rgb_pixels, dep_pixels)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * rgb_pixels.size(0)
        y_true.extend(y.detach().cpu().tolist())
        y_score.extend(torch.sigmoid(logits).detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(y_true, y_score)
    return avg_loss, auc


def parse_args():
    parser = argparse.ArgumentParser(description="Train two-stream VideoMAE RGB+Depth")
    parser.add_argument("--anchor-offset-sec", type=float, default=ANCHOR_OFFSET_SEC)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    anchor_offset_sec = args.anchor_offset_sec
    offset_tag = str(anchor_offset_sec).replace(".", "p").replace("-", "m")
    run_name = args.run_name or f"videomae-depth-ofs{offset_tag}"
    best_ckpt = OUT_DIR / f"best_videomae_depth_ofs{offset_tag}.pt"

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = torch.cuda.is_available()
    print(f"Device: {device}", flush=True)

    wandb_enabled = (
        (not args.disable_wandb)
        and bool(os.getenv("WANDB_API_KEY"))
        and (wandb is not None)
    )
    if not args.disable_wandb and wandb is None:
        print("wandb not installed; running without W&B")
    if not args.disable_wandb and wandb is not None and not os.getenv("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is not set in environment")

    if wandb_enabled:
        assert wandb is not None
        os.environ["WANDB_CONSOLE"] = "off"
        os.environ["WANDB_DIR"] = str(OUT_DIR)
        os.environ["WANDB_CACHE_DIR"] = str(OUT_DIR / "wandb-cache")
        wandb.init(
            project="detect-to-protect",
            name=run_name,
            config={
                "seed": SEED,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "weight_decay": WEIGHT_DECAY,
                "clip_len": CLIP_LEN,
                "fps": FPS,
                "anchor_offset_sec": anchor_offset_sec,
                "model": MODEL_NAME,
                "fusion": "two-stream-late",
                "modalities": "rgb+depth",
            },
        )
        print(f"W&B run: {wandb.run.url}", flush=True)
    else:
        print("W&B disabled", flush=True)

    print(f"Loading {MODEL_NAME} ...", flush=True)
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    model = TwoStreamVideoMAE(MODEL_NAME).to(device)
    print("Model loaded.", flush=True)

    dataset = TwoStreamDataset(
        csv_path=TRAIN_CSV,
        frames_dir=FRAMES_DIR,
        depth_dir=DEPTH_DIR,
        processor=processor,
        fps=FPS,
        clip_len=CLIP_LEN,
        anchor_offset_sec=anchor_offset_sec,
    )
    print(f"Usable videos with frames: {len(dataset)}", flush=True)

    labels = dataset.df["target"].to_numpy(dtype=np.int64)
    idx = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        idx, test_size=VAL_SPLIT, random_state=SEED, stratify=labels
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_mem,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    best_val_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = run_epoch(model, val_loader, criterion, None, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} | "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}",
            flush=True,
        )

        if wandb_enabled:
            assert wandb is not None
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auc": train_auc,
                "val_loss": val_loss,
                "val_auc": val_auc,
            })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_auc": val_auc,
                    "config": {
                        "clip_len": CLIP_LEN,
                        "anchor_offset_sec": anchor_offset_sec,
                        "model_name": MODEL_NAME,
                        "fusion": "two-stream-late",
                        "modalities": "rgb+depth",
                    },
                },
                best_ckpt,
            )
            print(f"Saved best checkpoint (val_auc={val_auc:.4f})", flush=True)
            print(f"Checkpoint path: {best_ckpt}", flush=True)
            if wandb_enabled:
                assert wandb is not None
                wandb.log({"best_val_auc": val_auc})

    print(f"Done. Best val_auc={best_val_auc:.4f}", flush=True)
    if wandb_enabled:
        assert wandb is not None
        wandb.finish()


if __name__ == "__main__":
    main()