"""
Three-stream VideoMAE RGB + Depth + Segmentation late fusion training script.

Usage:
    python src/train_videomae_full.py
    python src/train_videomae_full.py --anchor-offset-sec 0.5 --run-name videomae-full-ofs0.5
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
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, get_cosine_schedule_with_warmup
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
SEG_DIR = DATA_DIR / "segmentation" / "train"

SEED = 42
BATCH_SIZE = 4
ACCUM_STEPS = 4  # Effective batch size = BATCH_SIZE * ACCUM_STEPS
NUM_WORKERS = 4
EPOCHS = 10
FREEZE_EPOCHS = 2 # Epochs to train only the head
HEAD_LR = 1e-4
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
FPS = 10
CLIP_LEN = 16
ANCHOR_OFFSET_SEC = 0.0
MODEL_NAME = "MCG-NJU/videomae-base"


class ThreeStreamDataset(Dataset):
    def __init__(self, csv_path, frames_dir, depth_dir, seg_dir,
                 processor, fps, clip_len, anchor_offset_sec):
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
        row = self.df.iloc[idx]
        video_id = row["id"]

        frames = np.load(self.frames_dir / f"{video_id}.npy")
        depth = np.load(self.depth_dir / f"{video_id}.npy")
        seg = np.load(self.seg_dir / f"{video_id}.npy")

        n = len(frames)
        start, end = self._get_clip_indices(n, row.get("time_of_event"))

        rgb_clip = self._pad(frames[start:end], frames.dtype)
        dep_clip = self._pad(depth[start:end], depth.dtype)
        seg_clip = self._pad(seg[start:end], seg.dtype)

        rgb_list = [rgb_clip[i] for i in range(self.clip_len)]
        rgb_pixels = self.processor(rgb_list, return_tensors="pt")["pixel_values"].squeeze(0)
        dep_pixels = self._to_pixels(dep_clip, normalize=True)   
        seg_pixels = self._to_pixels(seg_clip, normalize=False)  

        y = torch.tensor(row["target"], dtype=torch.float32)
        return rgb_pixels, dep_pixels, seg_pixels, y


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
        
        # Upgraded MLP Fusion Head
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
        fused = torch.cat([rgb_feat, dep_feat, seg_feat], dim=-1)
        return self.fusion_head(fused).squeeze(-1)


def run_epoch(model, loader, criterion, optimizer, scheduler, device, accum_steps):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true, y_score = [], []

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for i, (rgb_pixels, dep_pixels, seg_pixels, y) in enumerate(tqdm(loader, leave=False)):
        rgb_pixels = rgb_pixels.to(device, non_blocking=True)
        dep_pixels = dep_pixels.to(device, non_blocking=True)
        seg_pixels = seg_pixels.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(rgb_pixels, dep_pixels, seg_pixels)
            loss = criterion(logits, y)
            
            if is_train:
                # Normalize loss for gradient accumulation
                loss = loss / accum_steps
                loss.backward()

                # Step optimizer every `accum_steps` or at the end of the loader
                if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

        # Re-scale loss for accurate logging
        step_loss = loss.item() * accum_steps if is_train else loss.item()
        total_loss += step_loss * rgb_pixels.size(0)
        y_true.extend(y.detach().cpu().tolist())
        y_score.extend(torch.sigmoid(logits).detach().cpu().tolist())

    return total_loss / len(loader.dataset), roc_auc_score(y_true, y_score)


def parse_args():
    parser = argparse.ArgumentParser(description="Train three-stream VideoMAE RGB+Depth+Seg")
    parser.add_argument("--anchor-offset-sec", type=float, default=ANCHOR_OFFSET_SEC)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--head-lr", type=float, default=HEAD_LR)
    parser.add_argument("--backbone-lr", type=float, default=BACKBONE_LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--accum-steps", type=int, default=ACCUM_STEPS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--freeze-epochs", type=int, default=FREEZE_EPOCHS)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    anchor_offset_sec = args.anchor_offset_sec
    offset_tag = str(anchor_offset_sec).replace(".", "p").replace("-", "m")
    run_name = args.run_name or f"videomae-full-ofs{offset_tag}"
    best_ckpt = OUT_DIR / f"best_videomae_full_ofs{offset_tag}.pt"

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
                "accum_steps": args.accum_steps,
                "effective_batch_size": args.batch_size * args.accum_steps,
                "epochs": args.epochs,
                "freeze_epochs": args.freeze_epochs,
                "head_lr": args.head_lr,
                "backbone_lr": args.backbone_lr,
                "weight_decay": WEIGHT_DECAY,
                "clip_len": CLIP_LEN,
                "fps": FPS,
                "anchor_offset_sec": anchor_offset_sec,
                "model": MODEL_NAME,
                "fusion": "three-stream-late-mlp",
                "modalities": "rgb+depth+seg",
            },
        )
    else:
        print("W&B disabled", flush=True)

    print(f"Loading {MODEL_NAME} ...", flush=True)
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    model = ThreeStreamVideoMAE(MODEL_NAME).to(device)
    print("Model loaded.", flush=True)

    dataset = ThreeStreamDataset(
        csv_path=TRAIN_CSV, frames_dir=FRAMES_DIR, depth_dir=DEPTH_DIR, seg_dir=SEG_DIR,
        processor=processor, fps=FPS, clip_len=CLIP_LEN, anchor_offset_sec=anchor_offset_sec,
    )

    labels = dataset.df["target"].to_numpy(dtype=np.int64)
    idx = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(idx, test_size=VAL_SPLIT, random_state=SEED, stratify=labels)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_mem)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_mem)

    criterion = nn.BCEWithLogitsLoss()
    best_val_auc = 0.0

    # Initial Setup: Freeze backbones, train only the head
    print(f"Freezing backbones for the first {args.freeze_epochs} epochs...", flush=True)
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.head_lr, weight_decay=WEIGHT_DECAY)
    scheduler = None # No scheduler during the frozen phase

    for epoch in range(1, args.epochs + 1):
        
        # Transition: Unfreeze and apply differential LRs & scheduler
        if epoch == args.freeze_epochs + 1:
            print("Unfreezing backbones. Applying differential learning rates...", flush=True)
            for param in model.parameters():
                param.requires_grad = True
                
            optimizer = torch.optim.AdamW([
                {"params": model.rgb_encoder.parameters(), "lr": args.backbone_lr},
                {"params": model.dep_encoder.parameters(), "lr": args.backbone_lr},
                {"params": model.seg_encoder.parameters(), "lr": args.backbone_lr},
                {"params": model.fusion_head.parameters(), "lr": args.head_lr},
            ], weight_decay=WEIGHT_DECAY)

            total_train_steps = (len(train_loader) // args.accum_steps) * (args.epochs - args.freeze_epochs)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=int(total_train_steps * 0.1), 
                num_training_steps=total_train_steps
            )

        train_loss, train_auc = run_epoch(model, train_loader, criterion, optimizer, scheduler, device, args.accum_steps)
        val_loss, val_auc = run_epoch(model, val_loader, criterion, None, None, device, args.accum_steps)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} | "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}",
            flush=True,
        )

        if wandb_enabled:
            assert wandb is not None
            current_lr = optimizer.param_groups[0]['lr'] if optimizer else 0.0
            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc,
                       "val_loss": val_loss, "val_auc": val_auc, "learning_rate": current_lr})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "config": {"clip_len": CLIP_LEN, "anchor_offset_sec": anchor_offset_sec,
                           "model_name": MODEL_NAME, "fusion": "three-stream-late-mlp",
                           "modalities": "rgb+depth+seg"},
            }, best_ckpt)
            if wandb_enabled:
                assert wandb is not None
                wandb.log({"best_val_auc": val_auc})

    print(f"Done. Best val_auc={best_val_auc:.4f}", flush=True)
    if wandb_enabled:
        assert wandb is not None
        wandb.finish()


if __name__ == "__main__":
    main()