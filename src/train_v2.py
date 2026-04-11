"""
src/train_v2.py — VideoMAE with unfrozen last 2 transformer blocks + classifier head
Usage: python src/train_v2.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers import VideoMAEForVideoClassification
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = "/hpc/group/coursess26/ids705/team-project/detect-to-protect"
DATA = f"{BASE}/data"

PATHS = {
    "train_csv":    f"{DATA}/train.csv",
    "frames_train": f"{DATA}/frames/train",
    "seg_train":    f"{DATA}/segmentation/train",
    "depth_train":  f"{DATA}/depth/train",
    "output_dir":   f"{BASE}/outputs",
}
os.makedirs(PATHS["output_dir"], exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEED           = 42
BATCH_SIZE     = 4
NUM_WORKERS    = 4
EPOCHS         = 20
LR_HEAD        = 1e-4   # classifier head
LR_BACKBONE    = 1e-5   # unfrozen transformer blocks
WEIGHT_DECAY   = 1e-4
CLIP_LEN       = 100
FPS            = 10
VAL_SPLIT      = 0.2
UNFREEZE_LAST  = 2      # number of transformer blocks to unfreeze from the end
BEST_CKPT      = f"{PATHS['output_dir']}/best_videomae_v2.pt"
MODEL_ID       = "MCG-NJU/videomae-base-finetuned-kinetics"


# ── Dataset ────────────────────────────────────────────────────────────────────
class NexarDataset(Dataset):
    def __init__(self, csv_path, frames_dir, seg_dir, depth_dir,
                 split="train", fps=10, clip_len=100):
        self.frames_dir = frames_dir
        self.seg_dir    = seg_dir
        self.depth_dir  = depth_dir
        self.split      = split
        self.fps        = fps
        self.clip_len   = clip_len
        df = pd.read_csv(csv_path)
        df["id"] = df["id"].astype(str).str.zfill(5)
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        video_id = row["id"]
        frames = np.load(os.path.join(self.frames_dir, f"{video_id}.npy"))
        seg    = np.load(os.path.join(self.seg_dir,    f"{video_id}.npy"))
        depth  = np.load(os.path.join(self.depth_dir,  f"{video_id}.npy"))
        N = len(frames)
        if self.split == "train":
            toe = row.get("time_of_event")
            if pd.notna(toe):
                end   = min(int(toe * self.fps), N)
                start = max(end - self.clip_len, 0)
            else:
                start = max(N - self.clip_len, 0)
                end   = N
        else:
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
        video    = torch.cat([frames_t, depth_t, seg_t], dim=1)
        label    = torch.tensor(row["target"], dtype=torch.float32)
        return video, label


# ── VideoMAE preprocessing ─────────────────────────────────────────────────────
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
    pin_mem = torch.cuda.is_available()
    print(f"Device : {device}")
    print(f"AMP    : {use_amp}")

    # Dataset + split
    full_ds = NexarDataset(
        csv_path=PATHS["train_csv"], frames_dir=PATHS["frames_train"],
        seg_dir=PATHS["seg_train"],  depth_dir=PATHS["depth_train"],
        split="train", fps=FPS, clip_len=CLIP_LEN,
    )
    labels_np          = full_ds.df["target"].astype(int).values
    all_idx            = np.arange(len(full_ds))
    train_idx, val_idx = train_test_split(
        all_idx, test_size=VAL_SPLIT, random_state=SEED, stratify=labels_np
    )
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=pin_mem)
    val_loader   = DataLoader(Subset(full_ds, val_idx),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_mem)
    print(f"Train  : {len(train_idx)} samples ({len(train_loader)} batches)")
    print(f"Val    : {len(val_idx)} samples ({len(val_loader)} batches)")

    # Model
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_ID, num_labels=1, ignore_mismatched_sizes=True,
    )

    # Freeze all params first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last UNFREEZE_LAST transformer blocks
    num_blocks = len(model.videomae.encoder.layer)
    for i in range(num_blocks - UNFREEZE_LAST, num_blocks):
        for p in model.videomae.encoder.layer[i].parameters():
            p.requires_grad = True
    print(f"Unfrozen transformer blocks: {num_blocks - UNFREEZE_LAST} to {num_blocks - 1}")

    # Always unfreeze classifier head
    for p in model.classifier.parameters():
        p.requires_grad = True

    model      = model.to(device)
    NUM_FRAMES = int(model.config.num_frames)
    IMG_SIZE   = int(model.config.image_size)
    mean       = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std        = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    criterion = nn.BCEWithLogitsLoss()

    # Separate param groups for different LRs
    backbone_params   = [p for n, p in model.named_parameters()
                         if p.requires_grad and "classifier" not in n]
    head_params       = [p for p in model.classifier.parameters()]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params,     "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scaler       = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch_video, batch_label in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train"):
            batch_video  = batch_video.to(device, non_blocking=True)
            batch_label  = batch_label.to(device, non_blocking=True)
            pixel_values = preprocess_video(batch_video, NUM_FRAMES, IMG_SIZE, mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(pixel_values=pixel_values).logits.squeeze(-1)
                loss   = criterion(logits, batch_label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * batch_video.size(0)
        train_loss /= len(train_loader.dataset)

        # Val
        model.eval()
        val_loss        = 0.0
        y_true, y_score = [], []
        with torch.no_grad():
            for batch_video, batch_label in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} val"):
                batch_video  = batch_video.to(device, non_blocking=True)
                batch_label  = batch_label.to(device, non_blocking=True)
                pixel_values = preprocess_video(batch_video, NUM_FRAMES, IMG_SIZE, mean, std)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(pixel_values=pixel_values).logits.squeeze(-1)
                    loss   = criterion(logits, batch_label)
                probs     = torch.sigmoid(logits)
                val_loss += loss.item() * batch_video.size(0)
                y_true.extend(batch_label.cpu().tolist())
                y_score.extend(probs.cpu().tolist())
        val_loss /= len(val_loader.dataset)
        val_auc   = roc_auc_score(y_true, y_score)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_auc={val_auc:.4f} | best={best_val_auc:.4f}"
        )
        sys.stdout.flush()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "config": {"NUM_FRAMES": NUM_FRAMES, "IMG_SIZE": IMG_SIZE},
            }, BEST_CKPT)
            print(f"  -> Saved best checkpoint (val_auc={val_auc:.4f})")
            sys.stdout.flush()

    print(f"\nDone. Best val_auc={best_val_auc:.4f} | Checkpoint: {BEST_CKPT}")


if __name__ == "__main__":
    main()
