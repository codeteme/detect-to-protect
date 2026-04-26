"""
Re-run validation inference for a saved checkpoint and save y_true + y_scores.

Reconstructs the exact same val split used during training (seed=42, 80/20 stratified)
so the AUC here should exactly match the stored val_auc in the checkpoint.

Usage (run from project root on the cluster):

  # VideoMAE RGB
  python src/eval_save_preds.py --type rgb \
      --checkpoint outputs/best_videomae_clip16_ofs0p0.pt \
      --out outputs/preds_videomae_rgb_ofs0p0.npz

  # VideoMAE RGB offset 0.5
  python src/eval_save_preds.py --type rgb \
      --checkpoint outputs/best_videomae_clip16_ofs0p5.pt \
      --out outputs/preds_videomae_rgb_ofs0p5.npz

  # VideoMAE RGB + Depth
  python src/eval_save_preds.py --type depth \
      --checkpoint outputs/best_videomae_depth_ofs0p0.pt \
      --out outputs/preds_videomae_depth_ofs0p0.npz

  # VideoMAE RGB + Depth offset 0.5
  python src/eval_save_preds.py --type depth \
      --checkpoint outputs/best_videomae_depth_ofs0p5.pt \
      --out outputs/preds_videomae_depth_ofs0p5.npz

  # VideoMAE RGB + Seg
  python src/eval_save_preds.py --type seg \
      --checkpoint outputs/best_videomae_seg_ofs0p0.pt \
      --out outputs/preds_videomae_seg_ofs0p0.npz

  # VideoMAE RGB + Seg offset 0.5
  python src/eval_save_preds.py --type seg \
      --checkpoint outputs/best_videomae_seg_ofs0p5.pt \
      --out outputs/preds_videomae_seg_ofs0p5.npz

  # VideoMAE RGB + Depth + Seg
  python src/eval_save_preds.py --type full \
      --checkpoint outputs/best_videomae_full_ofs0p0.pt \
      --out outputs/preds_videomae_full_ofs0p0.npz

  # VideoMAE RGB + Depth + Seg offset 0.5
  python src/eval_save_preds.py --type full \
      --checkpoint outputs/best_videomae_full_ofs0p5.pt \
      --out outputs/preds_videomae_full_ofs0p5.npz

  # TinyVideoCNN baselines
  python src/eval_save_preds.py --type baseline \
      --checkpoint outputs/best_baseline_scratch.pt \
      --out outputs/preds_baseline_scratch.npz

  python src/eval_save_preds.py --type baseline \
      --checkpoint outputs/best_baseline_scratch_clip32_ofs0p0.pt \
      --out outputs/preds_baseline_clip32_ofs0p0.npz

  python src/eval_save_preds.py --type baseline \
      --checkpoint outputs/best_baseline_scratch_clip32_ofs0p5.pt \
      --out outputs/preds_baseline_clip32_ofs0p5.npz

  python src/eval_save_preds.py --type baseline \
      --checkpoint outputs/best_baseline_scratch_clip64_ofs0p0.pt \
      --out outputs/preds_baseline_clip64_ofs0p0.npz

  python src/eval_save_preds.py --type baseline \
      --checkpoint outputs/best_baseline_scratch_clip100_ofs0p0.pt \
      --out outputs/preds_baseline_clip100_ofs0p0.npz
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEED = 42
VAL_SPLIT = 0.2
BATCH_SIZE = 4
NUM_WORKERS = 0  # login node has limited RAM; no worker processes

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Dataset + model factories — one per training script
# ---------------------------------------------------------------------------

def build_rgb(cfg):
    from train_videomae import NexarFramesDataset
    from transformers import VideoMAEImageProcessor
    processor = VideoMAEImageProcessor.from_pretrained(cfg["model_name"])
    return NexarFramesDataset(
        csv_path=DATA_DIR / "train.csv",
        frames_dir=DATA_DIR / "frames" / "train",
        processor=processor,
        fps=10,
        clip_len=cfg["clip_len"],
        anchor_offset_sec=cfg["anchor_offset_sec"],
    )


def build_depth(cfg):
    from train_videomae_depth import TwoStreamDataset
    from transformers import VideoMAEImageProcessor
    processor = VideoMAEImageProcessor.from_pretrained(cfg["model_name"])
    return TwoStreamDataset(
        csv_path=DATA_DIR / "train.csv",
        frames_dir=DATA_DIR / "frames" / "train",
        depth_dir=DATA_DIR / "depth" / "train",
        processor=processor,
        fps=10,
        clip_len=cfg["clip_len"],
        anchor_offset_sec=cfg["anchor_offset_sec"],
    )


def build_seg(cfg):
    from train_videomae_seg import TwoStreamDataset
    from transformers import VideoMAEImageProcessor
    processor = VideoMAEImageProcessor.from_pretrained(cfg["model_name"])
    return TwoStreamDataset(
        csv_path=DATA_DIR / "train.csv",
        frames_dir=DATA_DIR / "frames" / "train",
        seg_dir=DATA_DIR / "segmentation" / "train",
        processor=processor,
        fps=10,
        clip_len=cfg["clip_len"],
        anchor_offset_sec=cfg["anchor_offset_sec"],
    )


def build_full(cfg):
    from train_videomae_full import ThreeStreamDataset
    from transformers import VideoMAEImageProcessor
    processor = VideoMAEImageProcessor.from_pretrained(cfg["model_name"])
    return ThreeStreamDataset(
        csv_path=DATA_DIR / "train.csv",
        frames_dir=DATA_DIR / "frames" / "train",
        depth_dir=DATA_DIR / "depth" / "train",
        seg_dir=DATA_DIR / "segmentation" / "train",
        processor=processor,
        fps=10,
        clip_len=cfg["clip_len"],
        anchor_offset_sec=cfg["anchor_offset_sec"],
    )


def build_baseline(cfg):
    from train_baseline import NexarFramesDataset
    return NexarFramesDataset(
        csv_path=DATA_DIR / "train.csv",
        frames_dir=DATA_DIR / "frames" / "train",
        fps=10,
        clip_len=cfg["clip_len"],
        anchor_offset_sec=cfg["anchor_offset_sec"],
    )


# ---------------------------------------------------------------------------
# Inference loops — one per batch shape
# ---------------------------------------------------------------------------

def infer_single(model, loader, device):
    """For RGB VideoMAE and baseline: batch = (pixel_values, y)."""
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for pixel_values, y in tqdm(loader, desc="Eval", leave=False):
            pixel_values = pixel_values.to(device, non_blocking=True)
            out = model(pixel_values=pixel_values)
            probs = torch.sigmoid(out.logits.squeeze(-1))
            y_score.extend(probs.cpu().numpy().tolist())
            y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_score)


def infer_baseline_model(model, loader, device):
    """TinyVideoCNN forward pass differs — no keyword arg."""
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x = x.to(device, non_blocking=True)
            logits = model(x).squeeze(-1)
            probs = torch.sigmoid(logits)
            y_score.extend(probs.cpu().numpy().tolist())
            y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_score)


def infer_two_stream(model, loader, device):
    """For depth/seg: batch = (rgb_pixels, aux_pixels, y)."""
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for a, b, y in tqdm(loader, desc="Eval", leave=False):
            a = a.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            logits = model(a, b)
            probs = torch.sigmoid(logits.squeeze(-1))
            y_score.extend(probs.cpu().numpy().tolist())
            y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_score)


def infer_three_stream(model, loader, device):
    """For full: batch = (rgb_pixels, dep_pixels, seg_pixels, y)."""
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for a, b, c, y in tqdm(loader, desc="Eval", leave=False):
            a = a.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            logits = model(a, b, c)
            probs = torch.sigmoid(logits.squeeze(-1))
            y_score.extend(probs.cpu().numpy().tolist())
            y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_score)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True,
                        choices=["rgb", "depth", "seg", "full", "baseline"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    cfg.setdefault("clip_len", 16)
    cfg.setdefault("anchor_offset_sec", 0.0)
    cfg.setdefault("model_name", "MCG-NJU/videomae-base")
    print(f"Checkpoint: epoch={ckpt.get('epoch')}, stored val_auc={ckpt.get('val_auc', '?'):.4f}")
    print(f"Config: {cfg}")

    # Build dataset
    model_type = args.type
    if model_type == "rgb":
        dataset = build_rgb(cfg)
    elif model_type == "depth":
        dataset = build_depth(cfg)
    elif model_type == "seg":
        dataset = build_seg(cfg)
    elif model_type == "full":
        dataset = build_full(cfg)
    else:
        dataset = build_baseline(cfg)

    # Reconstruct identical val split
    labels = dataset.df["target"].to_numpy(dtype=np.int64)
    idx = np.arange(len(dataset))
    _, val_idx = train_test_split(idx, test_size=VAL_SPLIT, random_state=SEED, stratify=labels)
    clip_ids = dataset.df["id"].iloc[val_idx].to_numpy(dtype=str)

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Val set: {len(val_idx)} samples")

    # Build model and run inference
    if model_type == "rgb":
        from transformers import VideoMAEForVideoClassification
        model = VideoMAEForVideoClassification.from_pretrained(
            cfg["model_name"], num_labels=1, ignore_mismatched_sizes=True,
        )
        model = model.to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        y_true, y_score = infer_single(model, val_loader, device)

    elif model_type == "depth":
        from train_videomae_depth import TwoStreamVideoMAE
        model = TwoStreamVideoMAE(cfg["model_name"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        y_true, y_score = infer_two_stream(model, val_loader, device)

    elif model_type == "seg":
        from train_videomae_seg import TwoStreamVideoMAE
        model = TwoStreamVideoMAE(cfg["model_name"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        y_true, y_score = infer_two_stream(model, val_loader, device)

    elif model_type == "full":
        from train_videomae_full import ThreeStreamVideoMAE
        model = ThreeStreamVideoMAE(cfg["model_name"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        y_true, y_score = infer_three_stream(model, val_loader, device)

    else:  # baseline
        from train_baseline import TinyVideoCNN
        model = TinyVideoCNN().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        y_true, y_score = infer_baseline_model(model, val_loader, device)

    auc = roc_auc_score(y_true, y_score)
    print(f"Recomputed val AUC : {auc:.4f}")
    print(f"Stored val AUC     : {ckpt.get('val_auc', '?'):.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    np.savez(out_path, y_true=y_true, y_score=y_score, clip_ids=clip_ids)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
