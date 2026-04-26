"""
Error analysis: identify which validation clips the model fails on,
and whether failures cluster around lighting/night conditions.

Brightness is estimated from the mean pixel value of the last frame in the
RGB clip window — a simple, cheap proxy for day (bright) vs night (dark).

Usage (from project root):
    # Best model, reconstruct val split automatically
    python src/error_analysis.py \
        --preds outputs/preds_videomae_full_ofs0p0.npz

    # Fix a threshold (default: F1-optimal)
    python src/error_analysis.py \
        --preds outputs/preds_videomae_full_ofs0p0.npz \
        --threshold 0.5

    # Save per-clip CSV for further inspection
    python src/error_analysis.py \
        --preds outputs/preds_videomae_full_ofs0p0.npz \
        --save-csv outputs/error_analysis_full.csv

    # Save thumbnail images of the worst failures
    python src/error_analysis.py \
        --preds outputs/preds_videomae_full_ofs0p0.npz \
        --save-thumbnails outputs/thumbnails
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FRAMES_DIR = DATA_DIR / "frames" / "train"
SEED = 42
VAL_SPLIT = 0.2

# Clips with mean last-frame brightness below this (0–255) are "dark"
DARK_THRESHOLD = 60


def f1_optimal_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    return float(thresholds[np.argmax(f1)])


def get_clip_brightness(clip_id: str, frames_dir: Path, clip_len: int = 16) -> float:
    """Mean pixel brightness (0–255) of the last frame in the clip window."""
    npy_path = frames_dir / f"{clip_id}.npy"
    if not npy_path.exists():
        return float("nan")
    frames = np.load(npy_path)  # shape: (N, H, W, 3), uint8
    if frames.ndim != 4 or frames.shape[0] == 0:
        return float("nan")
    last_frame = frames[min(clip_len, frames.shape[0]) - 1]  # (H, W, 3)
    return float(last_frame.mean())


def reconstruct_clip_ids(csv_path: Path, frames_dir: Path) -> np.ndarray:
    """Reproduce the exact val split from training and return ordered clip IDs."""
    df = pd.read_csv(csv_path)
    df["id"] = df["id"].astype(str).str.zfill(5)
    available = {p.stem for p in frames_dir.glob("*.npy")}
    df = df[df["id"].isin(available)].reset_index(drop=True)
    labels = df["target"].to_numpy(dtype=np.int64)
    idx = np.arange(len(df))
    _, val_idx = train_test_split(idx, test_size=VAL_SPLIT, random_state=SEED, stratify=labels)
    return df["id"].iloc[val_idx].to_numpy(dtype=str)


def print_group_metrics(label: str, mask: np.ndarray,
                         y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray):
    yt, yp, ys = y_true[mask], y_pred[mask], y_score[mask]
    n = mask.sum()
    if n == 0:
        print(f"  {label}: no clips")
        return
    tp = ((yp == 1) & (yt == 1)).sum()
    fp = ((yp == 1) & (yt == 0)).sum()
    fn = ((yp == 0) & (yt == 1)).sum()
    tn = ((yp == 0) & (yt == 0)).sum()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    try:
        auc = roc_auc_score(yt, ys) if len(np.unique(yt)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    print(f"  {label} (n={n}): AUC={auc:.3f}  Recall={recall:.1%}  FalseAlarmRate={far:.1%}"
          f"  [TP={tp} FP={fp} FN={fn} TN={tn}]")


def save_thumbnails(df_errors: pd.DataFrame, frames_dir: Path, out_dir: Path,
                     error_type: str, n: int = 10):
    """Save the last frame of the n worst errors of a given type as PNG."""
    try:
        from PIL import Image
    except ImportError:
        print("  (Pillow not installed — skipping thumbnail export)")
        return

    subset = df_errors[df_errors["error_type"] == error_type].copy()
    if len(subset) == 0:
        return
    # Worst = highest model confidence in the wrong direction
    if error_type == "FN":
        subset = subset.nlargest(n, "y_score")   # confident but missed
    else:
        subset = subset.nlargest(n, "y_score")   # highest false-alarm score

    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in subset.iterrows():
        npy_path = frames_dir / f"{row['clip_id']}.npy"
        if not npy_path.exists():
            continue
        frames = np.load(npy_path)
        last_frame = frames[-1] if frames.ndim == 4 else frames
        img = Image.fromarray(last_frame.astype(np.uint8))
        fname = f"{error_type}_{row['clip_id']}_score{row['y_score']:.2f}.png"
        img.save(out_dir / fname)
    print(f"  Saved {min(n, len(subset))} {error_type} thumbnails → {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True,
                   help="Path to .npz from eval_save_preds.py")
    p.add_argument("--threshold", type=float, default=None,
                   help="Classification threshold (default: F1-optimal)")
    p.add_argument("--dark-threshold", type=float, default=DARK_THRESHOLD,
                   help=f"Mean brightness below which a clip is 'dark' (default: {DARK_THRESHOLD})")
    p.add_argument("--save-csv", default=None,
                   help="Save per-clip results to this CSV path")
    p.add_argument("--save-thumbnails", default=None,
                   help="Save worst-failure frame thumbnails to this directory")
    p.add_argument("--top-n", type=int, default=10,
                   help="Number of worst failures to show/save (default: 10)")
    return p.parse_args()


def main():
    args = parse_args()

    data = np.load(args.preds, allow_pickle=True)
    y_true = data["y_true"]
    y_score = data["y_score"]

    # Get clip IDs: from .npz if present (new format), else reconstruct
    if "clip_ids" in data:
        clip_ids = data["clip_ids"].astype(str)
        print("Loaded clip IDs from .npz")
    else:
        print("clip_ids not in .npz — reconstructing val split from train.csv ...")
        clip_ids = reconstruct_clip_ids(DATA_DIR / "train.csv", FRAMES_DIR)
        if len(clip_ids) != len(y_true):
            print(f"ERROR: reconstructed {len(clip_ids)} IDs but .npz has {len(y_true)} rows.")
            sys.exit(1)
        print(f"Reconstructed {len(clip_ids)} val clip IDs")

    threshold = args.threshold if args.threshold is not None else f1_optimal_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    print(f"\nThreshold: {threshold:.3f}  (AUC={roc_auc_score(y_true, y_score):.4f})")

    # -------------------------------------------------------------------------
    # Compute per-clip brightness
    # -------------------------------------------------------------------------
    print(f"\nComputing brightness for {len(clip_ids)} clips ...")
    brightness = np.array([get_clip_brightness(cid, FRAMES_DIR) for cid in clip_ids])
    valid = ~np.isnan(brightness)
    print(f"  {valid.sum()} clips with brightness data  ({(~valid).sum()} missing frame files)")

    dark_mask = valid & (brightness < args.dark_threshold)
    bright_mask = valid & (brightness >= args.dark_threshold)

    # -------------------------------------------------------------------------
    # Error type per clip
    # -------------------------------------------------------------------------
    error_type = np.where(
        (y_pred == 1) & (y_true == 1), "TP",
        np.where((y_pred == 0) & (y_true == 0), "TN",
        np.where((y_pred == 1) & (y_true == 0), "FP", "FN"))
    )

    # -------------------------------------------------------------------------
    # Summary by lighting condition
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("METRICS BY LIGHTING CONDITION")
    print(f"  Dark threshold: mean brightness < {args.dark_threshold:.0f}")
    print(f"{'='*60}")
    print_group_metrics("All clips    ", valid,        y_true, y_pred, y_score)
    print_group_metrics("Bright (day) ", bright_mask,  y_true, y_pred, y_score)
    print_group_metrics("Dark (night) ", dark_mask,    y_true, y_pred, y_score)

    # -------------------------------------------------------------------------
    # Brightness distribution among errors
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("BRIGHTNESS DISTRIBUTION BY ERROR TYPE")
    print(f"{'='*60}")
    for et in ["TP", "TN", "FP", "FN"]:
        mask = valid & (error_type == et)
        if mask.sum() == 0:
            continue
        b = brightness[mask]
        dark_frac = (b < args.dark_threshold).mean()
        print(f"  {et} (n={mask.sum():3d}): mean_brightness={b.mean():.1f}  "
              f"dark_frac={dark_frac:.1%}  [min={b.min():.1f} max={b.max():.1f}]")

    # -------------------------------------------------------------------------
    # Worst failures
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"TOP {args.top_n} WORST FALSE NEGATIVES (missed collisions, highest model score first)")
    print(f"{'='*60}")
    fn_mask = error_type == "FN"
    fn_df = pd.DataFrame({
        "clip_id": clip_ids[fn_mask],
        "y_score": y_score[fn_mask],
        "brightness": brightness[fn_mask],
        "dark": brightness[fn_mask] < args.dark_threshold,
    }).sort_values("y_score", ascending=False).head(args.top_n)
    print(fn_df.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"TOP {args.top_n} WORST FALSE POSITIVES (false alarms, highest model score first)")
    print(f"{'='*60}")
    fp_mask = error_type == "FP"
    fp_df = pd.DataFrame({
        "clip_id": clip_ids[fp_mask],
        "y_score": y_score[fp_mask],
        "brightness": brightness[fp_mask],
        "dark": brightness[fp_mask] < args.dark_threshold,
    }).sort_values("y_score", ascending=False).head(args.top_n)
    print(fp_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # Optional CSV
    # -------------------------------------------------------------------------
    if args.save_csv:
        out_df = pd.DataFrame({
            "clip_id": clip_ids,
            "y_true": y_true,
            "y_score": y_score,
            "y_pred": y_pred,
            "error_type": error_type,
            "brightness": brightness,
            "dark": brightness < args.dark_threshold,
        })
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.save_csv, index=False)
        print(f"\nPer-clip CSV → {args.save_csv}")

    # -------------------------------------------------------------------------
    # Optional thumbnails
    # -------------------------------------------------------------------------
    if args.save_thumbnails:
        thumb_dir = Path(args.save_thumbnails)
        all_errors = pd.DataFrame({
            "clip_id": clip_ids,
            "y_score": y_score,
            "error_type": error_type,
            "brightness": brightness,
        })
        print()
        save_thumbnails(all_errors, FRAMES_DIR, thumb_dir, "FN", args.top_n)
        save_thumbnails(all_errors, FRAMES_DIR, thumb_dir, "FP", args.top_n)


if __name__ == "__main__":
    main()
