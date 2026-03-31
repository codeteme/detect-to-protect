"""
feature_extractor.py

Runs the full feature extraction pipeline on one video:
  1. YOLO + ByteTrack detection per frame
  2. Depth Anything V2 depth map per frame
  3. Extracts feature vector per tracked object per frame
  4. Saves to data/features/<video_id>.parquet

Usage:
    python src/pipeline/feature_extractor.py --video_dir data/frames/00007
"""

import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline as hf_pipeline

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def load_models():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    yolo = YOLO("configs/yolov8n.pt")

    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    return yolo, depth_pipe


def get_depth_map(depth_pipe, frame_bgr):
    """Run depth estimation on a BGR frame. Returns float32 (H,W) in [0,1]."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    result = depth_pipe(image)
    depth = np.array(result["depth"], dtype=np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


def get_bbox_features(depth_map, x1, y1, x2, y2, frame_w, frame_h):
    """Extract the 14-feature vector for one bounding box."""
    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
    x1i = max(0, x1i); y1i = max(0, y1i)
    x2i = min(frame_w, x2i); y2i = min(frame_h, y2i)

    # Bounding box geometry (normalized)
    cx = ((x1 + x2) / 2) / frame_w
    cy = ((y1 + y2) / 2) / frame_h
    w  = (x2 - x1) / frame_w
    h  = (y2 - y1) / frame_h
    area = w * h
    aspect_ratio = w / (h + 1e-8)

    # Is the object in the ego lane? (center 30% of frame width)
    ego_lane = 1.0 if 0.35 <= cx <= 0.65 else 0.0

    # Depth features
    crop = depth_map[y1i:y2i, x1i:x2i]
    if crop.size == 0:
        depth_mean = depth_min = depth_p5 = depth_var = 0.0
    else:
        depth_mean = float(crop.mean())
        depth_min  = float(crop.min())
        depth_p5   = float(np.percentile(crop, 5))
        depth_var  = float(crop.var())

    return {
        "cx": cx, "cy": cy, "w": w, "h": h,
        "area": area, "aspect_ratio": aspect_ratio,
        "ego_lane": ego_lane,
        "depth_mean": depth_mean, "depth_min": depth_min,
        "depth_p5": depth_p5, "depth_var": depth_var,
    }


def extract_features(video_dir: str, output_dir: str = "data/features"):
    video_dir  = Path(video_dir)
    video_id   = video_dir.name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path   = output_dir / f"{video_id}.parquet"

    frame_paths = sorted(video_dir.glob("*.jpg"))
    if not frame_paths:
        raise ValueError(f"No frames in {video_dir}")

    print(f"\nVideo: {video_id} | Frames: {len(frame_paths)}")
    yolo, depth_pipe = load_models()

    rows = []

    for frame_idx, frame_path in enumerate(tqdm(frame_paths)):
        frame_bgr = cv2.imread(str(frame_path))
        frame_h, frame_w = frame_bgr.shape[:2]

        # --- Depth ---
        depth_map = get_depth_map(depth_pipe, frame_bgr)

        # --- Detection + Tracking ---
        results = yolo.track(
            frame_bgr,
            persist=True,
            tracker="configs/bytetrack.yaml",
            classes=list(VEHICLE_CLASSES.keys()),
            conf=0.15,
            iou=0.5,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            conf = float(box.conf)
            if conf < 0.15:
                continue

            track_id = int(box.id) if box.id is not None else -1
            class_id = int(box.cls)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            feats = get_bbox_features(
                depth_map, x1, y1, x2, y2, frame_w, frame_h
            )

            rows.append({
                "video_id":   video_id,
                "frame_idx":  frame_idx,
                "frame_name": frame_path.stem,
                "track_id":   track_id,
                "class_id":   class_id,
                "conf":       conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                **feats,
                # Labels — to be filled in later from CSV
                "collision_label": -1,
                "ttc_seconds":     -1.0,
            })

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)

    print(f"\nSaved {len(df)} rows → {out_path}")
    print(df[["frame_idx", "track_id", "depth_mean", "depth_p5", "ego_lane"]].head(10))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", default="data/features")
    args = parser.parse_args()

    extract_features(args.video_dir, args.output_dir)