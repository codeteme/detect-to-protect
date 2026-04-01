"""
feature_extractor.py

Runs the full feature extraction pipeline on one video:
  1. YOLO + ByteTrack detection per frame
  2. Depth Anything V2 depth map per frame
  3. Extracts static + temporal feature vector per tracked object per frame
  4. Saves to data/features/<video_id>.parquet

Static features (per frame):
  Geometry : cx, cy, w, h, area, aspect_ratio, ego_lane
  Depth    : depth_mean, depth_min, depth_p5, depth_var

Temporal features (computed over a sliding window of TEMPORAL_WINDOW frames):
  area_growth_rate : how fast the bounding box is growing (looming signal)
  ttc_looming      : physics-based time-to-collision estimate in seconds
  depth_rate       : rate of depth change (negative = object getting closer)
  cx_vel           : lateral velocity of object center
  cy_vel           : vertical velocity of object center
  track_age        : number of frames this track has been continuously seen

Usage:
    python src/pipeline/feature_extractor.py --video_dir data/frames/00007
"""

import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from collections import deque
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline as hf_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Number of frames to look back when computing temporal deltas.
# At 10 FPS this is a 0.5-second window.
TEMPORAL_WINDOW = 5

# FPS of the extracted frames — used to convert frame-based TTC to seconds.
FRAME_RATE = 10.0

# Sentinel value for ttc_looming when the object is not approaching.
TTC_NO_APPROACH = 999.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    yolo = YOLO("configs/yolov8n.pt")

    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    return yolo, depth_pipe


# ---------------------------------------------------------------------------
# Per-frame feature extraction
# ---------------------------------------------------------------------------

def get_depth_map(depth_pipe, frame_bgr):
    """Run depth estimation on a BGR frame. Returns float32 (H, W) in [0, 1]."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    result = depth_pipe(image)
    depth = np.array(result["depth"], dtype=np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


def get_static_features(depth_map, x1, y1, x2, y2, frame_w, frame_h):
    """
    Extract static (single-frame) features for one bounding box.

    Returns a dict with geometry and depth features, all normalized.
    """
    x1i = max(0, int(x1))
    y1i = max(0, int(y1))
    x2i = min(frame_w, int(x2))
    y2i = min(frame_h, int(y2))

    # Normalized bounding box geometry
    cx = ((x1 + x2) / 2) / frame_w
    cy = ((y1 + y2) / 2) / frame_h
    w  = (x2 - x1) / frame_w
    h  = (y2 - y1) / frame_h
    area = w * h
    aspect_ratio = w / (h + 1e-8)

    # Is the object in the ego lane? (center 30% of frame width)
    ego_lane = 1.0 if 0.35 <= cx <= 0.65 else 0.0

    # Depth statistics within the bounding box
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


# ---------------------------------------------------------------------------
# Temporal feature computation
# ---------------------------------------------------------------------------

def get_temporal_features(track_id, current_frame, track_history):
    """
    Compute temporal features for a track by comparing the current frame
    snapshot against the snapshot TEMPORAL_WINDOW frames ago.

    track_history: dict mapping track_id -> deque of past frame snapshots.
    Each snapshot is a dict with keys: area, depth_mean, cx, cy.

    Returns a dict of temporal features.
    If the track has fewer than TEMPORAL_WINDOW past frames, returns zeros
    (not enough history yet).
    """
    history = track_history.get(track_id, deque())

    # Not enough history — return neutral zero values
    if len(history) < TEMPORAL_WINDOW:
        return {
            "area_growth_rate": 0.0,
            "ttc_looming":      TTC_NO_APPROACH,
            "depth_rate":       0.0,
            "cx_vel":           0.0,
            "cy_vel":           0.0,
            "track_age":        len(history),
        }

    past_frame = history[-TEMPORAL_WINDOW]  # snapshot from N frames ago

    # Rate of change per frame (divide by window size to normalize)
    area_growth_rate = (current_frame["area"] - past_frame["area"]) / TEMPORAL_WINDOW
    depth_rate       = (current_frame["depth_mean"] - past_frame["depth_mean"]) / TEMPORAL_WINDOW
    cx_vel           = (current_frame["cx"] - past_frame["cx"]) / TEMPORAL_WINDOW
    cy_vel           = (current_frame["cy"] - past_frame["cy"]) / TEMPORAL_WINDOW

    # Physics-based TTC: how many seconds until the bbox fills the frame?
    # Derived from the optical looming formula: TTC = area / (d_area/dt)
    # Only meaningful when the object is growing (approaching).
    if area_growth_rate > 0:
        ttc_in_frames = current_frame["area"] / area_growth_rate
        ttc_looming   = ttc_in_frames / FRAME_RATE
    else:
        ttc_looming = TTC_NO_APPROACH  # object is not approaching

    return {
        "area_growth_rate": area_growth_rate,
        "ttc_looming":      ttc_looming,
        "depth_rate":       depth_rate,
        "cx_vel":           cx_vel,
        "cy_vel":           cy_vel,
        "track_age":        len(history),
    }


def update_track_history(track_id, snapshot, track_history):
    """
    Append the current frame snapshot to a track's history buffer.
    Each buffer is capped at TEMPORAL_WINDOW entries (oldest dropped automatically).

    snapshot: dict with keys area, depth_mean, cx, cy.
    """
    if track_id not in track_history:
        track_history[track_id] = deque(maxlen=TEMPORAL_WINDOW)
    track_history[track_id].append(snapshot)


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_features(video_dir: str, output_dir: str = "data/features"):
    video_dir  = Path(video_dir)
    video_id   = video_dir.name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path   = output_dir / f"{video_id}.parquet"

    frame_paths = sorted(video_dir.glob("*.jpg"))
    if not frame_paths:
        raise ValueError(f"No frames found in {video_dir}")

    print(f"\nVideo: {video_id} | Frames: {len(frame_paths)}")
    yolo, depth_pipe = load_models()

    # Stores the last TEMPORAL_WINDOW snapshots for each track ID
    track_history = {}

    rows = []

    for frame_idx, frame_path in enumerate(tqdm(frame_paths)):
        frame_bgr = cv2.imread(str(frame_path))
        frame_h, frame_w = frame_bgr.shape[:2]

        depth_map = get_depth_map(depth_pipe, frame_bgr)

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

            static = get_static_features(depth_map, x1, y1, x2, y2, frame_w, frame_h)

            # Snapshot of the fields needed for temporal computation
            snapshot = {
                "area":       static["area"],
                "depth_mean": static["depth_mean"],
                "cx":         static["cx"],
                "cy":         static["cy"],
            }

            temporal = get_temporal_features(track_id, snapshot, track_history)

            # Update history AFTER computing temporal features so we don't
            # compare the current frame against itself
            update_track_history(track_id, snapshot, track_history)

            rows.append({
                "video_id":   video_id,
                "frame_idx":  frame_idx,
                "frame_name": frame_path.stem,
                "track_id":   track_id,
                "class_id":   class_id,
                "conf":       conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                **static,
                **temporal,
                # Labels — filled in later by label_builder.py
                "collision_label": -1,
                "ttc_seconds":     -1.0,
            })

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)

    print(f"\nSaved {len(df)} rows → {out_path}")
    print(df[["frame_idx", "track_id", "area_growth_rate", "ttc_looming", "depth_rate"]].head(10))
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", default="data/features")
    args = parser.parse_args()

    extract_features(args.video_dir, args.output_dir)
