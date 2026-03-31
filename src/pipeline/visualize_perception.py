"""
visualize_perception.py

Creates a side-by-side visualization video for one extracted video folder:
  1) YOLOv8 + ByteTrack detections with track IDs
  2) Depth Anything V2 heatmap

Usage:
    python src/pipeline/visualize_perception.py --video_dir data/frames/00003
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline as hf_pipeline
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def load_models():
    """Load YOLO detector/tracker and depth estimator once."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    yolo = YOLO("configs/yolov8n.pt")
    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    return yolo, depth_pipe


def get_depth_map(depth_pipe, frame_bgr: np.ndarray) -> np.ndarray:
    """Return normalized float32 depth map in [0, 1]."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    result = depth_pipe(image)
    depth = np.array(result["depth"], dtype=np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


def draw_tracks(frame_bgr: np.ndarray, boxes) -> np.ndarray:
    """Draw YOLO + ByteTrack detections and IDs on the frame."""
    vis = frame_bgr.copy()

    if boxes is None or len(boxes) == 0:
        cv2.putText(
            vis,
            "No vehicle detections",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    for box in boxes:
        conf = float(box.conf)
        if conf < 0.15:
            continue

        class_id = int(box.cls)
        if class_id not in VEHICLE_CLASSES:
            continue

        track_id = int(box.id) if box.id is not None else -1
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 230, 80), 2)

        label = f"{VEHICLE_CLASSES[class_id]} | ID {track_id} | {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_top = max(0, y1 - th - 10)
        cv2.rectangle(vis, (x1, y_top), (x1 + tw + 8, y1), (80, 230, 80), -1)
        cv2.putText(
            vis,
            label,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return vis


def make_panel(frame_bgr: np.ndarray, tracked_bgr: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    """Create a 2x1 panel: tracked detections and depth heatmap."""
    h, w = frame_bgr.shape[:2]

    depth_vis = (depth_map * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    depth_color = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_LINEAR)

    cv2.putText(
        tracked_bgr,
        "YOLOv8 + ByteTrack",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        depth_color,
        "Depth Anything V2 (bright = far)",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    panel = np.hstack([tracked_bgr, depth_color])
    return panel


def visualize_video(video_dir: str, output_path: str = "outputs/perception_preview.mp4", fps: int = 10):
    video_dir = Path(video_dir)
    frame_paths = sorted(video_dir.glob("*.jpg"))
    if not frame_paths:
        raise ValueError(f"No frames found in {video_dir}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yolo, depth_pipe = load_models()

    sample = cv2.imread(str(frame_paths[0]))
    h, w = sample.shape[:2]
    out_w = w * 2
    out_h = h

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    print(f"Frames: {len(frame_paths)}")
    print(f"Writing: {output_path}")

    for frame_path in tqdm(frame_paths):
        frame = cv2.imread(str(frame_path))

        results = yolo.track(
            frame,
            persist=True,
            tracker="configs/bytetrack.yaml",
            classes=list(VEHICLE_CLASSES.keys()),
            conf=0.15,
            iou=0.5,
            verbose=False,
        )
        boxes = results[0].boxes

        depth_map = get_depth_map(depth_pipe, frame)
        tracked = draw_tracks(frame, boxes)
        panel = make_panel(frame, tracked, depth_map)

        writer.write(panel)

    writer.release()
    print("Done.")
    print(f"Saved visualization video: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="Path to extracted frames folder, e.g. data/frames/00003")
    parser.add_argument("--output", default="outputs/perception_preview.mp4", help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    args = parser.parse_args()

    visualize_video(args.video_dir, args.output, args.fps)
