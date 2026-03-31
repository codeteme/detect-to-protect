"""
detect.py

Runs YOLOv8 vehicle detection on extracted frames from one video.

Usage:
    python src/pipeline/detect.py --video_dir data/frames/00003
"""

import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# COCO class IDs for vehicles only
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def detect_video(video_dir: str):
    """
    Run YOLO detection on all frames in a directory.

    Args:
        video_dir: Path to folder of extracted frames (e.g. data/frames/00003)
    """
    video_dir = Path(video_dir)
    frame_paths = sorted(video_dir.glob("*.jpg"))

    if not frame_paths:
        raise ValueError(f"No frames found in {video_dir}")

    print(f"Loading YOLOv8n...")
    model = YOLO("configs/yolov8n.pt")  # downloads ~6MB on first run

    print(f"Running detection on {len(frame_paths)} frames...\n")

    for frame_path in tqdm(frame_paths):
        frame = cv2.imread(str(frame_path))

        # Run detection + ByteTrack tracking
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(VEHICLE_CLASSES.keys()),
            conf=0.15,          # ADD THIS — lower detection threshold
            iou=0.5,            # ADD THIS — standard NMS threshold
            verbose=False,
        )

        detections = results[0].boxes
        if detections is None or len(detections) == 0:
            continue

        for box in detections:
            track_id = int(box.id) if box.id is not None else -1
            class_id = int(box.cls)
            confidence = float(box.conf)
            if confidence < 0.15:        # ADD THIS — skip very weak detections
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(
                f"Frame {frame_path.stem} | "
                f"Track {track_id:3d} | "
                f"{VEHICLE_CLASSES[class_id]:12s} | "
                f"conf={confidence:.2f} | "
                f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="Path to extracted frames folder")
    args = parser.parse_args()

    detect_video(args.video_dir)