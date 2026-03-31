"""
extract_frames.py

Takes a video file, extracts frames at a fixed FPS, saves as JPEGs.

Usage:
    python src/pipeline/extract_frames.py --video data/raw/video_0001.mp4
"""

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path: str, output_dir: str = "data/frames", fps: int = 10):
    """
    Extract frames from a video at a fixed FPS.

    Args:
        video_path: Path to the input video file.
        output_dir: Root directory to save frames.
        fps: Frames per second to extract.
    """
    video_path = Path(video_path)
    video_name = video_path.stem  # e.g. "video_0001"

    # Each video gets its own subfolder
    save_dir = Path(output_dir) / video_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / native_fps

    # How many native frames to skip between each saved frame
    frame_interval = max(1, round(native_fps / fps))

    print(f"Video      : {video_name}")
    print(f"Native FPS : {native_fps:.1f}")
    print(f"Duration   : {duration_sec:.1f}s")
    print(f"Extracting : every {frame_interval} frames → ~{fps} FPS")
    print(f"Saving to  : {save_dir}")

    saved = 0
    native_idx = 0

    with tqdm(total=int(duration_sec * fps), unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if native_idx % frame_interval == 0:
                # Filename encodes the frame index for easy sorting
                filename = save_dir / f"frame_{saved:05d}.jpg"
                cv2.imwrite(str(filename), frame)
                saved += 1
                pbar.update(1)

            native_idx += 1

    cap.release()
    print(f"\nDone. Saved {saved} frames to {save_dir}")
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output_dir", default="data/frames", help="Root output directory")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS to extract")
    args = parser.parse_args()

    extract_frames(args.video, args.output_dir, args.fps)