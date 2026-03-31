"""
depth.py

Runs Depth Anything V2 on a single frame and returns a depth map.

Usage:
    python src/pipeline/depth.py --frame data/frames/00001/frame_00150.jpg
"""

import numpy as np
import cv2
import argparse
import torch
from PIL import Image
from transformers import pipeline as hf_pipeline


def load_depth_model():
    """Load Depth Anything V2 Small. Downloads ~100MB on first run."""
    print("Loading Depth Anything V2 Small...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    return pipe


def estimate_depth(pipe, frame_path: str) -> np.ndarray:
    """
    Run depth estimation on a single frame.

    Returns:
        depth_map: float32 numpy array of shape (H, W), values in [0, 1]
                   Higher value = farther away (relative, not metric)
    """
    image = Image.open(frame_path).convert("RGB")
    result = pipe(image)

    # Convert PIL depth image to numpy float32, normalize to [0,1]
    depth = np.array(result["depth"], dtype=np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


def extract_bbox_depth_features(depth_map: np.ndarray, bbox: tuple) -> dict:
    """
    Extract depth statistics from within a bounding box.

    Args:
        depth_map: (H, W) float32 array
        bbox: (x1, y1, x2, y2) in pixels

    Returns:
        dict of depth features for this object
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Clamp to image bounds
    h, w = depth_map.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    crop = depth_map[y1:y2, x1:x2]

    if crop.size == 0:
        return {"depth_mean": 0, "depth_min": 0, "depth_p5": 0, "depth_var": 0}

    return {
        "depth_mean": float(crop.mean()),
        "depth_min":  float(crop.min()),
        "depth_p5":   float(np.percentile(crop, 5)),   # closest point in bbox
        "depth_var":  float(crop.var()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True, help="Path to a single frame JPEG")
    args = parser.parse_args()

    pipe = load_depth_model()
    depth_map = estimate_depth(pipe, args.frame)

    print(f"\nDepth map shape : {depth_map.shape}")
    print(f"Depth map range : min={depth_map.min():.3f}, max={depth_map.max():.3f}")

    # Test: extract features from a fake bbox in the center of the frame
    h, w = depth_map.shape
    test_bbox = (w//4, h//4, 3*w//4, 3*h//4)
    features = extract_bbox_depth_features(depth_map, test_bbox)
    print(f"\nDepth features for center bbox {test_bbox}:")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")

    # Save a visualization so you can actually see the depth map
    depth_vis = (depth_map * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    out_path = "depth_preview.jpg"
    cv2.imwrite(out_path, depth_colored)
    print(f"\nDepth visualization saved to: {out_path}")
    print("Bright = far, Dark = close")