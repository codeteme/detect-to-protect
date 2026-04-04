import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NexarDataset(Dataset):
    """
    Loads preprocessed .npy files (frames, segmentation, depth) for each video
    and assembles them into a [T, 5, H, W] tensor.

    Args:
        csv_path:    path to train.csv or test.csv
        frames_dir:  directory containing per-video frames .npy files
        seg_dir:     directory containing per-video segmentation .npy files
        depth_dir:   directory containing per-video depth .npy files
        split:       'train' or 'test'
        fps:         frames per second used during preprocessing (default 10)
        clip_len:    number of frames to use per video (default 100 = 10s @ 10fps)
    """

    def __init__(self, csv_path, frames_dir, seg_dir, depth_dir,
                 split='train', fps=10, clip_len=100):
        self.frames_dir = frames_dir
        self.seg_dir = seg_dir
        self.depth_dir = depth_dir
        self.split = split
        self.fps = fps
        self.clip_len = clip_len

        df = pd.read_csv(csv_path)
        df['id'] = df['id'].astype(str).str.zfill(5)
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row['id']

        # --- Step 2: load .npy files ---
        frames = np.load(os.path.join(self.frames_dir, f"{video_id}.npy"))   # [N, H, W, 3] uint8
        seg    = np.load(os.path.join(self.seg_dir,    f"{video_id}.npy"))   # [N, H, W]    uint8
        depth  = np.load(os.path.join(self.depth_dir,  f"{video_id}.npy"))   # [N, H, W]    float16

        # --- Step 3: trim to clip_len frames ---
        N = len(frames)

        if self.split == 'train':
            toe = row.get('time_of_event')
            if pd.notna(toe):
                # Positive: anchor to time_of_event
                end   = min(int(toe * self.fps), N)
                start = max(end - self.clip_len, 0)
            else:
                # Negative: take last clip_len frames
                start = max(N - self.clip_len, 0)
                end   = N
        else:
            # Test: already ~10s, take all
            start, end = 0, N

        frames = frames[start:end]
        seg    = seg[start:end]
        depth  = depth[start:end]

        # Pad to clip_len if shorter (e.g. video shorter than 10s)
        T = len(frames)
        if T < self.clip_len:
            pad = self.clip_len - T
            frames = np.concatenate([np.zeros((pad, *frames.shape[1:]), dtype=frames.dtype), frames], axis=0)
            seg    = np.concatenate([np.zeros((pad, *seg.shape[1:]),    dtype=seg.dtype),    seg],    axis=0)
            depth  = np.concatenate([np.zeros((pad, *depth.shape[1:]),  dtype=depth.dtype),  depth],  axis=0)

        # --- Step 4: assemble [T, 5, H, W] tensor ---
        # frames: [T, H, W, 3] uint8  → [T, 3, H, W] float32 in [0, 1]
        # seg:    [T, H, W]    uint8  → [T, 1, H, W] float32 in {0, 1}
        # depth:  [T, H, W]    float16 → [T, 1, H, W] float32 in [0, 1]
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        seg_t    = torch.from_numpy(seg.astype(np.float32)).unsqueeze(1)
        depth_t  = torch.from_numpy(depth.astype(np.float32)).unsqueeze(1)

        video = torch.cat([frames_t, depth_t, seg_t], dim=1)  # [T, 5, H, W]

        if self.split == 'test':
            return video, video_id

        label = torch.tensor(row['target'], dtype=torch.float32)
        return video, label
