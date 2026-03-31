"""
label_builder.py

Joins train.csv event timestamps to extracted feature parquet files.
Adds collision_label and ttc_seconds columns to each parquet.

Usage:
    python src/pipeline/label_builder.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# How many seconds before event = "danger zone" (collision_label = 1)
DANGER_WINDOW_SEC = 3.0
FPS = 10  # must match extract_frames.py


def build_labels(
    features_dir: str = "data/features",
    labels_csv:   str = "data/train.csv",
    output_dir:   str = "data/features",
):
    features_dir = Path(features_dir)
    output_dir   = Path(output_dir)
    labels_df    = pd.read_csv(labels_csv)

    # Normalize id to string with leading zeros
    labels_df["id"] = labels_df["id"].astype(str).str.zfill(5)

    parquet_files = sorted(features_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {features_dir}")

    all_dfs = []

    for parquet_path in parquet_files:
        video_id = parquet_path.stem  # e.g. "00007"
        df = pd.read_parquet(parquet_path)

        # Match to train.csv
        row = labels_df[labels_df["id"] == video_id]
        if row.empty:
            print(f"[SKIP] {video_id} not found in train.csv")
            continue

        target         = int(row["target"].values[0])
        time_of_event  = row["time_of_event"].values[0]
        time_of_alert  = row["time_of_alert"].values[0]

        # Compute frame time in seconds
        df["frame_time_sec"] = df["frame_idx"] / FPS

        if target == 1 and not pd.isna(time_of_event):
            time_of_event = float(time_of_event)
            time_of_alert = float(time_of_alert)

            # TTC = time remaining until collision event
            df["ttc_seconds"] = time_of_event - df["frame_time_sec"]

            # Collision label: 1 if within danger window AND approaching
            df["collision_label"] = (
                (df["ttc_seconds"] >= 0) &
                (df["ttc_seconds"] <= DANGER_WINDOW_SEC)
            ).astype(int)

            df["time_of_event"] = time_of_event
            df["time_of_alert"] = time_of_alert

        else:
            df["ttc_seconds"]    = -1.0
            df["collision_label"] = 0
            df["time_of_event"]  = np.nan
            df["time_of_alert"]  = np.nan

        df["target"] = target

        # Save labeled parquet
        out_path = output_dir / f"{video_id}_labeled.parquet"
        df.to_parquet(out_path, index=False)

        # Summary
        n_positive = df["collision_label"].sum()
        n_total    = len(df)
        print(f"\n{video_id} | target={target} | event={time_of_event}s")
        print(f"  Total rows     : {n_total}")
        print(f"  Positive frames: {n_positive} ({100*n_positive/n_total:.1f}%)")
        print(f"  Negative frames: {n_total - n_positive}")
        if target == 1:
            # Show TTC range for positive frames
            pos = df[df["collision_label"] == 1]
            print(f"  TTC range      : {pos['ttc_seconds'].min():.2f}s → {pos['ttc_seconds'].max():.2f}s")

        all_dfs.append(df)

    # Combine all into one master dataset
    master = pd.concat(all_dfs, ignore_index=True)
    master_path = output_dir / "master_labeled.parquet"
    master.to_parquet(master_path, index=False)

    print(f"\n{'='*50}")
    print(f"Master dataset: {len(master)} rows")
    print(f"Total positive: {master['collision_label'].sum()}")
    print(f"Total negative: {(master['collision_label'] == 0).sum()}")
    print(f"Class ratio   : 1:{(master['collision_label']==0).sum() // max(master['collision_label'].sum(),1)}")
    print(f"Saved → {master_path}")

    return master


if __name__ == "__main__":
    build_labels()