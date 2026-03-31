"""
train_classifier.py

Trains XGBoost collision classifier on labeled features.
Runs ablation: Config A (bbox only) vs B (+ depth) vs C (+ depth + seg).
Since we have no segmentation yet, we run A and B.

Usage:
    python src/pipeline/train_classifier.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import xgboost as xgb

# Feature sets for ablation
FEATURES_A = [
    "cx", "cy", "w", "h", "area", "aspect_ratio", "ego_lane"
]
FEATURES_B = FEATURES_A + [
    "depth_mean", "depth_min", "depth_p5", "depth_var"
]

LABEL_COL = "collision_label"
N_SPLITS  = 5


def evaluate(y_true, y_pred, y_prob, label=""):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # FNR at TTC < 2s — most safety-critical metric
    print(f"\n  {'Config':>8}: {label}")
    print(f"  Precision : {precision_score(y_true, y_pred):.3f}")
    print(f"  Recall    : {recall_score(y_true, y_pred):.3f}")
    print(f"  F1        : {f1_score(y_true, y_pred):.3f}")
    print(f"  AUC-ROC   : {roc_auc_score(y_true, y_prob):.3f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")


def run_ablation(df: pd.DataFrame):
    y = df[LABEL_COL].values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Class imbalance ratio
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    scale_pos_weight = neg / pos
    print(f"\nClass ratio — neg:{neg} pos:{pos} scale_pos_weight={scale_pos_weight:.1f}")

    for config_name, features in [("A_bbox_only", FEATURES_A),
                                   ("B_bbox_depth", FEATURES_B)]:
        X = df[features].values
        all_preds = np.zeros(len(y))
        all_probs = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            model.fit(X_train, y_train)

            all_probs[val_idx] = model.predict_proba(X_val)[:, 1]
            all_preds[val_idx] = (all_probs[val_idx] >= 0.3).astype(int)

        evaluate(y, all_preds, all_probs, label=config_name)

        # Safety-critical: FNR at TTC < 2s
        if "ttc_seconds" in df.columns:
            critical = df["ttc_seconds"].between(0, 2)
            if critical.sum() > 0:
                fn_critical = ((all_preds[critical] == 0) & (y[critical] == 1)).sum()
                fnr_critical = fn_critical / y[critical].sum()
                print(f"  FNR@TTC<2s: {fnr_critical:.3f}  ({fn_critical}/{y[critical].sum()} missed)")


def main():
    master_path = Path("data/features/master_labeled.parquet")
    if not master_path.exists():
        raise FileNotFoundError("Run label_builder.py first")

    df = pd.read_parquet(master_path)
    print(f"Loaded {len(df)} rows from master dataset")

    # Drop rows with missing features
    all_features = FEATURES_B
    df = df.dropna(subset=all_features + [LABEL_COL])
    print(f"After dropna: {len(df)} rows")

    run_ablation(df)


if __name__ == "__main__":
    main()