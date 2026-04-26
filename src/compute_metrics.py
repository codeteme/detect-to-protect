"""
Compute classification metrics from saved prediction .npz files.

Thresholds are F1-optimal by default. Use --threshold to fix one value
across all models, or --recall-target to find the threshold that meets
a minimum recall (useful for safety-critical framing).

Usage (from project root):
    python src/compute_metrics.py                           # all outputs/preds_*.npz
    python src/compute_metrics.py outputs/preds_foo.npz    # specific files
    python src/compute_metrics.py --threshold 0.5          # fixed threshold
    python src/compute_metrics.py --recall-target 0.90     # target 90% recall
    python src/compute_metrics.py --ci                     # add 95% bootstrap CI on AUC
"""

from pathlib import Path
import argparse
import sys

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    precision_recall_curve,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"


def f1_optimal_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    idx = np.argmax(f1)
    return float(thresholds[idx])


def recall_target_threshold(y_true, y_score, target_recall):
    _, recall, thresholds = precision_recall_curve(y_true, y_score)
    # Walk from high-recall end; find first threshold that still meets target
    candidates = np.where(recall[:-1] >= target_recall)[0]
    if len(candidates) == 0:
        return float(thresholds[0])
    return float(thresholds[candidates[-1]])


def bootstrap_auc_ci(y_true, y_score, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compute(y_true, y_score, threshold, ci=False):
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)
    result = dict(auc=auc, precision=precision, recall=recall, f1=f1,
                  threshold=threshold, cm=cm, ci=None)
    if ci:
        result["ci"] = bootstrap_auc_ci(y_true, y_score)
    return result


def print_summary_table(results, threshold_label):
    print(f"\nThreshold selection: {threshold_label}\n")
    has_ci = any(m["ci"] is not None for m in results.values())
    cols   = ["Model",  "AUC",  "95% CI",          "Thresh", "Precision", "Recall", "F1   "]
    widths = [38,        6,      18,                 7,         9,           7,        6]
    if not has_ci:
        cols.pop(2); widths.pop(2)
    header = "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for name, m in sorted(results.items()):
        ci_str = f"({m['ci'][0]:.3f}–{m['ci'][1]:.3f})" if m["ci"] else ""
        row = [name[:38], f"{m['auc']:.3f}"]
        if has_ci:
            row.append(ci_str)
        row += [f"{m['threshold']:.3f}", f"{m['precision']:.3f}",
                f"{m['recall']:.3f}", f"{m['f1']:.3f}"]
        print("  ".join(f"{v:<{w}}" for v, w in zip(row, widths)))
    print()


def print_confusion_matrices(results):
    for name, m in sorted(results.items()):
        tn, fp, fn, tp = m["cm"].ravel()
        total_pos = tp + fn
        total_neg = tn + fp
        far = fp / total_neg * 100 if total_neg > 0 else 0.0
        print(f"  {name}")
        print(f"    {'':30s}  Predicted: No   Predicted: Yes")
        print(f"    Actual: No collision    {tn:>13}   {fp:>13}")
        print(f"    Actual: Collision       {fn:>13}   {tp:>13}")
        print(f"    → Caught {tp}/{total_pos} collisions  ({m['recall']*100:.1f}% recall)")
        print(f"    → {fp}/{total_neg} non-collision clips incorrectly flagged  ({far:.1f}% false alarm rate)")
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="Classification metrics from saved .npz predictions")
    parser.add_argument("files", nargs="*",
                        help=".npz prediction files (default: all outputs/preds_*.npz)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--threshold", type=float, default=None,
                       help="Fixed classification threshold for all models")
    group.add_argument("--recall-target", type=float, default=None,
                       metavar="R",
                       help="Set threshold to achieve at least this recall (e.g. 0.90)")
    parser.add_argument("--ci", action="store_true",
                        help="Compute 95%% bootstrap CI on AUC (2,000 iterations)")
    return parser.parse_args()


def main():
    args = parse_args()

    paths = [Path(f) for f in args.files] if args.files else sorted(OUT_DIR.glob("preds_*.npz"))
    if not paths:
        print("No prediction files found. Run eval_save_preds.py first.")
        sys.exit(1)

    if args.threshold is not None:
        threshold_label = f"fixed at {args.threshold:.2f}"
    elif args.recall_target is not None:
        threshold_label = f"minimum recall ≥ {args.recall_target:.0%} (per model)"
    else:
        threshold_label = "F1-optimal (per model)"

    results = {}
    for path in paths:
        name = path.stem
        if name.startswith("preds_"):
            name = name[len("preds_"):]
        data = np.load(path)
        y_true, y_score = data["y_true"], data["y_score"]

        if args.threshold is not None:
            threshold = args.threshold
        elif args.recall_target is not None:
            threshold = recall_target_threshold(y_true, y_score, args.recall_target)
        else:
            threshold = f1_optimal_threshold(y_true, y_score)

        if args.ci:
            print(f"  {name} — bootstrapping CI...")
        results[name] = compute(y_true, y_score, threshold, ci=args.ci)

    print("\n" + "=" * 65)
    print("CLASSIFICATION METRICS SUMMARY")
    print("=" * 65)
    print_summary_table(results, threshold_label)

    print("=" * 65)
    print("CONFUSION MATRICES  (plain-language breakdown per model)")
    print("=" * 65 + "\n")
    print_confusion_matrices(results)


if __name__ == "__main__":
    main()
