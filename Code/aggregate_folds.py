"""
aggregate_folds.py

Reads the per-fold test_results.json files written by test_phase2.py,
combines them into a single CV summary table, and reports:
    - Mean ± std of mIoU across the 5 folds
    - Per-class IoU averaged across folds where that class was present in test
    - Per-fold overall accuracy

Usage:
    python aggregate_folds.py \\
        --pattern "checkpoints_phase2/PointNet2_from_*_fold*_<comment>" \\
        --classes 17 \\
        --data_root "D:/.../Machining_Tools/train"

The --pattern argument is a glob: it should match all 5 of one
experiment's fold directories.  Example for a linear-probe experiment
of PointNet2 transferred from MFCAD++:
    "checkpoints_phase2/PointNet2_from_PointNet2_MFCAD++_*_frozen_fold*_*"
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def load_class_names(data_root):
    root = Path(data_root)
    for f in root.glob("_global_mapping_*.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and "name_to_id" in d:
            return {int(v): k for k, v in d["name_to_id"].items()}
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", required=True,
                    help="Glob pattern matching all fold directories of one experiment.")
    ap.add_argument("--classes", type=int, default=17)
    ap.add_argument("--data_root", default=None,
                    help="For looking up class names.")
    ap.add_argument("--results_filename", default="test_results.json",
                    help="Which JSON to read from each fold directory "
                         "(use test_results_bymiou.json or test_results_byloss.json "
                         "to aggregate by different checkpoint-selection criteria).")
    ap.add_argument("--out", default=None,
                    help="Where to write aggregated JSON.  Default: cv_summary.json next to the first match.")
    args = ap.parse_args()

    dirs = sorted(glob.glob(args.pattern))
    if not dirs:
        raise SystemExit(f"No directories match: {args.pattern}")
    print(f"Found {len(dirs)} fold directories:")
    for d in dirs:
        print(f"  {d}")
    print()

    names = load_class_names(args.data_root) if args.data_root else {}

    # Collect per-fold results
    fold_mious, fold_accs = [], []
    # per_class_per_fold[c] = list of IoU values (None excluded) for class c across folds
    per_class_per_fold = [[] for _ in range(args.classes)]

    for d in dirs:
        results_path = os.path.join(d, args.results_filename)
        if not os.path.isfile(results_path):
            print(f"  WARNING: no {args.results_filename} in {d} — skipping.")
            continue
        with open(results_path, encoding="utf-8") as f:
            r = json.load(f)
        fold_mious.append(r["miou_present_classes"])
        fold_accs.append(r["overall_accuracy"])
        for c, iou in enumerate(r["per_class_iou"]):
            if iou is not None:
                per_class_per_fold[c].append(iou)

    if not fold_mious:
        raise SystemExit("No test_results.json files found in any fold directory.")

    fold_mious = np.array(fold_mious)
    fold_accs  = np.array(fold_accs)

    # Per-class aggregate (mean over folds where present)
    per_class_mean = []
    per_class_std  = []
    per_class_n    = []
    for c in range(args.classes):
        if per_class_per_fold[c]:
            arr = np.array(per_class_per_fold[c])
            per_class_mean.append(float(arr.mean()))
            per_class_std.append(float(arr.std()))
            per_class_n.append(len(arr))
        else:
            per_class_mean.append(None)
            per_class_std.append(None)
            per_class_n.append(0)

    # mIoU averaged over classes that appeared in at least one fold
    present_means = [m for m in per_class_mean if m is not None]
    overall_class_avg_miou = float(np.mean(present_means)) if present_means else 0.0

    # --- Print ---
    print("=" * 72)
    print(f"Folds aggregated   : {len(fold_mious)}")
    print(f"Per-fold mIoU      : {[f'{x*100:.2f}' for x in fold_mious]}")
    print(f"  mean ± std       : {fold_mious.mean()*100:.2f} ± {fold_mious.std()*100:.2f}")
    print(f"Per-fold accuracy  : {[f'{x*100:.2f}' for x in fold_accs]}")
    print(f"  mean ± std       : {fold_accs.mean()*100:.2f} ± {fold_accs.std()*100:.2f}")
    print()
    print(f"Per-class IoU (mean ± std over folds where class was present in test):")
    name_w = max((len(names.get(c, f'class_{c}')) for c in range(args.classes)), default=12)
    name_w = max(name_w, 12)
    print(f"  {'ID':>3}  {'Class':<{name_w}}  {'Folds':>6}  {'mean IoU':>10}  {'std IoU':>8}")
    print("  " + "-" * (3 + 2 + name_w + 2 + 6 + 2 + 10 + 2 + 8))
    for c in range(args.classes):
        name = names.get(c, f"class_{c}")
        nf = per_class_n[c]
        if nf == 0:
            row = f"  {c:>3}  {name:<{name_w}}  {'0':>6}  {'n/a':>10}  {'n/a':>8}"
        else:
            row = (f"  {c:>3}  {name:<{name_w}}  {nf:>6}  "
                   f"{per_class_mean[c]*100:>8.2f}%  "
                   f"{per_class_std[c]*100:>6.2f}%")
        print(row)
    print()
    print(f"Mean IoU averaged over per-class means: {overall_class_avg_miou*100:.2f}%")
    print("=" * 72)

    summary = {
        "n_folds": int(len(fold_mious)),
        "fold_dirs": dirs,
        "per_fold_miou":           fold_mious.tolist(),
        "per_fold_accuracy":       fold_accs.tolist(),
        "miou_mean":               float(fold_mious.mean()),
        "miou_std":                float(fold_mious.std()),
        "accuracy_mean":           float(fold_accs.mean()),
        "accuracy_std":            float(fold_accs.std()),
        "per_class_mean_iou":      per_class_mean,
        "per_class_std_iou":       per_class_std,
        "per_class_fold_count":    per_class_n,
        "overall_class_avg_miou":  overall_class_avg_miou,
    }

    out = args.out or os.path.join(os.path.dirname(dirs[0]) or ".", "cv_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {out}")


if __name__ == "__main__":
    main()