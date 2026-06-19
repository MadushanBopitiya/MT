"""
test_phase2.py

Standalone evaluator for a Phase 2 checkpoint on one fold's test set.
Usage parallels Phase 1's test script.  Called either once per fold
manually, or in a loop from run_all_folds.bat / .sh.

What it does:
    1. Loads a checkpoint produced by train_phase2.py.
    2. Builds the same model and loads the weights.
    3. Runs inference on the fold's test split (no gradients, no augmentation).
    4. Computes per-class IoU and mean IoU (over classes present in test set).
    5. Writes results JSON next to the checkpoint.

Example:
    python test_phase2.py \\
        --checkpoint checkpoints_phase2/PointNet2_from_..._fold0_xxx/best_model_miou.pth \\
        --data_root  "D:/.../Machining_Tools/train" \\
        --test_file  "D:/.../Machining_Tools/train/splits/fold0_test.txt" \\
        --model PointNet2 --classes 17
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloaders.thesis_dataset_phase2 import ThesisDatasetPhase2
from models.pointnet2 import PointNet2
from models.dgcnn    import DGCNN
from models.kpconv   import KPConv
from models.pt       import PointTransformer
from models.pvt      import PVT


def get_model(name, num_classes, num_points, device):
    if name == "PointNet2":
        return PointNet2(num_classes=num_classes, normal_channel=True).to(device)
    elif name == "DGCNN":
        return DGCNN(num_classes=num_classes, k=40).to(device)
    elif name == "KPConv":
        return KPConv(num_classes=num_classes, K_nb=40).to(device)
    elif name == "PVT":
        k = 32 if num_points >= 4096 else 16
        return PVT(num_classes=num_classes, num_points=num_points, k=k, S=64).to(device)
    elif name == "PointTransformer":
        return PointTransformer(num_classes=num_classes, num_points=num_points).to(device)
    raise ValueError(f"Unknown model: {name}")


def load_class_names(data_root):
    """Pull id -> name from _global_mapping_*.json if present."""
    root = Path(data_root)
    for f in root.glob("_global_mapping_*.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and "name_to_id" in d:
            return {int(v): k for k, v in d["name_to_id"].items()}
    return {}


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    n_correct = n_points = 0

    for batch in loader:
        pos = batch["pos"].to(device, non_blocking=True)
        x   = batch["x"].to(device, non_blocking=True)
        y   = batch["y"].to(device, non_blocking=True)
        pred, _ = model(pos, x)                  # [B, C, N]
        pred_cls = pred.argmax(dim=1)            # [B, N]
        mask = (y != -1)
        n_correct += (pred_cls[mask] == y[mask]).sum().item()
        n_points  += mask.sum().item()

        for c in range(num_classes):
            pc = (pred_cls == c) & mask
            gc = (y == c) & mask
            tp[c] += (pc & gc).sum().item()
            fp[c] += (pc & ~gc).sum().item()
            fn[c] += (~pc & gc).sum().item()

    per_class_iou, present_mask = [], []
    for c in range(num_classes):
        denom = tp[c] + fp[c] + fn[c]
        if denom > 0 and (tp[c] + fn[c]) > 0:
            # class actually has GT in this test set
            per_class_iou.append(float(tp[c] / denom))
            present_mask.append(True)
        else:
            per_class_iou.append(None)
            present_mask.append(False)

    present = [v for v, m in zip(per_class_iou, present_mask) if m]
    miou_present = float(np.mean(present)) if present else 0.0
    overall_acc  = n_correct / max(n_points, 1)
    return {
        "per_class_iou": per_class_iou,
        "miou_present_classes": miou_present,
        "n_classes_present_in_test": int(sum(present_mask)),
        "overall_accuracy": overall_acc,
        "n_test_points": n_points,
        "tp": tp.tolist(), "fp": fp.tolist(), "fn": fn.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to best_model_miou.pth (saved by train_phase2.py).")
    ap.add_argument("--data_root",  required=True,
                    help="Path containing xyz/, seg/.")
    ap.add_argument("--test_file",  required=True,
                    help="Path to fold{k}_test.txt list of stems.")
    ap.add_argument("--model", required=True,
                    choices=["PointNet2", "DGCNN", "KPConv", "PVT", "PointTransformer"])
    ap.add_argument("--classes", type=int, default=17)
    ap.add_argument("--num_points", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ignore_class_ids", type=int, nargs="*", default=None,
                    help="Class IDs to mask to -1 in the test set.  Should match "
                         "the --ignore_class_ids used during training so that "
                         "ignored classes' points contribute nothing to the IoU.")
    ap.add_argument("--out", type=str, default=None,
                    help="Where to write results.json.  Default: alongside the checkpoint.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load checkpoint (handle both {state_dict, args, ...} and bare state_dict)
    raw = torch.load(args.checkpoint, map_location=device)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # Build model and load
    model = get_model(args.model, args.classes, args.num_points, device)
    model.load_state_dict(state, strict=True)

    # Data
    test_ds = ThesisDatasetPhase2(args.data_root, args.test_file, augment=False,
                                   ignore_class_ids=args.ignore_class_ids)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # Evaluate
    res = evaluate(model, test_loader, device, args.classes)
    res["checkpoint"]   = args.checkpoint
    res["test_file"]    = args.test_file
    res["model"]        = args.model
    res["n_test_parts"] = len(test_ds)

    # Pretty per-class table
    names = load_class_names(args.data_root)
    name_w = max((len(names.get(c, f"class_{c}")) for c in range(args.classes)), default=12)
    name_w = max(name_w, 12)
    print()
    print(f"{'ID':>3}  {'Class':<{name_w}}  {'IoU':>8}")
    print("-" * (3 + 2 + name_w + 2 + 8))
    for c in range(args.classes):
        iou = res["per_class_iou"][c]
        name = names.get(c, f"class_{c}")
        s = "  n/a " if iou is None else f"{iou*100:>6.2f}%"
        print(f"{c:>3}  {name:<{name_w}}  {s}")
    print("-" * (3 + 2 + name_w + 2 + 8))
    print(f"   mIoU (over {res['n_classes_present_in_test']} present classes): "
          f"{res['miou_present_classes']*100:.2f}%")
    print(f"   Overall accuracy: {res['overall_accuracy']*100:.2f}%  "
          f"({res['n_test_points']:,} points)")

    # Save JSON
    out_path = args.out or os.path.join(os.path.dirname(args.checkpoint), "test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()