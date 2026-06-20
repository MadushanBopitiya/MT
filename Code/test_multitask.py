#!/usr/bin/env python3
"""
test_multitask.py

Evaluates a multi-task checkpoint on one fold's test set.  Computes
per-class IoU + mIoU for BOTH tasks separately and writes two result
JSONs:

    test_results_tool_<suffix>.json
    test_results_module_<suffix>.json

Each is the same format as test_phase2.py's output, so the existing
aggregate_folds.py works on them without modification.

Usage example:
    python test_multitask.py \\
        --checkpoint .../fold0/best_model_tool_miou.pth \\
        --tool_data_root   .../Machining_Tools_drop10/train \\
        --module_data_root .../Module_Types/train \\
        --test_file        .../splits/fold0_test.txt \\
        --suffix bytoolmiou
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloaders.thesis_dataset_multitask import ThesisDatasetMultiTask
from models.pt_multitask import PointTransformerMultiTask


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


def per_class_iou(tp, fp, fn, num_classes):
    per_class, present = [], []
    for c in range(num_classes):
        denom = tp[c] + fp[c] + fn[c]
        if denom > 0 and (tp[c] + fn[c]) > 0:
            per_class.append(float(tp[c] / denom))
            present.append(True)
        else:
            per_class.append(None)
            present.append(False)
    vals = [v for v, m in zip(per_class, present) if m]
    miou = float(np.mean(vals)) if vals else 0.0
    return per_class, miou, int(sum(present))


@torch.no_grad()
def evaluate(model, loader, device, n_tool, n_module):
    model.eval()
    tp_t = np.zeros(n_tool, dtype=np.int64); fp_t = np.zeros_like(tp_t); fn_t = np.zeros_like(tp_t)
    tp_m = np.zeros(n_module, dtype=np.int64); fp_m = np.zeros_like(tp_m); fn_m = np.zeros_like(tp_m)
    n_correct_t = n_correct_m = 0
    n_points_t = n_points_m = 0

    for batch in loader:
        pos = batch["pos"].to(device, non_blocking=True)
        x   = batch["x"].to(device, non_blocking=True)
        yt  = batch["y_tool"].to(device, non_blocking=True)
        ym  = batch["y_module"].to(device, non_blocking=True)
        tool_log, module_log = model(pos, x)
        pt_cls = tool_log.argmax(dim=1)
        pm_cls = module_log.argmax(dim=1)
        m_t = (yt != -1); m_m = (ym != -1)
        n_correct_t += int((pt_cls[m_t] == yt[m_t]).sum().item())
        n_correct_m += int((pm_cls[m_m] == ym[m_m]).sum().item())
        n_points_t += int(m_t.sum().item()); n_points_m += int(m_m.sum().item())
        for c in range(n_tool):
            pc = (pt_cls == c) & m_t; gc = (yt == c) & m_t
            tp_t[c] += int((pc & gc).sum().item())
            fp_t[c] += int((pc & ~gc).sum().item())
            fn_t[c] += int((~pc & gc).sum().item())
        for c in range(n_module):
            pc = (pm_cls == c) & m_m; gc = (ym == c) & m_m
            tp_m[c] += int((pc & gc).sum().item())
            fp_m[c] += int((pc & ~gc).sum().item())
            fn_m[c] += int((~pc & gc).sum().item())

    iou_t, miou_t, n_present_t = per_class_iou(tp_t, fp_t, fn_t, n_tool)
    iou_m, miou_m, n_present_m = per_class_iou(tp_m, fp_m, fn_m, n_module)
    return {
        "tool": {
            "per_class_iou": iou_t,
            "miou_present_classes": miou_t,
            "n_classes_present_in_test": n_present_t,
            "overall_accuracy": n_correct_t / max(n_points_t, 1),
            "n_test_points": n_points_t,
            "tp": tp_t.tolist(), "fp": fp_t.tolist(), "fn": fn_t.tolist(),
        },
        "module": {
            "per_class_iou": iou_m,
            "miou_present_classes": miou_m,
            "n_classes_present_in_test": n_present_m,
            "overall_accuracy": n_correct_m / max(n_points_m, 1),
            "n_test_points": n_points_m,
            "tp": tp_m.tolist(), "fp": fp_m.tolist(), "fn": fn_m.tolist(),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",        required=True)
    ap.add_argument("--tool_data_root",    required=True)
    ap.add_argument("--module_data_root",  required=True)
    ap.add_argument("--test_file",         required=True)
    ap.add_argument("--num_classes_tool",   type=int, default=17)
    ap.add_argument("--num_classes_module", type=int, default=6)
    ap.add_argument("--num_points", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers",    type=int, default=4)
    ap.add_argument("--tool_ignore_class_ids",   type=int, nargs="*", default=None)
    ap.add_argument("--module_ignore_class_ids", type=int, nargs="*", default=None)
    ap.add_argument("--out_tool",   type=str, required=True,
                    help="Where to write the tool results JSON.")
    ap.add_argument("--out_module", type=str, required=True,
                    help="Where to write the module results JSON.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    raw = torch.load(args.checkpoint, map_location=device)
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model = PointTransformerMultiTask(
        num_classes_tool=args.num_classes_tool,
        num_classes_module=args.num_classes_module,
        num_points=args.num_points,
    ).to(device)
    model.load_state_dict(state, strict=True)

    test_ds = ThesisDatasetMultiTask(
        args.tool_data_root, args.module_data_root, args.test_file,
        augment=False,
        tool_ignore_class_ids=args.tool_ignore_class_ids,
        module_ignore_class_ids=args.module_ignore_class_ids,
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    res = evaluate(model, loader, device,
                   args.num_classes_tool, args.num_classes_module)
    res["checkpoint"] = args.checkpoint
    res["test_file"]  = args.test_file
    res["n_test_parts"] = len(test_ds)

    # ----- Pretty print -----
    print()
    print("=" * 60)
    print(f"TOOL    -- mIoU: {res['tool']['miou_present_classes']*100:>6.2f}%  "
          f"Acc: {res['tool']['overall_accuracy']*100:>6.2f}%  "
          f"({res['tool']['n_classes_present_in_test']} classes present)")
    print(f"MODULE  -- mIoU: {res['module']['miou_present_classes']*100:>6.2f}%  "
          f"Acc: {res['module']['overall_accuracy']*100:>6.2f}%  "
          f"({res['module']['n_classes_present_in_test']} classes present)")
    print("=" * 60)

    # ----- Write separate JSONs in the same format as test_phase2.py -----
    def write_task_json(out_path, task_res):
        # Format-compatible with aggregate_folds.py
        record = dict(task_res)
        record["checkpoint"] = res["checkpoint"]
        record["test_file"]  = res["test_file"]
        record["n_test_parts"] = res["n_test_parts"]
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Wrote: {out_path}")

    write_task_json(args.out_tool,   res["tool"])
    write_task_json(args.out_module, res["module"])


if __name__ == "__main__":
    main()
