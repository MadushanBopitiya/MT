#!/usr/bin/env python3
"""
train_multitask.py

Trains a PointTransformerMultiTask on a single CV fold.  Same overall
structure as train_phase2.py but adapted for two heads and a combined
loss:
    L_total = alpha * L_tool + (1 - alpha) * L_module

Saves three checkpoints per fold:
    best_model_tool_miou.pth    -- best by ToolType mIoU on test
    best_model_module_miou.pth  -- best by ModuleType mIoU on test
    best_model_loss.pth         -- best by combined test loss
    last_model.pth              -- final epoch (or early-stopping point)

Early stopping uses combined test loss (matches Phase 1/2 conventions).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataloaders.thesis_dataset_multitask import ThesisDatasetMultiTask
from models.pt_multitask import PointTransformerMultiTask


# ---------------------------------------------------------------------------
# Utilities (logging, seed, etc.)
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(self.path, "w").close()

    def log(self, msg):
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Pretrained loading -- shape-mismatch tolerant
# ---------------------------------------------------------------------------
def load_pretrained_compatible(model, ckpt_path, logger):
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if any(k.startswith("module.") for k in raw):
        raw = {k.replace("module.", "", 1): v for k, v in raw.items()}

    model_state = model.state_dict()
    loaded, skipped = {}, []
    for k, v in raw.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                loaded[k] = v
            else:
                skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
        else:
            skipped.append((k, tuple(v.shape), None))
    missing = [k for k in model_state if k not in loaded]

    model.load_state_dict(loaded, strict=False)

    logger.log(f"Pretrained: loaded {len(loaded)} / {len(model_state)} params from {ckpt_path}.")
    if skipped:
        logger.log(f"  Shape-mismatch (treated as new): {len(skipped)} param(s)")
        for k, src, dst in skipped[:8]:
            logger.log(f"    - {k}: ckpt {src} -> model {dst if dst else '(absent)'}")
    if missing:
        logger.log(f"  Kept random init (not in ckpt): {len(missing)} param(s)")
        for k in missing[:8]:
            logger.log(f"    - {k}")

    return set(loaded.keys())


def split_params_for_optimizer(model, encoder_param_names, freeze_encoder):
    encoder_params, head_params = [], []
    for name, p in model.named_parameters():
        if name in encoder_param_names:
            if freeze_encoder:
                p.requires_grad = False
            else:
                p.requires_grad = True
                encoder_params.append(p)
        else:
            p.requires_grad = True
            head_params.append(p)
    return encoder_params, head_params


# ---------------------------------------------------------------------------
# Class weights (same as train_phase2.py, with optional ignore_ids)
# ---------------------------------------------------------------------------
def compute_class_weights(inventory_path, num_classes, device, ignore_ids=None):
    with open(inventory_path, encoding="utf-8") as f:
        inv = json.load(f)
    counts = np.zeros(num_classes, dtype=np.float64)
    if isinstance(inv, list):
        entries = inv
    elif isinstance(inv, dict) and "classes" in inv:
        entries = inv["classes"]
    else:
        entries = [{"id": int(k),
                    "point_count": v.get("points", 0) if isinstance(v, dict) else 0}
                   for k, v in inv.items()]
    for e in entries:
        cid = int(e["id"])
        if 0 <= cid < num_classes:
            n = e.get("point_count")
            if n is None: n = e.get("n_points")
            if n is None: n = e.get("points", 0)
            counts[cid] = max(int(n), 0)
    if ignore_ids:
        for cid in ignore_ids:
            if 0 <= cid < num_classes:
                counts[cid] = 0
    total = counts.sum()
    if total == 0:
        return None
    safe = np.where(counts > 0, counts, total)
    w = total / (num_classes * safe)
    w = w * num_classes / w.sum()
    return torch.tensor(w, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------
def compute_per_class_iou(tp, fp, fn, num_classes):
    per_class, present = [], []
    for c in range(num_classes):
        denom = tp[c] + fp[c] + fn[c]
        if denom > 0 and (tp[c] + fn[c]) > 0:
            per_class.append(float(tp[c] / denom))
            present.append(True)
        else:
            per_class.append(None)
            present.append(False)
    miou = float(np.mean([v for v, m in zip(per_class, present) if m])) if any(present) else 0.0
    return per_class, miou


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data
    ap.add_argument("--tool_data_root",   required=True)
    ap.add_argument("--module_data_root", required=True)
    ap.add_argument("--split_dir",        required=True,
                    help="Directory containing fold{k}_train.txt and fold{k}_test.txt "
                         "(ToolType splits -- ModuleType labels accessed by stem rewriting).")
    ap.add_argument("--fold", type=int, required=True)
    # Modality config
    ap.add_argument("--num_classes_tool",   type=int, default=17)
    ap.add_argument("--num_classes_module", type=int, default=6)
    ap.add_argument("--num_points", type=int, default=4096)
    ap.add_argument("--tool_ignore_class_ids",   type=int, nargs="*", default=None)
    ap.add_argument("--module_ignore_class_ids", type=int, nargs="*", default=None)
    # Inventories for class weights (optional)
    ap.add_argument("--tool_inventory",   type=str, default=None)
    ap.add_argument("--module_inventory", type=str, default=None)
    ap.add_argument("--class_weights", action="store_true",
                    help="Use inverse-frequency class weights for both tasks "
                         "(reads from --tool_inventory and --module_inventory).")
    # Pretrained
    ap.add_argument("--pretrained_path", type=str, default="",
                    help="Path to Phase 1 PointTransformer checkpoint.")
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="Train only the two heads (linear-probe-style).")
    # Multi-task loss weight
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Weight on tool loss: L = alpha * L_tool + (1-alpha) * L_module.")
    # Training hyperparams
    ap.add_argument("--epochs",     type=int, default=100)
    ap.add_argument("--patience",   type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head",     type=float, default=1e-3)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--workers",    type=int, default=4)
    # Output
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--comment",  type=str, required=True,
                    help="Experiment short name -- becomes the subfolder under out_root.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()
    set_seed(args.seed)

    save_dir = Path(args.out_root) / args.comment / f"fold{args.fold}"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(str(save_dir / "training_log.txt"))
    logger.log(f"--- {args.comment} / fold{args.fold} ---")
    logger.log(f"Started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")
    logger.log(f"alpha (tool weight) = {args.alpha}")

    # --- Data ---
    train_split = Path(args.split_dir) / f"fold{args.fold}_train.txt"
    test_split  = Path(args.split_dir) / f"fold{args.fold}_test.txt"

    if args.tool_ignore_class_ids:
        logger.log(f"Tool ignore IDs   (mask to -1): {sorted(set(args.tool_ignore_class_ids))}")
    if args.module_ignore_class_ids:
        logger.log(f"Module ignore IDs (mask to -1): {sorted(set(args.module_ignore_class_ids))}")

    train_ds = ThesisDatasetMultiTask(
        args.tool_data_root, args.module_data_root, str(train_split),
        augment=True,
        tool_ignore_class_ids=args.tool_ignore_class_ids,
        module_ignore_class_ids=args.module_ignore_class_ids,
    )
    test_ds = ThesisDatasetMultiTask(
        args.tool_data_root, args.module_data_root, str(test_split),
        augment=False,
        tool_ignore_class_ids=args.tool_ignore_class_ids,
        module_ignore_class_ids=args.module_ignore_class_ids,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    logger.log(f"Train parts: {len(train_ds)} | Test parts: {len(test_ds)}")

    # --- Model ---
    model = PointTransformerMultiTask(
        num_classes_tool=args.num_classes_tool,
        num_classes_module=args.num_classes_module,
        num_points=args.num_points,
    ).to(device)

    encoder_names = set()
    if args.pretrained_path:
        encoder_names = load_pretrained_compatible(model, args.pretrained_path, logger)
    else:
        logger.log("No pretrained checkpoint -- training from scratch.")

    encoder_params, head_params = split_params_for_optimizer(
        model, encoder_names, args.freeze_encoder)

    # --- Optimizer ---
    param_groups = []
    if encoder_params and not args.freeze_encoder:
        param_groups.append({"params": encoder_params, "lr": args.lr_backbone})
    param_groups.append({"params": head_params, "lr": args.lr_head})
    optimizer = Adam(param_groups, betas=(0.9, 0.999), weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    logger.log(f"Trainable params: {n_trainable:,} / {n_total:,}  "
               f"(encoder={'frozen' if args.freeze_encoder else 'trainable'}, "
               f"heads=trainable)")
    logger.log(f"Optimizer: Adam, encoder@{args.lr_backbone}, head@{args.lr_head}")

    # --- Class weights ---
    tool_weights = None
    module_weights = None
    if args.class_weights:
        if args.tool_inventory:
            tool_weights = compute_class_weights(
                args.tool_inventory, args.num_classes_tool, device,
                ignore_ids=args.tool_ignore_class_ids)
            if tool_weights is not None:
                logger.log(f"Tool class weights (mean=1):   "
                           f"{tool_weights.cpu().numpy().round(3).tolist()}")
        if args.module_inventory:
            module_weights = compute_class_weights(
                args.module_inventory, args.num_classes_module, device,
                ignore_ids=args.module_ignore_class_ids)
            if module_weights is not None:
                logger.log(f"Module class weights (mean=1): "
                           f"{module_weights.cpu().numpy().round(3).tolist()}")

    # --- Losses ---
    # Both heads end with LogSoftmax -> NLLLoss
    loss_tool   = nn.NLLLoss(weight=tool_weights,   ignore_index=-1)
    loss_module = nn.NLLLoss(weight=module_weights, ignore_index=-1)

    # --- Training loop ---
    best_tool_miou   = -1.0
    best_module_miou = -1.0
    best_loss        = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(args.epochs):
        # ----- train -----
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            pos = batch["pos"].to(device, non_blocking=True)
            x   = batch["x"].to(device, non_blocking=True)
            yt  = batch["y_tool"].to(device, non_blocking=True)
            ym  = batch["y_module"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            tool_log, module_log = model(pos, x)
            Lt = loss_tool(tool_log,   yt)
            Lm = loss_module(module_log, ym)
            L  = args.alpha * Lt + (1 - args.alpha) * Lm
            L.backward()
            optimizer.step()
            ep_loss += float(L.item())
            n_batches += 1
        train_loss = ep_loss / max(n_batches, 1)
        scheduler.step()

        # ----- eval -----
        model.eval()
        tp_t = np.zeros(args.num_classes_tool, dtype=np.int64)
        fp_t = np.zeros_like(tp_t)
        fn_t = np.zeros_like(tp_t)
        tp_m = np.zeros(args.num_classes_module, dtype=np.int64)
        fp_m = np.zeros_like(tp_m)
        fn_m = np.zeros_like(tp_m)
        test_loss_sum = 0.0
        n_test_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                pos = batch["pos"].to(device, non_blocking=True)
                x   = batch["x"].to(device, non_blocking=True)
                yt  = batch["y_tool"].to(device, non_blocking=True)
                ym  = batch["y_module"].to(device, non_blocking=True)
                tool_log, module_log = model(pos, x)
                Lt = loss_tool(tool_log,   yt)
                Lm = loss_module(module_log, ym)
                L  = args.alpha * Lt + (1 - args.alpha) * Lm
                test_loss_sum += float(L.item())
                n_test_batches += 1

                pt_cls = tool_log.argmax(dim=1)
                pm_cls = module_log.argmax(dim=1)
                m_t = (yt != -1)
                m_m = (ym != -1)
                for c in range(args.num_classes_tool):
                    pc = (pt_cls == c) & m_t
                    gc = (yt == c) & m_t
                    tp_t[c] += int((pc & gc).sum().item())
                    fp_t[c] += int((pc & ~gc).sum().item())
                    fn_t[c] += int((~pc & gc).sum().item())
                for c in range(args.num_classes_module):
                    pc = (pm_cls == c) & m_m
                    gc = (ym == c) & m_m
                    tp_m[c] += int((pc & gc).sum().item())
                    fp_m[c] += int((pc & ~gc).sum().item())
                    fn_m[c] += int((~pc & gc).sum().item())

        test_loss = test_loss_sum / max(n_test_batches, 1)
        _, miou_t = compute_per_class_iou(tp_t, fp_t, fn_t, args.num_classes_tool)
        _, miou_m = compute_per_class_iou(tp_m, fp_m, fn_m, args.num_classes_module)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "test_loss": test_loss,
            "tool_miou": miou_t, "module_miou": miou_m,
        })
        logger.log(f"Epoch {epoch:>3}: "
                   f"tr_loss={train_loss:.4f}  te_loss={test_loss:.4f}  "
                   f"tool_mIoU={miou_t*100:.2f}%  module_mIoU={miou_m*100:.2f}%")

        # --- save checkpoints ---
        if miou_t > best_tool_miou:
            best_tool_miou = miou_t
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "tool_miou": miou_t, "module_miou": miou_m},
                       save_dir / "best_model_tool_miou.pth")
        if miou_m > best_module_miou:
            best_module_miou = miou_m
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "tool_miou": miou_t, "module_miou": miou_m},
                       save_dir / "best_model_module_miou.pth")
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "tool_miou": miou_t, "module_miou": miou_m},
                       save_dir / "best_model_loss.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            logger.log(f"Early stopping at epoch {epoch}  "
                       f"(no test-loss improvement for {args.patience} epochs)")
            break

    torch.save({"model_state_dict": model.state_dict()}, save_dir / "last_model.pth")

    # --- summary ---
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "comment": args.comment, "fold": args.fold,
        "alpha": args.alpha,
        "best_tool_miou":   best_tool_miou,
        "best_module_miou": best_module_miou,
        "best_test_loss":   best_loss,
        "epochs_trained":   len(history),
        "args": vars(args),
    }
    with open(save_dir / "fold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.log(f"\nBest tool mIoU   : {best_tool_miou*100:.2f}%")
    logger.log(f"Best module mIoU : {best_module_miou*100:.2f}%")
    logger.log(f"Best test loss   : {best_loss:.4f}")


if __name__ == "__main__":
    main()
