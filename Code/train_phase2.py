"""
train_phase2.py

Fine-tunes a Phase 1 pretrained model on one fold of the Machining_Tools
cross-validation split.  Designed to be called five times (once per fold)
from a wrapper script.

Key Phase 2 features (none of which Phase 1 needed):
    - Loads Phase 1 pretrained weights, with shape-mismatch filtering so
      the final classifier (different num_classes) is automatically left
      as the model's fresh random init.
    - Optional --freeze_encoder for linear probing: every param that
      DID load from the checkpoint is frozen; only the fresh head trains.
    - Discriminative learning rates when not frozen: lower LR for the
      encoder (loaded params), higher LR for the new head (fresh params).
    - CrossEntropyLoss with ignore_index=-1 so unmapped points
      (blank CSV rows, missing labels) don't contribute to the loss.
    - Optional --class_weights flag derives inverse-frequency weights
      from _class_inventory_*.json.  Default OFF (uniform weights), as
      requested for the "include everything first" baseline.
    - Reads from a SPLIT FILE LIST (fold{k}_train.txt) rather than
      scanning a directory.

Example:
    python train_phase2.py \\
        --model PointNet2 \\
        --classes 17 \\
        --data_root  "D:/.../Machining_Tools/train" \\
        --split_dir  "D:/.../Machining_Tools/train/splits" \\
        --fold 0 \\
        --pretrained_path "checkpoints/PointNet2_MFCAD++_best.pth" \\
        --epochs 100 --batch_size 8 \\
        --comment "fold0_linear_probe" --freeze_encoder
"""

import os
import sys
import json
import argparse
import random
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports — mirror Phase 1 layout (dataloaders/ and models/)
from dataloaders.thesis_dataset_phase2 import ThesisDatasetPhase2
from models.pointnet2 import PointNet2
from models.dgcnn    import DGCNN
from models.kpconv   import KPConv
from models.pt       import PointTransformer
from models.pvt      import PVT


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


class Logger:
    def __init__(self, filepath):
        self.f = open(filepath, "w", encoding="utf-8")

    def log(self, msg):
        print(msg)
        self.f.write(str(msg) + "\n")
        self.f.flush()


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
    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Pretrained loading — shape-mismatch filtering
# ---------------------------------------------------------------------------
def load_pretrained_compatible(model, ckpt_path, logger):
    """
    Load a Phase 1 checkpoint.  Keys whose tensor shapes don't match the
    Phase 2 model are skipped — typically just the final classifier
    layer (Phase 1 num_classes != Phase 2 num_classes).

    Returns:
        loaded_names : set of param names that were loaded from ckpt.
                       These are the "encoder"; everything else is the
                       newly initialized "head".
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    # Some checkpoints wrap state_dict; some are saved directly
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    # Strip 'module.' if present (DataParallel)
    if any(k.startswith("module.") for k in raw):
        raw = {k.replace("module.", "", 1): v for k, v in raw.items()}

    model_state = model.state_dict()
    loaded, skipped_shape, missing_in_ckpt = {}, [], []
    for k, v in raw.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                loaded[k] = v
            else:
                skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
        else:
            skipped_shape.append((k, tuple(v.shape), None))
    for k in model_state:
        if k not in loaded:
            missing_in_ckpt.append(k)

    model.load_state_dict(loaded, strict=False)

    logger.log(f"Pretrained: loaded {len(loaded)} / {len(model_state)} params from {ckpt_path}.")
    if skipped_shape:
        logger.log(f"  Shape-mismatch (treated as new head): {len(skipped_shape)} param(s)")
        for k, src, dst in skipped_shape[:6]:
            logger.log(f"    - {k}: ckpt {src} -> model {dst if dst else '(absent)'}")
    if missing_in_ckpt:
        logger.log(f"  Kept random init (not in ckpt): {len(missing_in_ckpt)} param(s)")
        for k in missing_in_ckpt[:6]:
            logger.log(f"    - {k}")

    return set(loaded.keys())


def split_params_for_optimizer(model, encoder_param_names, freeze_encoder):
    """
    Partition model parameters into 'encoder' (loaded from pretrained)
    and 'head' (newly initialized).  If freeze_encoder, encoder params
    get requires_grad=False and are excluded from the optimizer.
    """
    encoder_params, head_params = [], []
    encoder_names, head_names   = [], []
    for name, p in model.named_parameters():
        if name in encoder_param_names:
            if freeze_encoder:
                p.requires_grad = False
            else:
                p.requires_grad = True
                encoder_params.append(p)
                encoder_names.append(name)
        else:
            p.requires_grad = True
            head_params.append(p)
            head_names.append(name)
    return encoder_params, encoder_names, head_params, head_names


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def compute_class_weights(inventory_path, num_classes, device, ignore_ids=None):
    """
    Inverse-frequency weights from per-class point counts in the inventory.
    Returned tensor has mean 1 across classes (so the loss magnitude stays
    comparable to the unweighted case).

    If ignore_ids is given, those classes are treated as having zero
    points before normalisation -- the inventory's stored count for
    them is overridden.  This matters when --ignore_class_ids is used
    at training time: the masked points contribute nothing to the loss,
    so they must contribute nothing to the weight normalisation either,
    or the kept classes' weights get crushed.
    """
    with open(inventory_path, encoding="utf-8") as f:
        inv = json.load(f)
    counts = np.zeros(num_classes, dtype=np.float64)
    # Inventory may be a list of dicts or a dict keyed by id
    if isinstance(inv, list):
        entries = inv
    elif isinstance(inv, dict) and "classes" in inv:
        entries = inv["classes"]
    else:
        entries = [{"id": int(k), "point_count": v.get("points", 0) if isinstance(v, dict) else 0}
                   for k, v in inv.items()]
    for e in entries:
        cid = int(e["id"])
        if 0 <= cid < num_classes:
            counts[cid] = max(int(e.get("point_count", e.get("points", 0))), 0)
    # Honour --ignore_class_ids: zero those counts so they don't skew the
    # normalisation.  Without this, a rare-but-ignored class soaks up the
    # weight budget and crushes the kept classes' effective weights.
    if ignore_ids:
        for cid in ignore_ids:
            if 0 <= cid < num_classes:
                counts[cid] = 0
    total = counts.sum()
    if total == 0:
        return None
    # Inverse frequency, then rescale so mean = 1 over classes with data
    safe = np.where(counts > 0, counts, total)  # avoid div-by-zero
    w = total / (num_classes * safe)
    w = w * num_classes / w.sum()
    return torch.tensor(w, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for batch in loader:
        pos, x, y = batch["pos"].to(device), batch["x"].to(device), batch["y"].to(device)
        pred, _ = model(pos, x)                        # [B, num_classes, N]
        loss = criterion(pred, y)
        total_loss += loss.item() * y.numel()

        pred_cls = pred.argmax(dim=1)                  # [B, N]
        mask = (y != -1)
        total_correct += (pred_cls[mask] == y[mask]).sum().item()
        total_count   += mask.sum().item()

        # Confusion accumulators (ignore -1)
        for c in range(num_classes):
            pred_c = (pred_cls == c) & mask
            gt_c   = (y == c) & mask
            tp[c] += (pred_c & gt_c).sum().item()
            fp[c] += (pred_c & ~gt_c).sum().item()
            fn[c] += (~pred_c & gt_c).sum().item()

    # Per-class IoU; classes with no GT in this test set are reported as None
    iou = []
    for c in range(num_classes):
        denom = tp[c] + fp[c] + fn[c]
        iou.append(float(tp[c] / denom) if denom > 0 else None)

    present = [v for v in iou if v is not None]
    miou_present = float(np.mean(present)) if present else 0.0
    acc = total_correct / max(total_count, 1)
    avg_loss = total_loss / max(total_count, 1)
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "miou_present": miou_present,
        "per_class_iou": iou,
        "tp": tp.tolist(), "fp": fp.tolist(), "fn": fn.tolist(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data
    ap.add_argument("--data_root",  type=str, required=True,
                    help="Path containing xyz/, seg/ for Machining_Tools.")
    ap.add_argument("--split_dir",  type=str, required=True,
                    help="Path containing fold{k}_train.txt / fold{k}_test.txt.")
    ap.add_argument("--fold", type=int, required=True,
                    help="Which fold to run (0..K-1).")
    ap.add_argument("--inventory",  type=str, default=None,
                    help="Optional path to _class_inventory_ToolType.json "
                         "(used only with --class_weights).")
    ap.add_argument("--ignore_class_ids", type=int, nargs="*", default=None,
                    help="Class IDs to drop (mask to -1) at load time.  "
                         "Use this to exclude rare classes without re-processing "
                         "the data.  E.g.  --ignore_class_ids 1 3 5 6 8 10 11 13 14 15 16  "
                         "drops the 11 classes with fewer than 10 parts.")
    # Model
    ap.add_argument("--model",   type=str, required=True,
                    choices=["PointNet2", "DGCNN", "KPConv", "PVT", "PointTransformer"])
    ap.add_argument("--classes", type=int, default=17)
    ap.add_argument("--num_points", type=int, default=4096)
    # Pretrained / freezing
    ap.add_argument("--pretrained_path", type=str, default=None,
                    help="Phase 1 checkpoint to initialize from.  If omitted, "
                         "train from random init.")
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="Linear probe: only the fresh head trains.")
    # Optimizer / training
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head",     type=float, default=1e-3)
    ap.add_argument("--epochs",      type=int,   default=100)
    ap.add_argument("--patience",    type=int,   default=20)
    ap.add_argument("--batch_size",  type=int,   default=8)
    ap.add_argument("--grad_accum",  type=int,   default=1)
    ap.add_argument("--workers",     type=int,   default=4)
    # Loss
    ap.add_argument("--class_weights", action="store_true",
                    help="Use inverse-frequency class weights from inventory.")
    # Bookkeeping
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--comment", type=str, default="phase2")
    ap.add_argument("--out_root", type=str, default="checkpoints_phase2",
                    help="Where to save checkpoints and logs.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()
    set_seed(args.seed)

    # Output dir: {out_root}/{comment}/fold{f}
    # The "comment" is now the experiment short name.
    save_dir = Path(args.out_root) / args.comment / f"fold{args.fold}"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(str(save_dir / "training_log.txt"))
    logger.log(f"--- {args.comment} / fold{args.fold} ---")
    logger.log(f"Started: {datetime.datetime.now().isoformat(timespec='seconds')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")

    # --- Data ---
    train_split = Path(args.split_dir) / f"fold{args.fold}_train.txt"
    test_split  = Path(args.split_dir) / f"fold{args.fold}_test.txt"
    if args.ignore_class_ids:
        logger.log(f"Ignoring class IDs (masked to -1): {sorted(set(args.ignore_class_ids))}")
    train_ds = ThesisDatasetPhase2(args.data_root, str(train_split), augment=True,
                                    ignore_class_ids=args.ignore_class_ids)
    test_ds  = ThesisDatasetPhase2(args.data_root, str(test_split),  augment=False,
                                    ignore_class_ids=args.ignore_class_ids)

    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    logger.log(f"Train parts: {len(train_ds)} | Test parts: {len(test_ds)}")

    # --- Model + pretrained ---
    model = get_model(args.model, args.classes, args.num_points, device)
    if args.pretrained_path:
        if not os.path.isfile(args.pretrained_path):
            logger.log(f"ERROR: --pretrained_path not found: {args.pretrained_path}")
            sys.exit(1)
        encoder_param_names = load_pretrained_compatible(model, args.pretrained_path, logger)
    else:
        logger.log("No pretrained path given — training from random init.")
        encoder_param_names = set()  # everything is "head" (= trainable)

    encoder_p, encoder_n, head_p, head_n = split_params_for_optimizer(
        model, encoder_param_names, args.freeze_encoder,
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.log(f"Trainable params: {n_train:,} / {n_total:,}  "
               f"(encoder grouped={len(encoder_n)}, head grouped={len(head_n)})")

    # --- Optimizer ---
    if args.freeze_encoder:
        if not head_p:
            logger.log("ERROR: --freeze_encoder set but no head parameters found.")
            sys.exit(1)
        optimizer = optim.Adam(head_p, lr=args.lr_head)
        logger.log(f"Optimizer: Adam, head only, lr={args.lr_head}")
    else:
        param_groups = []
        if encoder_p:
            param_groups.append({"params": encoder_p, "lr": args.lr_backbone, "name": "encoder"})
        if head_p:
            param_groups.append({"params": head_p, "lr": args.lr_head, "name": "head"})
        optimizer = optim.Adam(param_groups)
        logger.log(f"Optimizer: Adam, encoder@{args.lr_backbone}, head@{args.lr_head}")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # --- Loss ---
    weights = None
    if args.class_weights:
        if not args.inventory:
            logger.log("ERROR: --class_weights set but no --inventory path.")
            sys.exit(1)
        weights = compute_class_weights(args.inventory, args.classes, device,
                                        ignore_ids=args.ignore_class_ids)
        logger.log(f"Class weights (mean=1): {weights.cpu().numpy().round(3).tolist()}")
    # NOTE: Phase 1 models end with LogSoftmax, so Phase 1 used NLLLoss.
    # We keep that convention for compatibility.
    criterion = nn.NLLLoss(ignore_index=-1, weight=weights)

    # --- AMP ---
    use_amp = args.model in ("KPConv", "PVT") and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    logger.log(f"AMP: {use_amp} | batch_size={args.batch_size} | "
               f"grad_accum={args.grad_accum} | effective bs={args.batch_size * args.grad_accum}")

    # --- Training loop ---
    history = {"train_loss": [], "test_loss": [],
               "train_acc": [], "test_acc": [], "test_miou": []}
    best_test_miou = -1.0
    best_test_loss = float("inf")
    epochs_no_improve = 0  # counts epochs since LOSS last improved (drives early stopping)

    for epoch in range(args.epochs):
        # Reset shuffling seed per epoch for reproducibility
        train_loader.generator.manual_seed(args.seed + epoch)

        # Train
        model.train()
        train_loss = train_correct = train_count = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}", ncols=100, leave=False)
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            pos = batch["pos"].to(device, non_blocking=True)
            x   = batch["x"].to(device, non_blocking=True)
            y   = batch["y"].to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                pred, _ = model(pos, x)
                loss = criterion(pred, y) / args.grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for grp in optimizer.param_groups for p in grp["params"]],
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * args.grad_accum * y.numel()
            mask = (y != -1)
            train_correct += (pred.argmax(1)[mask] == y[mask]).sum().item()
            train_count   += mask.sum().item()

        scheduler.step()

        avg_train_loss = train_loss / max(train_count, 1)
        train_acc = train_correct / max(train_count, 1)

        # Test (= held-out eval)
        test = evaluate(model, test_loader, criterion, device, args.classes)

        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(test["loss"])
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test["accuracy"])
        history["test_miou"].append(test["miou_present"])

        logger.log(
            f"Ep {epoch+1:>3}: "
            f"trL={avg_train_loss:.4f} trA={train_acc*100:.1f}% | "
            f"teL={test['loss']:.4f} teA={test['accuracy']*100:.1f}% "
            f"teMIoU={test['miou_present']*100:.2f}%"
        )

        # Track both metrics. Early stopping is driven by TEST LOSS (matches
        # Phase 1's train_universal.py convention).  We also save the best-
        # mIoU checkpoint independently for reporting, since mIoU is the
        # headline metric and best-loss != best-mIoU in general (the long
        # tail makes the two diverge).

        # Best-by-loss checkpoint (drives early stopping)
        if test["loss"] < best_test_loss:
            best_test_loss = test["loss"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": best_test_loss,
                "miou": test["miou_present"],
                "per_class_iou": test["per_class_iou"],
                "args": vars(args),
            }, save_dir / "best_model_loss.pth")
        else:
            epochs_no_improve += 1

        # Best-by-mIoU checkpoint (independent; informational)
        if test["miou_present"] > best_test_miou:
            best_test_miou = test["miou_present"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "miou": best_test_miou,
                "loss": test["loss"],
                "per_class_iou": test["per_class_iou"],
                "args": vars(args),
            }, save_dir / "best_model_miou.pth")

        # Always keep the last
        torch.save(model.state_dict(), save_dir / "last_model.pth")

        if epochs_no_improve >= args.patience:
            logger.log(f"Early stopping at epoch {epoch+1} (test loss did not improve for {args.patience} epochs).")
            break

    # --- Save history + plot ---
    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="train"); plt.plot(history["test_loss"], label="test")
    plt.title("Loss"); plt.xlabel("epoch"); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.plot([a*100 for a in history["train_acc"]], label="train")
    plt.plot([a*100 for a in history["test_acc"]],  label="test")
    plt.title("Accuracy (%)"); plt.xlabel("epoch"); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.plot([m*100 for m in history["test_miou"]], color="darkred")
    plt.title("Test mIoU (%)"); plt.xlabel("epoch"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "curves.png", dpi=120); plt.close()

    # Final fold-summary JSON for the aggregator
    final = {
        "fold": args.fold,
        "model": args.model,
        "pretrained_path": args.pretrained_path,
        "freeze_encoder": args.freeze_encoder,
        "class_weights": args.class_weights,
        "best_test_miou": best_test_miou,
        "best_test_loss": best_test_loss,
        "epochs_trained": len(history["train_loss"]),
    }
    with open(save_dir / "fold_summary.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    logger.log(f"DONE. Best test mIoU = {best_test_miou*100:.2f}%  |  "
               f"Best test loss = {best_test_loss:.4f}")


if __name__ == "__main__":
    main()