import os
import sys
import argparse
import torch
import torch.optim as optim
#from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast   # AMP — halves memory for KPConv & PVT
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import datetime
import random

# --- 1. CLUSTER SETUP ---
matplotlib.use('Agg') 

# --- LOCAL IMPORTS ---
from dataloaders.thesis_dataset import ThesisDataset
from models.pointnet2 import PointNet2
from models.dgcnn import DGCNN
from models.kpconv import KPConv          
from models.pt import PointTransformer
from models.pvt import PVT

# --- 2. SEEDING & REPRODUCIBILITY UTILITIES (RE-INSERTED) ---
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """Ensures each DataLoader worker has a unique, deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- 3. LOGGING UTILITY ---
class Logger:
    def __init__(self, filepath):
        self.file = open(filepath, "w")
    
    def log(self, string):
        print(string)
        self.file.write(string + '\n')
        self.file.flush()

# --- 4. ARGUMENT PARSER ---
def get_args():
    parser = argparse.ArgumentParser(description="Thesis Training Script")
    parser.add_argument('--dataset', type=str, default="MFCAD++", help='Name of dataset folder')
    parser.add_argument('--model', type=str, default="PointNet2", help='Model architecture')
    parser.add_argument('--classes', type=int, default=25, help='Number of classes')
    parser.add_argument('--root', type=str, default="../Data/processed", help='Path to processed data')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--workers', type=int, default=8, help='CPU workers for data loading')
    parser.add_argument('--comment', type=str, default="baseline", help='Custom text added to the checkpoint folder name')
    parser.add_argument('--num_points', type=int, default=4096, help='Point cloud density (4096 or 2048)')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps (e.g. 4 with --batch_size 4 gives effective bs=16)')
    return parser.parse_args()

# --- 5. PLOTTING FUNCTION ---
def update_loss_curve(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- 6. MODEL FACTORY ---
def get_model(model_name, num_classes, num_points, device):
    if model_name == "PointNet2":
        return PointNet2(num_classes=num_classes, normal_channel=True).to(device)
    elif model_name == "DGCNN":
        # k=40 is standard for 2048 points DGCNN segmentation
        return DGCNN(num_classes=num_classes, k=40).to(device)
    elif model_name == "KPConv":
        # dl0 scales with point density: 0.016 for N=4096, 0.02 for N=2048 (Sec. 3.3)
        # K_nb=40 caps the radius neighbourhood for the vectorised forward pass
        return KPConv(num_classes=num_classes, K_nb=40).to(device)
    elif model_name == "PVT":
        # k=32 local window for N=4096, k=16 for N=2048 (voxel branch)
        # S=64 external attention memory dimension (point branch, Sec. 3.2)
        k = 32 if num_points >= 4096 else 16
        return PVT(num_classes=num_classes, num_points=num_points, k=k, S=64).to(device)
    elif model_name == "PointTransformer":
        return PointTransformer(num_classes=num_classes, num_points=num_points, k=32).to(device)
    else:
        raise ValueError(f"❌ Unknown Model: {model_name}")

# --- 7. MAIN TRAINING LOOP ---
def main():
    args = get_args()
    
    # --- A. ACTIVATE SEEDING (HARDCODED) ---
    HARDCODED_SEED = 42
    set_seed(HARDCODED_SEED)
    
    # Decouple generators to stop validation splits from shifting the training random baseline
    train_g = torch.Generator()
    train_g.manual_seed(HARDCODED_SEED)
    
    val_g = torch.Generator()
    val_g.manual_seed(HARDCODED_SEED)
    
    # --- B. FOLDER SETUP ---
    # Injects your custom comment at the end of the folder name
    experiment_name = f"{args.model}_{args.dataset}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}_{args.comment}"
    save_dir = os.path.join("checkpoints", experiment_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = Logger(os.path.join(save_dir, "training_log.txt"))
    logger.log(f"--- STARTING EXPERIMENT: {experiment_name} ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using Device: {device}")
    
    # --- C. LOAD DATA (Hybrid Logic) ---
    try:
        full_train_ds = ThesisDataset(args.root, args.dataset, split='train')
        
        try:
            # Check for MFCAD++ style 'val' folder
            val_ds = ThesisDataset(args.root, args.dataset, split='val')
            train_ds = full_train_ds 
            logger.log(f"Found dedicated 'val' folder. Using standard split.")
            val_file_list = [str(sample['xyz']) for sample in val_ds.file_list]
        except:
            # Fusion 360 style: Random split (Locked by Generator 'g')
            logger.log("No 'val' folder found. Splitting 'train' set (80/20)...")
            total_size = len(full_train_ds)
            val_size = int(0.2 * total_size)
            train_size = total_size - val_size
            
            # Create a dedicated validation dataset clone and explicitly turn off augmentations
            full_val_ds = ThesisDataset(args.root, args.dataset, split='train')
            full_val_ds.split = 'val'

            # Symmetrically determine random indices using our separate validation generator
            indices = torch.randperm(total_size, generator=val_g).tolist()
            train_idx = indices[:train_size]
            val_idx = indices[train_size:]

            # Build subsets mapping points correctly to their respective active/inactive pipelines
            train_ds = torch.utils.data.Subset(full_train_ds, train_idx)
            val_ds = torch.utils.data.Subset(full_val_ds, val_idx)
            
            val_file_list = [str(full_train_ds.file_list[idx]['xyz']) for idx in val_idx]

        # Save Record
        val_record_path = os.path.join(save_dir, "validation_split.txt")
        with open(val_record_path, "w") as f:
            f.write(f"Seed Used: {HARDCODED_SEED}\nTotal Val Files: {len(val_file_list)}\n")
            for filename in val_file_list: f.write(filename + "\n")
        
        # Loaders with Multi-process Fix
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, 
            drop_last=True, num_workers=args.workers, pin_memory=True,
            worker_init_fn=seed_worker, generator=train_g
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.workers, pin_memory=True,
            generator=val_g 
        )
        
        logger.log(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    except Exception as e:
        logger.log(f"❌ Data Load Error: {e}")
        return

    # --- D. BUILD MODEL ---
    # Because set_seed was called earlier, these weights are now deterministic
    model = get_model(args.model, args.classes, args.num_points, device)
    
    # 1. AdamW Optimizer with strict Weight Decay
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 2. AMP GradScaler — active for KPConv and PVT; no-op for all other models
    # Halves GPU memory by keeping activations in float16 during the forward pass.
    # GradScaler handles the loss scaling needed to keep fp16 gradients stable.
    use_amp = args.model in ("KPConv", "PVT") and device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)

    effective_bs = args.batch_size * args.grad_accum
    logger.log(f"AMP enabled: {use_amp} | Batch size: {args.batch_size} | "
               f"Grad accum steps: {args.grad_accum} | Effective batch size: {effective_bs}")
    
    # 2. Build the Hybrid Scheduler
    # warmup_epochs = 5
    # Ramps from 1% of args.lr up to 100% of args.lr over this number of epochs
    # warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    
    # Cosine decay for the remaining epochs
    # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs - warmup_epochs), eta_min=1e-5)
    
    # Stitch them together
    # scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    criterion = nn.NLLLoss()

    # --- TRACKERS & EARLY STOPPING SETUP ---
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # 1. Best scores
    best_val_acc = 0.0
    min_val_loss = float('inf')
    
    # 2. Early Stopping settings
    patience = 20  
    epochs_no_improve = 0
    
    # 3. File paths for Double Saving
    path_best_acc = os.path.join(save_dir, "best_model_acc.pth")
    path_best_loss = os.path.join(save_dir, "best_model_loss.pth")
   
    
    # --- E. EPOCH LOOP ---
    for epoch in range(args.epochs):

        # Force shuffling sequence and dataset transformations to reset identically every epoch
        train_loader.generator.manual_seed(HARDCODED_SEED + epoch)

        # A. TRAIN
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}", ncols=100)
        
        optimizer.zero_grad()   # zero once before the accumulation loop

        for step, batch in enumerate(pbar):
            pos, x, y = batch['pos'].to(device), batch['x'].to(device), batch['y'].to(device)

            # Forward pass inside autocast (fp16 where possible, fp32 for BN/softmax)
            with autocast(enabled=use_amp):
                pred, _ = model(pos, x)
                # Divide loss by accum steps so gradients average correctly
                loss = criterion(pred, y) / args.grad_accum

            # Backward pass via scaler (no-op scaling when AMP is off)
            scaler.scale(loss).backward()

            # Only update weights after grad_accum mini-batches
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                # Unscale gradients in-place so we can clip them in their true range.
                # When AMP is off this is a no-op.
                scaler.unscale_(optimizer)

                # Gradient clipping: caps the global gradient norm at max_norm.
                # Prevents a single bad batch from triggering an explosion that
                # eventually surfaces as NaN losses several epochs later.
                # max_norm=1.0 is the standard default used in Transformer
                # codebases (HuggingFace, fairseq, official Point Transformer).
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Logging uses the unscaled loss value
            true_loss = loss.item() * args.grad_accum
            train_loss    += true_loss
            train_correct += pred.max(1)[1].eq(y).sum().item()
            train_total   += y.numel()
            pbar.set_postfix({'loss': f"{true_loss:.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = (train_correct / train_total) * 100

        # B. VALIDATE
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                pos, x, y = batch['pos'].to(device), batch['x'].to(device), batch['y'].to(device)
                with autocast(enabled=use_amp):
                    pred, _ = model(pos, x)
                val_loss += criterion(pred, y).item()
                val_correct += pred.max(1)[1].eq(y).sum().item()
                val_total += y.numel()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = (val_correct / val_total) * 100
        
        # C. LOGGING & SAVING
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)

        update_loss_curve(history, save_path=os.path.join(save_dir, "loss_curve.png"))

        log_msg = f"Ep {epoch+1}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {avg_val_acc:.2f}%"
        logger.log(log_msg)

        # --- NEW: UPDATE LEARNING RATE ---
        # scheduler.step()
        # current_lr = scheduler.get_last_lr()[0]
        # logger.log(f"   -> LR adjusted to: {current_lr:.6f} for next epoch")

        # --- STRATEGY A: Save "Most Correct" Model (Accuracy) ---
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), path_best_acc)
            logger.log(f"   >>> ⭐ New Best Accuracy: {best_val_acc:.2f}% (Saved: best_model_acc.pth)")

        # --- STRATEGY B: Save "Most Confident" Model (Loss) & Manage Patience ---
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0  # RESET counter because we found a better version
            torch.save(model.state_dict(), path_best_loss)
            logger.log(f"   >>> 🎯 New Best Generalization: {min_val_loss:.4f} (Saved: best_model_loss.pth)")
        else:
            epochs_no_improve += 1
            logger.log(f"   ⚠️ Loss hasn't improved for {epochs_no_improve}/{patience} epochs.")

        # --- STRATEGY C: Regular Safety Checkpoints (Every 20 Epochs) ---
        if (epoch + 1) % 20 == 0:
            chk_path = os.path.join(save_dir, f"checkpoint_ep{epoch+1}.pth")
            torch.save(model.state_dict(), chk_path)
            logger.log(f"   💾 Periodic checkpoint saved: checkpoint_ep{epoch+1}.pth")
            
        # Always save the absolute last state just in case
        torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))

        # --- THE KILL SWITCH: Early Stopping Trigger ---
        if epochs_no_improve >= patience:
            logger.log(f"\n🚨 EARLY STOPPING ACTIVATED at Epoch {epoch+1}")
            logger.log(f"Reason: Validation Loss failed to improve for {patience} consecutive epochs.")
            logger.log(f"Final results will be based on the best saved .pth files.")
            break # Exits the training loop early
            
if __name__ == "__main__":
    main()