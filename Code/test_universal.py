import os
import argparse
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# --- CLUSTER SETUP ---
matplotlib.use('Agg')

# --- LOCAL IMPORTS (all five models) ---
from dataloaders.thesis_dataset import ThesisDataset
from models.pointnet2 import PointNet2
from models.dgcnn import DGCNN
from models.kpconv import KPConv
from models.pt import PointTransformer
from models.pvt import PVT


# --- 1. METRICS FUNCTION ---
def compute_metrics(pred, target, num_classes):
    """
    Calculates Intersection over Union (IoU) and Accuracy per class.
    """
    ious = []
    accs = []
    pred = pred.flatten()
    target = target.flatten()

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        target_count = target_inds.sum()

        # IoU Calculation
        if union == 0:
            ious.append(np.nan)  # Ignore missing classes
        else:
            ious.append(float(intersection) / float(max(union, 1)))

        # Accuracy Calculation
        if target_count == 0:
            accs.append(np.nan)
        else:
            accs.append(float(intersection) / float(max(target_count, 1)))

    return ious, accs


# --- 2. ARGUMENT PARSER ---
def get_args():
    parser = argparse.ArgumentParser(description="Thesis Universal Testing Script")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--dataset', type=str, default="MFCAD++", help='Dataset Name')
    parser.add_argument('--root', type=str, default="../Data/processed", help='Path to data')
    parser.add_argument('--split', type=str, default="test", help='Split to evaluate')
    parser.add_argument('--classes', type=int, default=25, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model', type=str, default="PointNet2", help='Model Architecture')
    parser.add_argument('--num_points', type=int, default=4096, help='Point cloud density (must match training)')
    return parser.parse_args()


# --- 3. MODEL FACTORY (mirrors train_universal.py) ---
def get_model(model_name, num_classes, num_points, device):
    """
    Build a model identically to how it was built during training.
    Architecture configs must match exactly or state_dict loading will fail.
    """
    if model_name == "PointNet2":
        return PointNet2(num_classes=num_classes, normal_channel=True).to(device)
    elif model_name == "DGCNN":
        return DGCNN(num_classes=num_classes, k=40).to(device)
    elif model_name == "KPConv":
        return KPConv(num_classes=num_classes, K_nb=40).to(device)
    elif model_name == "PVT":
        k = 32 if num_points >= 4096 else 16
        return PVT(num_classes=num_classes, num_points=num_points, k=k, S=64).to(device)
    elif model_name == "PointTransformer":
        # nsamples auto-scaled by pt.py based on num_points
        return PointTransformer(num_classes=num_classes, num_points=num_points).to(device)
    else:
        raise ValueError(f"❌ Unknown Model: {model_name}")


# --- 4. MAIN TEST LOOP ---
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- TESTING {args.model} on {args.dataset} (Split: {args.split}) ---")
    print(f"Loading Checkpoint: {args.checkpoint}")
    print(f"Num Points: {args.num_points}  |  Batch Size: {args.batch_size}")

    # 1. Load Data
    try:
        test_ds = ThesisDataset(args.root, args.dataset, split=args.split)
        # ensure no train-time augmentation is applied
        test_ds.split = 'val'   # toggle augmentation off (same trick as in training)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)
        print(f"Test Samples: {len(test_ds)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Load Model
    model = get_model(args.model, args.classes, args.num_points, device)

    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint file not found!")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 3. Inference
    all_preds = []
    all_targets = []

    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, ncols=100):
            pos = batch['pos'].to(device)
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            pred, _ = model(pos, x)
            pred_choice = pred.max(1)[1]

            all_preds.append(pred_choice.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # 4. Process Results
    print("Calculating Metrics...")
    full_preds = np.concatenate(all_preds)
    full_targets = np.concatenate(all_targets)

    ious, accs = compute_metrics(full_preds, full_targets, args.classes)
    mean_iou = np.nanmean(ious) * 100
    mean_acc = np.nanmean(accs) * 100

    print(f"\n✅ RESULTS:")
    print(f"Mean IoU: {mean_iou:.2f}%")
    print(f"Mean Acc: {mean_acc:.2f}%")

    # 5. SAVE RESULTS
    # Per-checkpoint subfolders so testing both best_model_acc.pth and
    # best_model_loss.pth (or any other checkpoint) keeps results separated.
    # Final structure: checkpoints/<experiment>/test_results/<checkpoint_name>/
    model_dir = os.path.dirname(args.checkpoint)
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    results_dir = os.path.join(model_dir, "test_results", ckpt_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created new folder: {results_dir}")

    report_file = os.path.join(results_dir, "final_test_results.txt")
    matrix_file = os.path.join(results_dir, "confusion_matrix.png")

    # A. Confusion Matrix
    # Flatten to 1D for sklearn (sklearn expects [n_samples], we have [n_clouds, n_points])
    cm = confusion_matrix(full_targets.flatten(), full_preds.flatten())
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(args.classes),
                yticklabels=range(args.classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Test Confusion Matrix ({args.model} / {ckpt_name})\n'
              f'mIoU: {mean_iou:.2f}% | mAcc: {mean_acc:.2f}%')
    plt.savefig(matrix_file)
    plt.close()

    # B. Text Report
    with open(report_file, "w") as f:
        f.write(f"--- FINAL TEST RESULTS ---\n")
        f.write(f"Model:      {args.model}\n")
        f.write(f"Dataset:    {args.dataset} (split: {args.split})\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Num Points: {args.num_points}\n")
        f.write(f"Run Folder: {os.path.basename(model_dir)}\n\n")

        f.write(f"Overall Mean IoU: {mean_iou:.2f}%\n")
        f.write(f"Overall Mean Acc: {mean_acc:.2f}%\n")
        f.write("-" * 50 + "\n")

        f.write("PER-CLASS PERFORMANCE:\n")
        for i in range(args.classes):
            if not np.isnan(ious[i]):
                f.write(f"Class {i:2d}: IoU {ious[i]*100:.2f}% | Acc {accs[i]*100:.2f}%\n")
            else:
                f.write(f"Class {i:2d}: No samples in test set.\n")

        f.write("\n" + "-" * 50 + "\n")
        f.write("SCIKIT-LEARN REPORT (Precision/Recall/F1):\n")
        f.write(classification_report(full_targets.flatten(), full_preds.flatten(), digits=4))

    print(f"✅ Report saved to: {report_file}")
    print(f"✅ Matrix saved to: {matrix_file}")


if __name__ == "__main__":
    main()