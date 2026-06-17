#!/bin/bash
# ============================================================================
# slurm_phase2.sh — trains and evaluates one Phase 2 experiment
#                    (one architecture × one source × one strategy
#                    × all 5 CV folds) as a single SLURM job.
#
# Usage:
#     sbatch --job-name=<descriptive_name> slurm_phase2.sh
#
# Edit the EXPERIMENT VARIABLES below before each submission.  The four
# things that change per experiment are MODEL, PRETRAINED, FREEZE, and
# COMMENT — everything else stays the same across all 20 experiments.
# ============================================================================

#SBATCH --job-name=pn2_mfcad_linprobe
#SBATCH --partition=lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_%x_%j.log
#SBATCH --error=slurm_%x_%j.log

set -e

# === EXPERIMENT VARIABLES (edit per submission) =============================

MODEL=PointNet2

# Phase 1 checkpoint to fine-tune from.  Path is relative to ~/Thesis/Code.
# Use 'ls checkpoints/' on the cluster to find your actual folder name.
PRETRAINED="checkpoints/PointNet2_MFCAD++_lr0.001_bs16_ep150_Adam_nsample=64/best_model_acc.pth"

# Linear probe = "--freeze_encoder", full fine-tune = "" (empty)
FREEZE="--freeze_encoder"

# Experiment name — used as the subfolder under OUT_ROOT.  Keep it short
# and informative.  Output structure is:
#     <OUT_ROOT>/<COMMENT>/fold{0..4}/...
#     <OUT_ROOT>/<COMMENT>/cv_summary_bymiou.json
#     <OUT_ROOT>/<COMMENT>/cv_summary_byloss.json
# Suggested format: {arch}_{source}_{strategy}[_{variant}]
#   e.g.  pn2_mfcad_lp  /  pn2_mfcad_ft  /  pn2_mfcad_ft_drop10  /  pn2_mfcad_ft_drop10_cw
COMMENT="pn2_mfcad_lp"

# Optional class weighting (default off).  Set to "--class_weights" to enable.
CLASS_W=""

# Optional class dropping.  Whitespace-separated class IDs to mask to -1.
# Leave empty to keep all 17 classes.
# Common choices for ToolType:
#   "<10 parts (recommended): 1 3 5 6 8 10 11 13 14 15 16"  -> keeps 6 classes
#   "<5 parts:                1 5 6 8 10 11 13 15"          -> keeps 9 classes
#   "<20 parts:               1 2 3 5 6 8 10 11 13 14 15 16" -> keeps 5 classes
IGNORE_CLASSES=""

# ============================================================================
# === FIXED VARIABLES (rarely change) ========================================

CLASSES=17
NUM_POINTS=4096

DATA_ROOT="$HOME/Thesis/Data/processed/Industrial_Dataset/Machining_Tools/train"
SPLIT_DIR="$DATA_ROOT/splits"
INVENTORY="$DATA_ROOT/_class_inventory_ToolType.json"

EPOCHS=100
PATIENCE=20
BATCH_SIZE=8
LR_BACKBONE=1e-4
LR_HEAD=1e-3

OUT_ROOT="checkpoints_phase2/Machining_Tools"

# ============================================================================
# === ENVIRONMENT SETUP ======================================================

source ~/.bashrc
conda activate thesis_env
cd ~/Thesis/Code

echo "=== Job info ==="
echo "Job name : $SLURM_JOB_NAME"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Started  : $(date)"
nvidia-smi
echo
echo "=== Experiment ==="
echo "Model      : $MODEL"
echo "Data root  : $DATA_ROOT"
echo "Pretrained : $PRETRAINED"
echo "Freeze     : $FREEZE"
echo "Comment    : $COMMENT"
echo

# Sanity check: pretrained checkpoint exists (only if requested)
if [[ -n "$PRETRAINED" && ! -f "$PRETRAINED" ]]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED"
    echo "Existing checkpoint folders:"
    ls -1 checkpoints/ 2>/dev/null | sed 's/^/  /'
    exit 1
fi

SRC_TAG=$(basename "$PRETRAINED" .pth)
MODE="ft"
if [[ -n "$FREEZE" ]]; then
    MODE="frozen"
fi

# Build --ignore_class_ids argument only if IGNORE_CLASSES is non-empty
IGNORE_ARG=""
if [[ -n "$IGNORE_CLASSES" ]]; then
    IGNORE_ARG="--ignore_class_ids $IGNORE_CLASSES"
    echo "Class IDs to drop (mask to -1): $IGNORE_CLASSES"
fi

# ============================================================================
# === 5-FOLD LOOP ============================================================

for f in 0 1 2 3 4; do
    echo
    echo "----------------------------------------------------------------------"
    echo " FOLD $f  -  TRAINING"
    echo "----------------------------------------------------------------------"
    python train_phase2.py \
        --model "$MODEL" \
        --classes "$CLASSES" \
        --num_points "$NUM_POINTS" \
        --data_root "$DATA_ROOT" \
        --split_dir "$SPLIT_DIR" \
        --inventory "$INVENTORY" \
        --fold $f \
        --pretrained_path "$PRETRAINED" \
        $FREEZE \
        $CLASS_W \
        $IGNORE_ARG \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --batch_size "$BATCH_SIZE" \
        --lr_backbone "$LR_BACKBONE" \
        --lr_head "$LR_HEAD" \
        --comment "$COMMENT" \
        --out_root "$OUT_ROOT"

    echo
    echo "----------------------------------------------------------------------"
    echo " FOLD $f  -  TESTING (both checkpoints)"
    echo "----------------------------------------------------------------------"
    EXP_DIR="${OUT_ROOT}/${COMMENT}/fold${f}"

    # Test best-by-mIoU checkpoint
    python test_phase2.py \
        --checkpoint "${EXP_DIR}/best_model_miou.pth" \
        --data_root "$DATA_ROOT" \
        --test_file "${SPLIT_DIR}/fold${f}_test.txt" \
        --model "$MODEL" \
        --classes "$CLASSES" \
        --num_points "$NUM_POINTS" \
        --batch_size "$BATCH_SIZE" \
        $IGNORE_ARG \
        --out "${EXP_DIR}/test_results_bymiou.json" \
        || echo "WARNING: by-mIoU test failed on fold $f - continuing"

    # Test best-by-loss checkpoint
    python test_phase2.py \
        --checkpoint "${EXP_DIR}/best_model_loss.pth" \
        --data_root "$DATA_ROOT" \
        --test_file "${SPLIT_DIR}/fold${f}_test.txt" \
        --model "$MODEL" \
        --classes "$CLASSES" \
        --num_points "$NUM_POINTS" \
        --batch_size "$BATCH_SIZE" \
        $IGNORE_ARG \
        --out "${EXP_DIR}/test_results_byloss.json" \
        || echo "WARNING: by-loss test failed on fold $f - continuing"
done

# ============================================================================
# === AGGREGATE (both criteria) ==============================================

echo
echo "----------------------------------------------------------------------"
echo " AGGREGATING ALL 5 FOLDS  -  by mIoU-best checkpoint"
echo "----------------------------------------------------------------------"
python aggregate_folds.py \
    --pattern "${OUT_ROOT}/${COMMENT}/fold*" \
    --classes "$CLASSES" \
    --data_root "$DATA_ROOT" \
    --results_filename "test_results_bymiou.json" \
    --out "${OUT_ROOT}/${COMMENT}/cv_summary_bymiou.json"

echo
echo "----------------------------------------------------------------------"
echo " AGGREGATING ALL 5 FOLDS  -  by loss-best checkpoint"
echo "----------------------------------------------------------------------"
python aggregate_folds.py \
    --pattern "${OUT_ROOT}/${COMMENT}/fold*" \
    --classes "$CLASSES" \
    --data_root "$DATA_ROOT" \
    --results_filename "test_results_byloss.json" \
    --out "${OUT_ROOT}/${COMMENT}/cv_summary_byloss.json"

echo
echo "Job finished at $(date)"