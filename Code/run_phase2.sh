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

# Short tag for this experiment, used in output folder names.
# Convention: {source}_{strategy}
#   e.g.  mfcad_linprobe  /  mfcad_ft  /  fusion360_linprobe  /  fusion360_ft
COMMENT="mfcad_linprobe"

# Optional class weighting (default off).  Set to "--class_weights" to enable.
CLASS_W=""

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

OUT_ROOT="checkpoints_phase2"

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
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --batch_size "$BATCH_SIZE" \
        --lr_backbone "$LR_BACKBONE" \
        --lr_head "$LR_HEAD" \
        --comment "$COMMENT" \
        --out_root "$OUT_ROOT"

    echo
    echo "----------------------------------------------------------------------"
    echo " FOLD $f  -  TESTING"
    echo "----------------------------------------------------------------------"
    EXP_DIR="${OUT_ROOT}/${MODEL}_from_${SRC_TAG}_${MODE}_fold${f}_${COMMENT}"
    python test_phase2.py \
        --checkpoint "${EXP_DIR}/best_model_miou.pth" \
        --data_root "$DATA_ROOT" \
        --test_file "${SPLIT_DIR}/fold${f}_test.txt" \
        --model "$MODEL" \
        --classes "$CLASSES" \
        --num_points "$NUM_POINTS" \
        --batch_size "$BATCH_SIZE" \
        || echo "WARNING: test failed on fold $f - continuing"
done

# ============================================================================
# === AGGREGATE ==============================================================

echo
echo "----------------------------------------------------------------------"
echo " AGGREGATING ALL 5 FOLDS"
echo "----------------------------------------------------------------------"
python aggregate_folds.py \
    --pattern "${OUT_ROOT}/${MODEL}_from_${SRC_TAG}_${MODE}_fold*_${COMMENT}" \
    --classes "$CLASSES" \
    --data_root "$DATA_ROOT"

echo
echo "Job finished at $(date)"
