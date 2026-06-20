#!/bin/bash
# ============================================================================
# slurm_multitask.sh -- trains and evaluates one multi-task experiment
#                       (PointTransformer with two heads: ToolType + ModuleType)
#                       across all 5 CV folds in a single SLURM job.
#
# Output structure (same nested layout as slurm_phase2.sh):
#     ${OUT_ROOT}/${COMMENT}/fold{0..4}/
#                           ├── best_model_tool_miou.pth
#                           ├── best_model_module_miou.pth
#                           ├── best_model_loss.pth
#                           ├── test_results_tool_bytoolmiou.json
#                           ├── test_results_tool_byloss.json
#                           ├── test_results_module_bymodulemiou.json
#                           ├── test_results_module_byloss.json
#                           ├── training_log.txt
#                           ├── history.json
#                           └── fold_summary.json
#     ${OUT_ROOT}/${COMMENT}/cv_summary_tool_bytoolmiou.json
#     ${OUT_ROOT}/${COMMENT}/cv_summary_tool_byloss.json
#     ${OUT_ROOT}/${COMMENT}/cv_summary_module_bymodulemiou.json
#     ${OUT_ROOT}/${COMMENT}/cv_summary_module_byloss.json
# ============================================================================

#SBATCH --job-name=mt_pt_fusion360
#SBATCH --partition=lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_%x_%j.log
#SBATCH --error=slurm_%x_%j.log

set -e

# === EXPERIMENT VARIABLES ==================================================

# Pretrained encoder source.  Best to use Phase 1 PT + Fusion360 since
# that worked best for both tasks individually.
PRETRAINED="checkpoints/PointTransformer_Fusion360_lr0.001_bs16_ep150_Adam_nsample=64/best_model_acc.pth"

# Multi-task loss weight: L = ALPHA * L_tool + (1 - ALPHA) * L_module.
# Start with 0.5.  If results look skewed toward one task, sweep this.
ALPHA="0.5"

# Use class weights for both tasks (recommended).
CLASS_W="--class_weights"

# Linear probe = "--freeze_encoder", full fine-tune = "" (empty).
FREEZE=""

# Experiment name.  Lives at ${OUT_ROOT}/${COMMENT}/fold{0..4}/...
COMMENT="mt_pt_fusion360_ft_cw_alpha0.5"

# === FIXED VARIABLES =======================================================

NUM_CLASSES_TOOL=17
NUM_CLASSES_MODULE=6
NUM_POINTS=4096

TOOL_DATA_ROOT="$HOME/Thesis/Data/processed/Industrial_Dataset/Machining_Tools_drop10/train"
MODULE_DATA_ROOT="$HOME/Thesis/Data/processed/Industrial_Dataset/Module_Types/train"

# ToolType splits drive the multi-task training (73 parts, all with both labels).
SPLIT_DIR="$TOOL_DATA_ROOT/splits"

# Inventories for class weight computation
TOOL_INVENTORY="$TOOL_DATA_ROOT/_class_inventory_ToolType.json"
MODULE_INVENTORY="$MODULE_DATA_ROOT/_class_inventory_ModuleType.json"

# Drop the 11 rare tool classes (same as drop10 ToolType experiments)
# and DrillRev2ax (id=1) from ModuleType (matches single-task setup).
TOOL_IGNORE_CLASSES="1 3 5 6 8 10 11 13 14 15 16"
MODULE_IGNORE_CLASSES="1"

EPOCHS=100
PATIENCE=25
BATCH_SIZE=8
LR_BACKBONE=1e-4
LR_HEAD=1e-3

OUT_ROOT="checkpoints_phase2/MultiTask"

# === ENVIRONMENT ===========================================================

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
echo "Pretrained        : $PRETRAINED"
echo "alpha             : $ALPHA"
echo "Freeze            : $FREEZE"
echo "Comment           : $COMMENT"
echo "Tool data root    : $TOOL_DATA_ROOT"
echo "Module data root  : $MODULE_DATA_ROOT"
echo "Tool ignore IDs   : $TOOL_IGNORE_CLASSES"
echo "Module ignore IDs : $MODULE_IGNORE_CLASSES"
echo

if [[ -n "$PRETRAINED" && ! -f "$PRETRAINED" ]]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED"
    ls -1 checkpoints/ 2>/dev/null | sed 's/^/  /'
    exit 1
fi

# ============================================================================
# === 5-FOLD LOOP ============================================================

for f in 0 1 2 3 4; do
    echo
    echo "----------------------------------------------------------------------"
    echo " FOLD $f  -  TRAINING"
    echo "----------------------------------------------------------------------"
    python train_multitask.py \
        --tool_data_root   "$TOOL_DATA_ROOT" \
        --module_data_root "$MODULE_DATA_ROOT" \
        --split_dir        "$SPLIT_DIR" \
        --fold $f \
        --num_classes_tool   "$NUM_CLASSES_TOOL" \
        --num_classes_module "$NUM_CLASSES_MODULE" \
        --num_points         "$NUM_POINTS" \
        --tool_ignore_class_ids   $TOOL_IGNORE_CLASSES \
        --module_ignore_class_ids $MODULE_IGNORE_CLASSES \
        --tool_inventory   "$TOOL_INVENTORY" \
        --module_inventory "$MODULE_INVENTORY" \
        --pretrained_path "$PRETRAINED" \
        $FREEZE \
        $CLASS_W \
        --alpha "$ALPHA" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --batch_size "$BATCH_SIZE" \
        --lr_backbone "$LR_BACKBONE" \
        --lr_head "$LR_HEAD" \
        --comment "$COMMENT" \
        --out_root "$OUT_ROOT"

    EXP_DIR="${OUT_ROOT}/${COMMENT}/fold${f}"

    echo
    echo "----------------------------------------------------------------------"
    echo " FOLD $f  -  TESTING (3 checkpoints x 2 tasks)"
    echo "----------------------------------------------------------------------"

    # Test best-by-tool-mIoU checkpoint
    python test_multitask.py \
        --checkpoint "${EXP_DIR}/best_model_tool_miou.pth" \
        --tool_data_root   "$TOOL_DATA_ROOT" \
        --module_data_root "$MODULE_DATA_ROOT" \
        --test_file "${SPLIT_DIR}/fold${f}_test.txt" \
        --num_classes_tool   "$NUM_CLASSES_TOOL" \
        --num_classes_module "$NUM_CLASSES_MODULE" \
        --num_points "$NUM_POINTS" \
        --batch_size "$BATCH_SIZE" \
        --tool_ignore_class_ids   $TOOL_IGNORE_CLASSES \
        --module_ignore_class_ids $MODULE_IGNORE_CLASSES \
        --out_tool   "${EXP_DIR}/test_results_tool_bytoolmiou.json" \
        --out_module "${EXP_DIR}/test_results_module_bytoolmiou.json" \
        || echo "WARNING: by-tool-mIoU test failed on fold $f - continuing"

    # Test best-by-module-mIoU checkpoint
    python test_multitask.py \
        --checkpoint "${EXP_DIR}/best_model_module_miou.pth" \
        --tool_data_root   "$TOOL_DATA_ROOT" \
        --module_data_root "$MODULE_DATA_ROOT" \
        --test_file "${SPLIT_DIR}/fold${f}_test.txt" \
        --num_classes_tool   "$NUM_CLASSES_TOOL" \
        --num_classes_module "$NUM_CLASSES_MODULE" \
        --num_points "$NUM_POINTS" \
        --batch_size "$BATCH_SIZE" \
        --tool_ignore_class_ids   $TOOL_IGNORE_CLASSES \
        --module_ignore_class_ids $MODULE_IGNORE_CLASSES \
        --out_tool   "${EXP_DIR}/test_results_tool_bymodulemiou.json" \
        --out_module "${EXP_DIR}/test_results_module_bymodulemiou.json" \
        || echo "WARNING: by-module-mIoU test failed on fold $f - continuing"

    # Test best-by-loss checkpoint
    python test_multitask.py \
        --checkpoint "${EXP_DIR}/best_model_loss.pth" \
        --tool_data_root   "$TOOL_DATA_ROOT" \
        --module_data_root "$MODULE_DATA_ROOT" \
        --test_file "${SPLIT_DIR}/fold${f}_test.txt" \
        --num_classes_tool   "$NUM_CLASSES_TOOL" \
        --num_classes_module "$NUM_CLASSES_MODULE" \
        --num_points "$NUM_POINTS" \
        --batch_size "$BATCH_SIZE" \
        --tool_ignore_class_ids   $TOOL_IGNORE_CLASSES \
        --module_ignore_class_ids $MODULE_IGNORE_CLASSES \
        --out_tool   "${EXP_DIR}/test_results_tool_byloss.json" \
        --out_module "${EXP_DIR}/test_results_module_byloss.json" \
        || echo "WARNING: by-loss test failed on fold $f - continuing"
done

# ============================================================================
# === AGGREGATE ==============================================================

for combo in "tool_bytoolmiou:$TOOL_DATA_ROOT" \
             "module_bytoolmiou:$MODULE_DATA_ROOT" \
             "tool_bymodulemiou:$TOOL_DATA_ROOT" \
             "module_bymodulemiou:$MODULE_DATA_ROOT" \
             "tool_byloss:$TOOL_DATA_ROOT" \
             "module_byloss:$MODULE_DATA_ROOT"; do
    NAME="${combo%%:*}"
    DATA="${combo##*:}"
    if [[ "$NAME" == tool_* ]]; then
        CLASSES="$NUM_CLASSES_TOOL"
    else
        CLASSES="$NUM_CLASSES_MODULE"
    fi
    echo
    echo "----------------------------------------------------------------------"
    echo " AGGREGATING : $NAME"
    echo "----------------------------------------------------------------------"
    python aggregate_folds.py \
        --pattern "${OUT_ROOT}/${COMMENT}/fold*" \
        --classes "$CLASSES" \
        --data_root "$DATA" \
        --results_filename "test_results_${NAME}.json" \
        --out "${OUT_ROOT}/${COMMENT}/cv_summary_${NAME}.json"
done

echo
echo "Job finished at $(date)"
