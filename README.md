# Deep Learning-Based Detection of Machining Features and Tools from CAD-Derived Point Cloud Data

Master's thesis code repository — Madushan Bopitiya, Technical University of Munich (TUM)

This repository contains the source code, training pipelines and experimental scripts for a Master's thesis benchmarking three state-of-the-art point cloud segmentation architectures (PointNet++, DGCNN, Point Transformer) on the task of machining feature recognition from CAD-derived point clouds.

---

## Overview

The thesis presents a systematic two-phase study:

- **Phase 1**: benchmarks the three architectures under a unified protocol on two large CAD segmentation datasets — MFCAD++ (25 classes, 59,665 samples) and Fusion 360 Gallery (8 classes, 35,680 samples).
- **Phase 2**: transfers the six pre-trained models to a small industrial machining tools dataset (73 parts for ToolType with 17 classes, 75 parts for ModuleType with 6 classes) under 5-fold cross-validation.

A central finding of the thesis is the **transfer paradox**: pre-training on the Fusion 360 Gallery dataset, which yields lower absolute performance in Phase 1 than MFCAD++, produces consistently equal or better results when transferred to the industrial task. This suggests that semantic alignment between source and target annotation schemes matters more than raw benchmark performance when selecting a pre-trained model.

---

## Repository Structure

├── dataloaders/
│   ├── thesis_dataset.py                    # dataset loader for Phase 1
│   ├── thesis_dataset_phase2.py             # dataset loader for Phase 2
│   └── thesis_dataset_multitask.py          # Multi-task dataset (ToolType + ModuleType)
├── models/
│   ├── pointnet2.py                         # PointNet++ (part segmentation variant)
│   ├── dgcnn.py                             # DGCNN (part segmentation variant)
│   ├── pt.py                                # Point Transformer (PartSeg26 rewrite)
│   ├── pt_multitask.py                      # Multi-task Point Transformer
│   ├── kpconv.py                            # KPConv (auxiliary, not in main results)
│   └── pvt.py                               # PVT (auxiliary, not in main results)
├── src/
│   └── preprocessing/                       # CAD-to-point-cloud conversion scripts
├── train_universal.py                       # Phase 1 training
├── train_phase2.py                          # Phase 2 fine-tuning (5-fold CV)
├── train_multitask.py                       # Multi-task Phase 2 training
├── test_universal.py                        # Phase 1 evaluation
├── test_phase2.py                           # Phase 2 evaluation
├── test_multitask.py                        # Multi-task evaluation
├── aggregate_folds.py                       # Aggregates 5-fold CV results
├── run_phase2.sh                            # SLURM submission for Phase 2
└── slurm_multitask.sh                       # SLURM submission for multi-task
---

## Key Implementation Details

### Architectures

All three architectures are implemented as PyTorch ports of the reference implementations released by the original authors:

- **PointNet++**: 3 Set Abstraction levels + 3 Feature Propagation levels, Single-Scale Grouping (SSG) configuration. Reference: [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)
- **DGCNN**: 3 EdgeConv layers (k=40), part segmentation head. Reference: [WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn)
- **Point Transformer**: PartSeg26 configuration with vector attention. Reference: [Pointcept/Pointcept](https://github.com/Pointcept/Pointcept)

Documented deviations from the reference implementations (e.g., T-Net omission in DGCNN, 6-channel input for all three) are described in Chapter 3 of the thesis.

### Input Format

All architectures accept:
- **Point count**: 4096 points per sample
- **Feature dimensionality**: 6 channels (XYZ coordinates + surface normals)
- **Normal computation**: exact, inherited from B-Rep faces during CAD-to-point-cloud sampling

## Environment Setup

### Requirements

- Python 3.10+
- PyTorch 2.7+ with CUDA support
- NVIDIA GPU (experiments were run on NVIDIA A100 80GB)

### Installation

```bash
# Clone the repository
git clone https://github.com/MadushanBopitiya/MT.git
cd MT

# Create a conda environment (recommended)
conda create -n thesis_env python=3.10
conda activate thesis_env

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

The `requirements.txt` includes: numpy, matplotlib, tqdm, plyfile, einops.

**Note**: The Point Transformer implementation is a pure-PyTorch rewrite of the Pointcept `PartSeg26` architecture. It does NOT require the CUDA-only `pointops` kernel from the reference implementation.

---

## Datasets

### Public Datasets

- **MFCAD++**: [https://gitlab.com/qub_femg/machine-learning/mfcadplusplus_dataset](https://gitlab.com/qub_femg/machine-learning/mfcadplusplus_dataset)
- **Fusion 360 Gallery Segmentation**: [https://github.com/AutodeskAILab/Fusion360GalleryDataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset)

### Industrial Dataset

The Machining Tools dataset (ToolType and ModuleType modalities) used in Phase 2 is an industrial dataset not publicly released as part of this thesis. Access may be requested through the TUM Chair of Computing in Civil and Building Engineering.

### Preprocessed Data Format

Each processed point cloud sample consists of:
- `.xyz` file: N × 6 array (XYZ + normals)
- `.seg` file: N × 1 array of per-point class IDs
- Optional `.ply` file: same data with per-point RGB encoding of class labels for visualisation

The dataset loader (`dataloaders/thesis_dataset.py`) expects the following directory structure:

```
Data/processed/
├── MFCAD++/
│   ├── train/
│   │   ├── xyz/*.xyz
│   │   ├── seg/*.seg
│   │   └── ply/*.ply
│   ├── val/
│   └── test/
├── Fusion360/
│   ├── train/
│   └── test/
└── Industrial_Dataset/
    ├── Machining_Tools/
    │   └── train/
    └── Module_Types/
        └── train/
```

---

## Usage

### Phase 1: Training Individual Architectures

```bash
python train_universal.py \
    --dataset MFCAD++ \
    --model PointNet2 \
    --classes 25 \
    --num_points 4096 \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-3 \
    --comment baseline
```

Available `--model` values: `PointNet2`, `DGCNN`, `PointTransformer`
Available `--dataset` values: `MFCAD++`, `Fusion360`

### Phase 1: Evaluation

```bash
python test_phase1.py \
    --dataset MFCAD++ \
    --model PointNet2 \
    --checkpoint path/to/best_model_loss.pth
```

### Phase 2: Fine-Tuning (5-fold CV)

```bash
python train_phase2.py \
    --data_root path/to/Machining_Tools \
    --split_dir path/to/splits \
    --pretrained_path checkpoints/phase1/PointTransformer_MFCAD++_best.pth \
    --num_classes 17 \
    --inventory path/to/tool_inventory.json \
    --class_weights \
    --out_root outputs/phase2/PT_from_MFCAD++_ToolType
```

## Trained Model Checkpoints

Model checkpoints are not included in this repository due to file size constraints. To reproduce the reported results without retraining:

1. **Retrain**: run the training commands above to reproduce checkpoints from scratch.
2. **Request archive**: trained checkpoints and full training logs can be requested by contacting the author.

Alternatively, all reported results in the thesis are reproducible from the source code by following the SLURM submission scripts in `scripts/`.

---

## Compute Requirements

Experiments were conducted on the **TUM Leibniz Supercomputing Centre (LRZ)** using the `lrz-dgx-a100-80x8` partition (NVIDIA A100 80GB GPUs).

---

## Reproducibility

All experiments use fixed random seeds (seed=42 for Phase 1, seed=0 for Phase 2) for Python's `random` module, NumPy, PyTorch CPU, and PyTorch CUDA. `torch.backends.cudnn.deterministic` is set to `True` and `benchmark` to `False` to ensure reproducibility across runs.

Validation splits are deterministic and saved to `validation_split.txt` in each experiment folder.

---

## Contact

For questions about this repository or the thesis, please contact:

- **Author**: Madushan Bopitiya
- **Supervisor**: Stavros Nousias, Chair of Computing in Civil and Building Engineering, TUM

---

## Acknowledgements

This work was carried out at the Chair of Computing in Civil and Building Engineering at the Technical University of Munich, with computational resources provided by the Leibniz Supercomputing Centre (LRZ).

The implementations in this repository build upon the reference codebases released by:
- Qi et al. (PointNet++, [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2))
- Wang et al. (DGCNN, [WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn))
- Zhao et al. and Pointcept contributors (Point Transformer, [Pointcept/Pointcept](https://github.com/Pointcept/Pointcept))

---
