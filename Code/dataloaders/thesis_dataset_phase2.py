"""
thesis_dataset_phase2.py

Phase 2 dataset loader for fine-tuning experiments on the
Machining_Tools dataset.

Differences from Phase 1's ThesisDataset:
    - Reads a SPLIT FILE LIST (one part stem per line) instead of
      scanning a directory.  The CV splitter (make_cv_splits.py) writes
      these lists to fold{k}_train.txt / fold{k}_test.txt.
    - Single boolean `augment` instead of `split='train'/'val'` — the
      caller decides directly whether augmentations are on.
    - Sample dict format is unchanged: {'pos': [3,N], 'x': [6,N],
      'y': [N]}.  Labels can include -1 (unmapped); the trainer's
      CrossEntropyLoss(ignore_index=-1) handles those.

Augmentations match Phase 1 byte-for-byte:
    - Full SO(3) rotation (random Euler angles, Rz @ Ry @ Rx).
    - Uniform scale in [0.85, 0.95]  (asymmetric, intentional — keeps
      points inside the unit sphere after the translation below).
    - Translation ±0.02 per axis.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset


class ThesisDatasetPhase2(Dataset):
    def __init__(self, data_root, split_file, augment=True, use_npy=True):
        """
        Args:
            data_root:   Path containing xyz/, seg/ subfolders
                         (e.g. ".../Machining_Tools/train").
            split_file:  Text file listing part stems, one per line.
                         A stem like 'a01_bearing_holder_ToolType_pointcloud_n4096'
                         maps to '{data_root}/xyz/{stem}.xyz' and
                         '{data_root}/seg/{stem}.seg'.
            augment:     If True, apply Phase 1's stochastic
                         augmentations (rotation + scale + translation).
                         Use True for training, False for val/test.
            use_npy:     If True, prefer xyz_npy/*.npy and seg_npy/*.npy
                         when present (Phase 1's convert_to_npy.py
                         convention).  Falls back to .xyz/.seg text.
        """
        self.augment = augment

        self.data_root      = data_root
        self.xyz_folder     = os.path.join(data_root, "xyz")
        self.seg_folder     = os.path.join(data_root, "seg")
        self.xyz_npy_folder = os.path.join(data_root, "xyz_npy")
        self.seg_npy_folder = os.path.join(data_root, "seg_npy")

        self.use_npy = (
            use_npy
            and os.path.isdir(self.xyz_npy_folder)
            and os.path.isdir(self.seg_npy_folder)
        )

        if not os.path.isdir(self.xyz_folder) or not os.path.isdir(self.seg_folder):
            raise ValueError(
                f"Missing xyz/ or seg/ under {data_root}.  "
                f"Run prepare_tooltype_data.py first."
            )

        # Read split file: one part stem per line, ignoring blanks/comments
        if not os.path.isfile(split_file):
            raise ValueError(f"Split file not found: {split_file}")
        with open(split_file, encoding="utf-8") as fh:
            stems = [line.strip() for line in fh
                     if line.strip() and not line.startswith("#")]
        if not stems:
            raise ValueError(f"Split file is empty: {split_file}")

        # Build file_list, dropping any stem whose .xyz/.seg pair is missing
        self.file_list = []
        missing = []
        for stem in stems:
            xyz_path = os.path.join(self.xyz_folder, f"{stem}.xyz")
            seg_path = os.path.join(self.seg_folder, f"{stem}.seg")
            if not (os.path.exists(xyz_path) and os.path.exists(seg_path)):
                missing.append(stem)
                continue
            self.file_list.append({
                "xyz":     xyz_path,
                "seg":     seg_path,
                "xyz_npy": os.path.join(self.xyz_npy_folder, f"{stem}.npy"),
                "seg_npy": os.path.join(self.seg_npy_folder, f"{stem}.npy"),
                "stem":    stem,
            })

        loader_type = "binary .npy" if self.use_npy else "text .xyz/.seg"
        n_listed = len(stems)
        n_loaded = len(self.file_list)
        n_drop = len(missing)
        print(f"   > Phase2 dataset @ {data_root}  | split: "
              f"{os.path.basename(split_file)}  | augment={augment}  "
              f"| samples={n_loaded}/{n_listed} ({loader_type})")
        if missing:
            print(f"     WARNING: {n_drop} stems from split file had no matching "
                  f"files and were skipped: {missing[:3]}{'...' if n_drop > 3 else ''}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        # 1. Load
        try:
            if (self.use_npy
                    and os.path.exists(sample["xyz_npy"])
                    and os.path.exists(sample["seg_npy"])):
                data   = np.load(sample["xyz_npy"])
                labels = np.load(sample["seg_npy"])
            else:
                data   = np.loadtxt(sample["xyz"], dtype=np.float32)
                labels = np.loadtxt(sample["seg"], dtype=np.int64)
        except Exception as e:
            print(f"   Corrupt file: {sample['xyz']} ({e})")
            return self.__getitem__((idx + 1) % len(self.file_list))

        # 2. Separate
        points  = data[:, 0:3].astype(np.float32)
        normals = data[:, 3:6].astype(np.float32)

        # 3. Augmentations  (mirrors Phase 1's thesis_dataset.py exactly)
        if self.augment:
            # A. Full 3D Rotation (SO(3))
            angles = np.random.uniform(0, 2 * np.pi, size=3)
            ca, sa = np.cos(angles[0]), np.sin(angles[0])
            cb, sb = np.cos(angles[1]), np.sin(angles[1])
            cc, sc = np.cos(angles[2]), np.sin(angles[2])
            Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
            Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
            Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
            R = (Rz @ Ry @ Rx).astype(np.float32)
            points  = points  @ R.T
            normals = normals @ R.T

            # B. Asymmetric scale — keep inside unit sphere after translation
            scale = np.random.uniform(0.85, 0.95)
            points = points * scale

            # C. Translation
            shift = np.random.uniform(-0.02, 0.02, size=3).astype(np.float32)
            points = points + shift

        # 4. Build feature tensor (channels-first, matches Phase 1)
        features = np.concatenate((points, normals), axis=1)   # (N, 6)
        features = features.transpose(1, 0)                     # (6, N)
        points   = points.transpose(1, 0)                       # (3, N)

        return {
            "pos": torch.from_numpy(points).float(),    # (3, N)
            "x":   torch.from_numpy(features).float(),  # (6, N)
            "y":   torch.from_numpy(labels).long(),     # (N,)
        }
