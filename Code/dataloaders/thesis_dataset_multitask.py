"""
thesis_dataset_multitask.py

Multi-task version of ThesisDatasetPhase2.  For each part, returns
both ToolType and ModuleType labels along with the shared XYZ+normals
features.

Assumes:
    - Both datasets have the same parts (with matching xyz contents) but
      different .seg files and different filename suffixes:
          <stem>_ToolType_pointcloud_n4096.xyz/.seg
          <stem>_ModuleType_pointcloud_n4096.xyz/.seg
    - The split file lists ToolType stems (one per line).  The dataset
      derives the matching ModuleType stem by string replacement.
    - Augmentations match Phase 1's exactly (SO(3) rotation + scale
      [0.85, 0.95] + translation +-0.02).  Identical transforms applied
      to both labelings (which is correct since both label the same
      underlying point cloud).
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset


class ThesisDatasetMultiTask(Dataset):
    def __init__(self, tool_data_root, module_data_root, split_file,
                 augment=True,
                 tool_ignore_class_ids=None,
                 module_ignore_class_ids=None):
        """
        Args:
            tool_data_root      : .../Machining_Tools_drop10/train
            module_data_root    : .../Module_Types/train
            split_file          : path to fold{k}_train.txt (or _test.txt)
                                  listing ToolType stems
            augment             : whether to apply random rotation/scale/translation
            tool_ignore_class_ids   : Tool IDs to mask to -1 at load time
            module_ignore_class_ids : Module IDs to mask to -1 at load time
        """
        self.augment = augment

        self.tool_xyz_folder   = os.path.join(tool_data_root,   "xyz")
        self.tool_seg_folder   = os.path.join(tool_data_root,   "seg")
        self.module_seg_folder = os.path.join(module_data_root, "seg")

        self.tool_ignore_class_ids = (
            np.asarray(sorted(set(int(c) for c in tool_ignore_class_ids)),
                       dtype=np.int64)
            if tool_ignore_class_ids else None
        )
        self.module_ignore_class_ids = (
            np.asarray(sorted(set(int(c) for c in module_ignore_class_ids)),
                       dtype=np.int64)
            if module_ignore_class_ids else None
        )

        if not os.path.isfile(split_file):
            raise ValueError(f"Split file not found: {split_file}")
        with open(split_file, encoding="utf-8") as fh:
            tool_stems = [ln.strip() for ln in fh
                          if ln.strip() and not ln.startswith("#")]
        if not tool_stems:
            raise ValueError(f"Split file is empty: {split_file}")

        # Build file_list: drop stems where any of the three files is missing
        self.file_list = []
        missing = []
        for tool_stem in tool_stems:
            module_stem  = tool_stem.replace("_ToolType_", "_ModuleType_")
            tool_xyz     = os.path.join(self.tool_xyz_folder,   f"{tool_stem}.xyz")
            tool_seg     = os.path.join(self.tool_seg_folder,   f"{tool_stem}.seg")
            module_seg   = os.path.join(self.module_seg_folder, f"{module_stem}.seg")
            if not (os.path.exists(tool_xyz) and os.path.exists(tool_seg)
                    and os.path.exists(module_seg)):
                missing.append(tool_stem)
                continue
            self.file_list.append({
                "tool_xyz":   tool_xyz,
                "tool_seg":   tool_seg,
                "module_seg": module_seg,
                "stem":       tool_stem,
            })

        n_listed, n_loaded, n_drop = len(tool_stems), len(self.file_list), len(missing)
        print(f"   > Phase2-MT dataset  | tool={tool_data_root}  module={module_data_root}")
        print(f"     split: {os.path.basename(split_file)}  | augment={augment}  "
              f"| samples={n_loaded}/{n_listed}")
        if missing:
            print(f"     WARNING: {n_drop} stems had no matching files (tool xyz/seg "
                  f"or module seg) and were skipped: "
                  f"{missing[:3]}{'...' if n_drop > 3 else ''}")
        if self.tool_ignore_class_ids is not None:
            print(f"     tool_ignore_class_ids   = {self.tool_ignore_class_ids.tolist()}")
        if self.module_ignore_class_ids is not None:
            print(f"     module_ignore_class_ids = {self.module_ignore_class_ids.tolist()}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        try:
            data         = np.loadtxt(sample["tool_xyz"],   dtype=np.float32)
            tool_labels  = np.loadtxt(sample["tool_seg"],   dtype=np.int64)
            module_labels = np.loadtxt(sample["module_seg"], dtype=np.int64)
        except Exception as e:
            print(f"   Corrupt file in {sample['stem']}: ({e})")
            return self.__getitem__((idx + 1) % len(self.file_list))

        # Apply class-ID masking (independent for each task)
        if self.tool_ignore_class_ids is not None:
            mask = np.isin(tool_labels, self.tool_ignore_class_ids)
            if mask.any():
                tool_labels = tool_labels.copy()
                tool_labels[mask] = -1
        if self.module_ignore_class_ids is not None:
            mask = np.isin(module_labels, self.module_ignore_class_ids)
            if mask.any():
                module_labels = module_labels.copy()
                module_labels[mask] = -1

        points  = data[:, 0:3].astype(np.float32)
        normals = data[:, 3:6].astype(np.float32)

        # Same augmentation as ThesisDatasetPhase2 -- applied to the shared XYZ.
        # The labels (tool + module) are unchanged by rotation/scale/translation.
        if self.augment:
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

            scale = np.random.uniform(0.85, 0.95)
            points = points * scale

            shift = np.random.uniform(-0.02, 0.02, size=3).astype(np.float32)
            points = points + shift

        features = np.concatenate((points, normals), axis=1)  # (N, 6)
        features = features.transpose(1, 0)                    # (6, N)
        points   = points.transpose(1, 0)                      # (3, N)

        return {
            "pos":      torch.from_numpy(points).float(),         # (3, N)
            "x":        torch.from_numpy(features).float(),       # (6, N)
            "y_tool":   torch.from_numpy(tool_labels).long(),     # (N,)
            "y_module": torch.from_numpy(module_labels).long(),   # (N,)
        }
