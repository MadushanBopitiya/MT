import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset

class ThesisDataset(Dataset):
    def __init__(self, root_dir, dataset_name, split='train'):
        """
        Universal Dataset Loader for Flat Structure (XYZ/SEG folders).
        Optimized for Linux/Cluster paths.
        
        Expects:
            root_dir/dataset_name/split/xyz/*.xyz
            root_dir/dataset_name/split/seg/*.seg

        If convert_to_npy.py has been run, also expects:
            root_dir/dataset_name/split/xyz_npy/*.npy   (fast binary, preferred)
            root_dir/dataset_name/split/seg_npy/*.npy

        Returns tensors in (Channels, Points) format: [3, N] and [6, N].
        """
        self.split = split

        # 1. Build Base Path (Linux safe)
        self.data_path = os.path.join(root_dir, dataset_name, split)
        
        # 2. Define Sub-folders — text originals and binary alternatives
        self.xyz_folder     = os.path.join(self.data_path, "xyz")
        self.seg_folder     = os.path.join(self.data_path, "seg")
        self.xyz_npy_folder = os.path.join(self.data_path, "xyz_npy")
        self.seg_npy_folder = os.path.join(self.data_path, "seg_npy")

        # 3. Detect whether binary files are available
        self.use_npy = (
            os.path.exists(self.xyz_npy_folder) and
            os.path.exists(self.seg_npy_folder)
        )

        # 4. Validation
        if not os.path.exists(self.xyz_folder) or not os.path.exists(self.seg_folder):
            print(f"\n❌ CRITICAL ERROR: Folder not found!")
            print(f"   Looking for: {os.path.abspath(self.xyz_folder)}")
            print(f"   Note: Linux is CASE-SENSITIVE. Check 'Data' vs 'data'.\n")
            raise ValueError(f"Structure Error! Folders missing in: {self.data_path}")

        # 5. Gather Files (always index from the original .xyz files)
        self.xyz_files = sorted(glob.glob(os.path.join(self.xyz_folder, "*.xyz")))
        self.file_list = []

        if len(self.xyz_files) == 0:
            print(f"⚠️  Warning: No .xyz files found in {self.xyz_folder}")

        for xyz_path in self.xyz_files:
            stem     = os.path.basename(xyz_path).replace('.xyz', '')
            seg_path = os.path.join(self.seg_folder, f"{stem}.seg")

            if os.path.exists(seg_path):
                self.file_list.append({
                    'xyz':     xyz_path,
                    'seg':     seg_path,
                    'xyz_npy': os.path.join(self.xyz_npy_folder, f"{stem}.npy"),
                    'seg_npy': os.path.join(self.seg_npy_folder, f"{stem}.npy"),
                })

        loader_type = "binary .npy" if self.use_npy else "text .xyz/.seg (slow — run convert_to_npy.py to speed up)"
        print(f"   > [{dataset_name} | {split.upper()}] Found {len(self.file_list)} samples. Loading via: {loader_type}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        
        # 2. Load Data
        # Uses fast binary .npy files when available (xyz_npy/ and seg_npy/ folders).
        # Falls back to slow np.loadtxt from original text files if not converted yet.
        try:
            if self.use_npy and os.path.exists(sample['xyz_npy']) and os.path.exists(sample['seg_npy']):
                data   = np.load(sample['xyz_npy'])              # Shape: (N, 6)
                labels = np.load(sample['seg_npy'])              # Shape: (N,)
            else:
                data   = np.loadtxt(sample['xyz'], dtype=np.float32)
                labels = np.loadtxt(sample['seg'], dtype=np.int64)
        except Exception as e:
            print(f"❌ Corrupt file: {sample['xyz']} - {e}")
            return self.__getitem__((idx + 1) % len(self.file_list))

        # 3. Separate Features
        points  = data[:, 0:3]   # XYZ
        normals = data[:, 3:6]   # Normals
        
        # 4. Apply Stochastic Augmentations (Train Split Only)
        if self.split == 'train':
            # A. Full 3D Rotation (SO(3))
            angles = np.random.uniform(0, 2 * np.pi, size=3)
            Rx = np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]])
            R  = Rz @ Ry @ Rx
            points  = points  @ R.T
            normals = normals @ R.T

            # B. Scale Jitter — upper bound 0.95 (not 1.05) to ensure all points
            # remain inside the unit sphere after the ±0.02 translation below.
            scale  = np.random.uniform(0.85, 0.95)
            points = points * scale

            # C. Random Translation Shift (+/- 0.02)
            shift  = np.random.uniform(-0.02, 0.02, size=3)
            points = points + shift

        # 5. Prepare Input Tensor (Concatenate XYZ + Normals)
        features = np.concatenate((points, normals), axis=1)    # (N, 6)
        
        # 6. Transpose for PyTorch (Channels First)
        features = features.transpose(1, 0)                     # (6, N)
        points   = points.transpose(1, 0)                       # (3, N)
        
        # 7. To Tensor
        feat_tensor = torch.from_numpy(features).float()
        pos_tensor  = torch.from_numpy(points).float()
        lbl_tensor  = torch.from_numpy(labels).long()
        
        return {
            'pos': pos_tensor,   # (3, N)
            'x':   feat_tensor,  # (6, N)
            'y':   lbl_tensor    # (N,)
        }