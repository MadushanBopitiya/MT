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
            
        Returns tensors in (Channels, Points) format: [3, N] and [6, N].
        """
        # Save split variable to identify train vs val/test sets
        self.split = split

        # 1. Build Base Path (Linux safe)
        self.data_path = os.path.join(root_dir, dataset_name, split)
        
        # 2. Define Sub-folders
        self.xyz_folder = os.path.join(self.data_path, "xyz")
        self.seg_folder = os.path.join(self.data_path, "seg")

        # 3. Validation (Crucial for Debugging on Cluster)
        if not os.path.exists(self.xyz_folder) or not os.path.exists(self.seg_folder):
            # Print the FULL absolute path so you see exactly where it's looking
            print(f"\n❌ CRITICAL ERROR: Folder not found!")
            print(f"   Looking for: {os.path.abspath(self.xyz_folder)}")
            print(f"   Note: Linux is CASE-SENSITIVE. Check 'Data' vs 'data'.\n")
            raise ValueError(f"Structure Error! Folders missing in: {self.data_path}")

        # 4. Gather Files
        # We list all .xyz files and assume a matching .seg file exists
        self.xyz_files = sorted(glob.glob(os.path.join(self.xyz_folder, "*.xyz")))
        self.file_list = []

        if len(self.xyz_files) == 0:
            print(f"⚠️  Warning: No .xyz files found in {self.xyz_folder}")

        for xyz_path in self.xyz_files:
            # Extract ID: works for both "/" (Linux) and "\" (Windows)
            stem = os.path.basename(xyz_path).replace('.xyz', '')
            
            # Construct expected SEG path
            seg_path = os.path.join(self.seg_folder, f"{stem}.seg")
            
            if os.path.exists(seg_path):
                self.file_list.append({'xyz': xyz_path, 'seg': seg_path})
            else:
                # Useful to know if you uploaded XYZs but forgot SEGs
                # print(f"⚠️ Warning: No matching label for {stem}")
                pass

        print(f"   > [{dataset_name} | {split.upper()}] Found {len(self.file_list)} samples.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 1. Get Paths
        sample = self.file_list[idx]
        
        # 2. Load Data
        # Note: np.loadtxt is slow for text files. 
        # On a cluster, if this is too slow, we convert to .npy binary later.
        try:
            data = np.loadtxt(sample['xyz'], dtype=np.float32)   # Shape: (N, 6)
            labels = np.loadtxt(sample['seg'], dtype=np.int64)   # Shape: (N,)
        except Exception as e:
            print(f"❌ Corrupt file: {sample['xyz']} - {e}")
            # Return a dummy random sample to prevent crash during 24h training
            return self.__getitem__((idx + 1) % len(self.file_list))

        # 3. Separate Features
        points = data[:, 0:3]   # XYZ
        normals = data[:, 3:6]  # Normals
        
        # 4. Apply Stochastic Augmentations (Train Split Only)
        if self.split == 'train':  # <--- INJECTED BLOCK STARTS HERE
            # A. Full 3D Rotation (SO(3))
            angles = np.random.uniform(0, 2 * np.pi, size=3)
            Rx = np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            
            points = points @ R.T
            normals = normals @ R.T

            # B. Subtle Scale Jitter (0.95 to 1.05)
            scale = np.random.uniform(0.85, 0.95)
            points = points * scale

            # C. Random Translation Shift (+/- 0.01)
            shift = np.random.uniform(-0.02, 0.02, size=3)
            points = points + shift # <--- INJECTED BLOCK ENDS HERE

        # 5. Prepare Input Tensor (Concatenate XYZ + Normals)
        features = np.concatenate((points, normals), axis=1) # (N, 6)
        
        # 6. Transpose for PyTorch (Channels First)
        # Conv1d expects (Batch, Channels, Length) -> (6, N)
        features = features.transpose(1, 0) 
        points   = points.transpose(1, 0)
        
        # 7. To Tensor
        feat_tensor = torch.from_numpy(features).float()
        pos_tensor  = torch.from_numpy(points).float()
        lbl_tensor  = torch.from_numpy(labels).long()
        
        return {
            'pos': pos_tensor,  # (3, N)
            'x': feat_tensor,   # (6, N)
            'y': lbl_tensor     # (N)
        }