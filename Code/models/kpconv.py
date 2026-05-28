import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleKPConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernel_points=15, influence_radius=0.05):
        """
        Pure PyTorch approximation of a Rigid KPConv Layer.
        Positions a set of parameterized kernel points in space and calculates
        linear correlation maps based on Euclidean proximity.
        """
        super(SimpleKPConv, self).__init__()
        self.num_kpoints = num_kernel_points
        self.radius = influence_radius
        self.out_channels = out_channels
        
        # Initialize kernel points uniformly inside a unit sphere shell layout
        # In the original paper, these are optimized using a parameter-free optimization algorithm.
        kpoints_init = torch.randn(num_kernel_points, 3)
        kpoints_init = F.normalize(kpoints_init, p=2, dim=1) * (influence_radius * 0.7)
        kpoints_init[0] = torch.zeros(3) # Fix the absolute center point rigidly at 0
        self.kernel_points = nn.Parameter(kpoints_init, requires_grad=False)
        
        # The weight matrices associated with each discrete kernel point
        self.weights = nn.Parameter(torch.randn(num_kernel_points, in_channels, out_channels))
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))

    def forward(self, pos, x):
        """
        Input Layout:
            pos: [B, 3, N] - Point coordinates
            x:   [B, C, N] - Incoming channel features
        """
        B, C, N = x.shape
        device = x.device
        
        # Reshape to standard tracking sheets: [B, N, C]
        pos_t = pos.transpose(1, 2).contiguous() # [B, N, 3]
        x_t = x.transpose(1, 2).contiguous()     # [B, N, C]
        
        # 1. Compute Pairwise Spacing Map across points
        inner = -2 * torch.matmul(pos_t, pos_t.transpose(2, 1))
        xx = torch.sum(pos_t**2, dim=2, keepdim=True)
        dist_matrix = xx + inner + xx.transpose(2, 1)
        dist_matrix = torch.clamp(dist_matrix, min=0.0) # [B, N, N]
        
        # Mask out points outside our strict continuous influence radius
        # KPConv uses radius-based neighbors instead of fixed k counts
        mask = dist_matrix < (self.radius ** 2) # [B, N, N]
        
        # 2. Map distances to Kernel Points
        # For simplicity and speed in pure PyTorch, we compute relative neighbor positions
        # x_out accumulator sheet
        x_out = torch.zeros(B, N, self.out_channels, device=device)
        
        # Calculate local neighbor difference configurations
        for b in range(B):
            for i in range(N):
                neighbors_idx = torch.where(mask[b, i])[0]
                if len(neighbors_idx) == 0:
                    continue
                
                # Center neighbors around the reference point target
                center_pos = pos_t[b, i] # [3]
                neigh_pos = pos_t[b, neighbors_idx] # [Num_Neighbors, 3]
                neigh_features = x_t[b, neighbors_idx] # [Num_Neighbors, C]
                
                relative_pos = neigh_pos - center_pos.unsqueeze(0) # [Num_Neighbors, 3]
                
                # Compute distance from each neighbor to our continuous kernel points
                # Shape: [Num_Neighbors, Num_Kernel_Points]
                kp_dist = torch.cdist(relative_pos, self.kernel_points) 
                
                # Linear correlation function h(k) = max(0, 1 - ||y - v|| / r)
                correlation = torch.clamp(1.0 - (kp_dist / self.radius), min=0.0)
                
                # Apply Kernel weights tracking transformation
                # correlation: [Neighbors, KPoints] -> weights: [KPoints, C, Out_C]
                # blended_weights layout shape: [Neighbors, C, Out_C]
                blended_weights = torch.einsum('nk,kco->nco', correlation, self.weights)
                
                # Transform incoming point features and aggregate locally
                transformed_feats = torch.einsum('nc,nco->no', neigh_features, blended_weights)
                x_out[b, i] = torch.sum(transformed_feats, dim=0)
                
        return x_out.permute(0, 2, 1).contiguous()


class KPConvResidualBlock(nn.Module):
    """
    The main structural block of the KPConv network architecture.
    Consists of a Unary (1x1) Convolution, a Rigid KPConv, and a final Unary layer with a Shortcut.
    """
    def __init__(self, in_channels, out_channels, num_kpoints=15, radius=0.05):
        super(KPConvResidualBlock, self).__init__()
        mid_channels = out_channels // 4
        
        self.unary1 = nn.Sequential(nn.Conv1d(in_channels, mid_channels, 1, bias=False),
                                    nn.BatchNorm1d(mid_channels),
                                    nn.LeakyReLU(0.2))
        
        self.kpconv = SimpleKPConv(mid_channels, mid_channels, num_kpoints, radius)
        self.bn_kp = nn.BatchNorm1d(mid_channels)
        self.leaky_kp = nn.LeakyReLU(0.2)
        
        self.unary2 = nn.Sequential(nn.Conv1d(mid_channels, out_channels, 1, bias=False),
                                    nn.BatchNorm1d(out_channels))
        
        # Linear shortcut matching layer if dimensions cross boundaries
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, bias=False),
                                          nn.BatchNorm1d(out_channels))
            
    def forward(self, pos, x):
        identity = self.shortcut(x)
        out = self.unary1(x)
        out = self.kpconv(pos, out)
        out = self.leaky_kp(self.bn_kp(out))
        out = self.unary2(out)
        return F.leaky_relu(out + identity, negative_slope=0.2)


class KPConv(nn.Module):
    def __init__(self, num_classes):
        """
        The literal multi-layer U-Net sequence outlined in the paper's 
        Part Segmentation specs (Section 3.4 / 4.4) using continuous radius groupings.
        """
        super(KPConv, self).__init__()
        
        # --- ENCODER LAYER STAGES ---
        # Stem Layer: Takes 6D inputs (XYZ + Normals) -> projects to initial features
        self.stem = nn.Sequential(nn.Conv1d(6, 64, 1, bias=False),
                                  nn.BatchNorm1d(64),
                                  nn.LeakyReLU(0.2))
        
        # EXPLICIT GRID SUBSAMPLING SIZE (dl_0)
        # CHANGE 0.02 FOR 2048 POINTS AND 0.016 FOR 4096 POINTS
        self.dl_0 = 0.016 
        
        # Radii calculated strictly via the paper's formula: r = 2.5 * dl
        r1 = 2.5 * self.dl_0               # Stage 1: 0.05
        r2 = 2.5 * (self.dl_0 * 2)         # Stage 2: 0.10
        r3 = 2.5 * (self.dl_0 * 4)         # Stage 3: 0.20
        r4 = 2.5 * (self.dl_0 * 8)         # Stage 4: 0.40
        
        # Stage 1 Blocks
        self.stage1 = KPConvResidualBlock(64, 64, radius=r1)
        # Stage 2 Blocks
        self.stage2 = KPConvResidualBlock(64, 128, radius=r2)
        # Stage 3 Blocks
        self.stage3 = KPConvResidualBlock(128, 256, radius=r3)
        # Stage 4 Blocks
        self.stage4 = KPConvResidualBlock(256, 512, radius=r4)
        
        # --- DECODER LAYER STAGES ---
        # The paper specifies a strict sequence of Unary layers mapping features backwards
        self.up_stage4 = nn.Sequential(nn.Conv1d(512 + 256, 256, 1, bias=False),
                                       nn.BatchNorm1d(256),
                                       nn.LeakyReLU(0.2))
        
        self.up_stage3 = nn.Sequential(nn.Conv1d(256 + 128, 128, 1, bias=False),
                                       nn.BatchNorm1d(128),
                                       nn.LeakyReLU(0.2))
        
        self.up_stage2 = nn.Sequential(nn.Conv1d(128 + 64, 64, 1, bias=False),
                                       nn.BatchNorm1d(64),
                                       nn.LeakyReLU(0.2))
        
        # --- CLASSIFICATION HEAD ---
        # Standard Unary multi-layer configuration processing back down to target labels
        self.head = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, pos, x):
        """
        Main pass configured to process inputs without modifications to train_universal.py.
        """
        # Step 1: Project raw features via Stem block
        feat = self.stem(x) # [B, 64, N]
        
        # Step 2: Pass down through the Encoder stages
        enc1 = self.stage1(pos, feat) # [B, 64, N]
        enc2 = self.stage2(pos, enc1) # [B, 128, N]
        enc3 = self.stage3(pos, enc2) # [B, 256, N]
        enc4 = self.stage4(pos, enc3) # [B, 512, N]
        
        # Step 3: Upsample and fuse via Encoder Skip-Connections
        # (In pure PyTorch, since point counts stay uniform across tensors, we concatenate directly)
        dec4 = torch.cat((enc4, enc3), dim=1)
        dec4 = self.up_stage4(dec4) # [B, 256, N]
        
        dec3 = torch.cat((dec4, enc2), dim=1)
        dec3 = self.up_stage3(dec3) # [B, 128, N]
        
        dec2 = torch.cat((dec3, enc1), dim=1)
        dec2 = self.up_stage2(dec2) # [B, 64, N]
        
        # Step 4: Final Classification Scoring Head pass
        pred = self.head(dec2) # [B, num_classes, N]
        
        return F.log_softmax(pred, dim=1), None