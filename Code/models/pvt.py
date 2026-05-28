import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def knn(x, k):
    """
    K-Nearest Neighbors for Local Window formulation.
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   
    return idx

def get_local_windows(x, k=20, pos=None):
    """
    Groups points into local computational windows to approximate SWA.
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    idx = knn(pos, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    x_trans = x.transpose(2, 1).contiguous()
    feature = x_trans.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    
    return feature # [B, N, K, C]

class RelativeAttention(nn.Module):
    def __init__(self, channels, num_points, L=32, s_max=1.0):
        """
        The Point Branch: Global Relative Attention.
        Follows Equations 5-8 from the PVT paper.
        """
        super(RelativeAttention, self).__init__()
        self.channels = channels
        self.L = L
        self.s_max = s_max
        self.s_quad = (2.0 * s_max) / L
        
        # Q, K, V Projections
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)
        
        # Learnable Look-up Tables for X, Y, Z (Eq 8)
        self.t_x = nn.Parameter(torch.randn(L, 1))
        self.t_y = nn.Parameter(torch.randn(L, 1))
        self.t_z = nn.Parameter(torch.randn(L, 1))
        
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, pos, x):
        # pos: [B, 3, N], x: [B, C, N]
        B, C, N = x.shape
        
        Q = self.q_conv(x).transpose(1, 2) # [B, N, C]
        K = self.k_conv(x).transpose(1, 2) # [B, N, C]
        V = self.v_conv(x).transpose(1, 2) # [B, N, C]
        
        # 1. Compute Relative Position Coordinates (Eq 6)
        pos_t = pos.transpose(1, 2) # [B, N, 3]
        pos_diff = pos_t.unsqueeze(2) - pos_t.unsqueeze(1) # [B, N, N, 3]
        
        # 2. Quantize into Look-up Table Indices (Eq 7)
        idx = torch.floor((pos_diff + self.s_max) / self.s_quad).long()
        idx = torch.clamp(idx, 0, self.L - 1)
        
        # 3. Retrieve and Sum Embeddings (Eq 8)
        bx = self.t_x[idx[..., 0]].squeeze(-1) # [B, N, N]
        by = self.t_y[idx[..., 1]].squeeze(-1)
        bz = self.t_z[idx[..., 2]].squeeze(-1)
        B_bias = bx + by + bz # [B, N, N]
        
        # 4. Attention Computation (Eq 5)
        energy = torch.matmul(Q, K.transpose(1, 2)) # [B, N, N]
        attention = F.softmax(energy + B_bias, dim=-1)
        
        F_ra = torch.matmul(attention, V).transpose(1, 2) # [B, C, N]
        
        # Add residual connection as per Figure 1 Point Branch logic
        F_global = self.mlp(F_ra) + x
        return F_global

class LocalWindowAttention(nn.Module):
    def __init__(self, channels, k=32):
        """
        The Voxel Branch Approximation: Local Window Feature Aggregation.
        Replaces the C++ sparse hash grid with batched k-NN window attention.
        """
        super(LocalWindowAttention, self).__init__()
        self.k = k
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)
        
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, pos, x):
        B, C, N = x.shape
        
        # Group into local windows
        x_windows = get_local_windows(x, k=self.k, pos=pos) # [B, N, K, C]
        
        # Center-point queries
        Q = self.q_conv(x).transpose(1, 2).unsqueeze(2) # [B, N, 1, C]
        
        # Neighbor keys and values
        # Transform x_windows [B, N, K, C] for linear layers -> view as [B, C, N*K]
        x_win_flat = x_windows.view(B, N*self.k, C).transpose(1, 2)
        K_win = self.k_conv(x_win_flat).transpose(1, 2).view(B, N, self.k, C)
        V_win = self.v_conv(x_win_flat).transpose(1, 2).view(B, N, self.k, C)
        
        # Local Self-Attention
        energy = torch.matmul(Q, K_win.transpose(2, 3)) / math.sqrt(C) # [B, N, 1, K]
        attention = F.softmax(energy, dim=-1)
        
        F_local = torch.matmul(attention, V_win).squeeze(2) # [B, N, C]
        F_local = F_local.transpose(1, 2) # [B, C, N]
        
        F_local_out = self.mlp(F_local) + x
        return F_local_out

class PVTBlock(nn.Module):
    def __init__(self, channels, num_points):
        """
        Dual-branch Point-Voxel Transformer Block combining Local and Global contexts.
        """
        super(PVTBlock, self).__init__()
        self.norm_local = nn.BatchNorm1d(channels)
        self.norm_global = nn.BatchNorm1d(channels)
        
        # CHANGE k=32 FOR 4096 POINTS & k=16 FOR 2048 POINTS
        self.voxel_branch = LocalWindowAttention(channels, k=32)
        self.point_branch = RelativeAttention(channels, num_points)
        
    def forward(self, pos, x):
        # Point Branch (Global Representation)
        f_global = self.point_branch(pos, self.norm_global(x))
        
        # Voxel Branch (Local Representation)
        f_local = self.voxel_branch(pos, self.norm_local(x))
        
        # Feature Fusion (Eq 11)
        return f_global + f_local

class PVT(nn.Module):
    def __init__(self, num_classes, num_points=4096):
        """
        Main PVT Architecture for Part Segmentation (Paper Figure 1).
        """
        super(PVT, self).__init__()
        
        # Initial Projection (Stem) -> Nx64
        self.stem = nn.Sequential(
            nn.Conv1d(6, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        
        # 3 Stacked PVT Blocks
        self.block1 = PVTBlock(64, num_points)
        self.block2 = PVTBlock(64, num_points)
        
        # Transition layer to increase dimensionality for Block 3
        self.transition = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        self.block3 = PVTBlock(128, num_points)
        
        # Global Feature Aggregation MLP
        self.global_mlp = nn.Sequential(
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )
        
        # Final Segmentation Head
        # Concat: Block 1 (64) + Block 2 (64) + Block 3 (128) + Global (1024) = 1280
        self.head = nn.Sequential(
            nn.Conv1d(1280, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Conv1d(256, 50, 1, bias=False),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Conv1d(50, num_classes, 1)
        )

    def forward(self, pos, x):
        B, C, N = x.shape
        
        # Stem
        f0 = self.stem(x) # [B, 64, N]
        
        # PVT Blocks
        f1 = self.block1(pos, f0) # [B, 64, N]
        f2 = self.block2(pos, f1) # [B, 64, N]
        f3 = self.block3(pos, self.transition(f2)) # [B, 128, N]
        
        # Global Aggregation
        f_global = self.global_mlp(f3) # [B, 1024, N]
        f_global = f_global.max(dim=2, keepdim=True)[0] # [B, 1024, 1]
        f_global = f_global.repeat(1, 1, N) # [B, 1024, N]
        
        # Dense Feature Fusion
        f_concat = torch.cat((f1, f2, f3, f_global), dim=1) # [B, 1280, N]
        
        # Point-wise Prediction
        pred = self.head(f_concat) # [B, num_classes, N]
        
        return F.log_softmax(pred, dim=1), None