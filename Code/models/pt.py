import torch
import torch.nn as nn
import torch.nn.functional as F

# --- STANDALONE GEOMETRIC UTILITIES ---

def knn(x, y, k):
    """
    Batched K-Nearest Neighbors using PyTorch cdist.
    x: [B, 3, N] (Query points)
    y: [B, 3, M] (Dataset points)
    Returns: [B, N, k] indices
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), y)
    xx = torch.sum(x ** 2, dim=1, keepdim=True).transpose(2, 1)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def index_points(points, idx):
    """
    Gathers points based on KNN indices.
    points: [B, C, N]
    idx: [B, N, k]
    Returns: [B, C, N, k]
    """
    B, C, N = points.shape
    _, N_idx, k = idx.shape
    
    # [B, N, k, C]
    points_trans = points.transpose(1, 2).contiguous()
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(-1, 1, 1)
    gathered_points = points_trans[batch_indices, idx, :]
    
    # [B, C, N, k]
    return gathered_points.permute(0, 3, 1, 2).contiguous()

def farthest_point_sample(xyz, npoint):
    """
    Iterative Farthest Point Sampling (FPS).
    xyz: [B, 3, N]
    Returns: [B, npoint] indices
    """
    device = xyz.device
    B, _, N = xyz.shape
    xyz = xyz.transpose(1, 2).contiguous() # [B, N, 3]
    
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        
    return centroids

# --- CORE POINT TRANSFORMER MODULES ---

class PointTransformerLayer(nn.Module):
    def __init__(self, channels, k=16):
        """
        Implementation of the Point Transformer Layer.
        'k' dictates the number of neighbors for vector self-attention.
        """
        super().__init__()
        self.k = k
        
        # phi, psi, alpha: Linear projections
        self.fc_phi = nn.Conv1d(channels, channels, 1, bias=False)
        self.fc_psi = nn.Conv1d(channels, channels, 1, bias=False)
        self.fc_alpha = nn.Conv1d(channels, channels, 1, bias=False)
        
        # Position encoding function delta (theta MLP)
        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False)
        )
        
        # Relation function gamma (MLP producing attention vectors)
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False)
        )

    def forward(self, pos, x):
        # x: [B, C, N], pos: [B, 3, N]
        idx = knn(pos, pos, self.k) # [B, N, k]
        
        # 1. Linear Projections
        phi_x = self.fc_phi(x).unsqueeze(-1)          # [B, C, N, 1]
        psi_x = index_points(self.fc_psi(x), idx)     # [B, C, N, k]
        alpha_x = index_points(self.fc_alpha(x), idx) # [B, C, N, k]
        
        # 2. Position Encoding (delta)
        pos_i = pos.unsqueeze(-1)                     # [B, 3, N, 1]
        pos_j = index_points(pos, idx)                # [B, 3, N, k]
        pos_diff = pos_i - pos_j                      # [B, 3, N, k]
        delta = self.pos_mlp(pos_diff)                # [B, C, N, k]
        
        # 3. Vector Attention Weight Generation (gamma)
        attn_raw = phi_x - psi_x + delta              # [B, C, N, k]
        attn_weight = F.softmax(self.attn_mlp(attn_raw), dim=-1) # [B, C, N, k]
        
        # 4. Feature Aggregation
        val = alpha_x + delta                         # [B, C, N, k]
        out = torch.sum(attn_weight * val, dim=-1)    # [B, C, N]
        
        return out

class PointTransformerBlock(nn.Module):
    def __init__(self, channels, k=16):
        """ Residual block containing the Point Transformer Layer. """
        super().__init__()
        self.linear1 = nn.Sequential(nn.Conv1d(channels, channels, 1, bias=False), nn.BatchNorm1d(channels), nn.ReLU(inplace=True))
        self.pt_layer = PointTransformerLayer(channels, k=k)
        self.linear2 = nn.Sequential(nn.Conv1d(channels, channels, 1, bias=False), nn.BatchNorm1d(channels))
        
    def forward(self, pos, x):
        identity = x
        x = self.linear1(x)
        x = self.pt_layer(pos, x)
        x = self.linear2(x)
        return F.relu(x + identity, inplace=True)

class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, npoint, k=16):
        """ Downsamples the point cloud via FPS and extracts local features via KNN max pooling. """
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, pos, x):
        # FPS to get new sub-sampled coordinates
        fps_idx = farthest_point_sample(pos, self.npoint) # [B, npoint]
        
        # Gather new positions
        new_pos = index_points(pos, fps_idx.unsqueeze(-1)).squeeze(-1) # [B, 3, npoint]
        
        # Apply MLP to incoming features
        x = self.mlp(x) # [B, out_channels, N]
        
        # KNN to gather local neighborhood features from old points around new points
        idx = knn(new_pos, pos, self.k) # [B, npoint, k]
        gathered_x = index_points(x, idx) # [B, out_channels, npoint, k]
        
        # Local Max Pooling
        new_x = torch.max(gathered_x, dim=-1)[0] # [B, out_channels, npoint]
        
        return new_pos, new_x

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        """ Upsamples point features via 3-NN interpolation and merges with skip connections. """
        super().__init__()
        self.mlp_up = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.mlp_skip = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, pos1, x1, pos2, x2):
        """
        pos1, x1: Higher resolution (Skip connection)
        pos2, x2: Lower resolution (To be upsampled)
        """
        x2 = self.mlp_up(x2)
        
        # CRITICAL: 3-NN Inverse Distance Weighting Interpolation strictly requires k=3. 
        # This acts independently of the attention 'k'.
        idx = knn(pos1, pos2, k=3) # [B, N1, 3]
        
        pos2_gathered = index_points(pos2, idx) # [B, 3, N1, 3]
        pos_diff = pos1.unsqueeze(-1) - pos2_gathered
        dist = torch.norm(pos_diff, dim=1) # [B, N1, 3]
        
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm # [B, N1, 3]
        
        x2_gathered = index_points(x2, idx) # [B, C, N1, 3]
        interpolated_x = torch.sum(x2_gathered * weight.unsqueeze(1), dim=-1) # [B, C, N1]
        
        x1 = self.mlp_skip(x1)
        
        # Summation skip connection
        return F.relu(x1 + interpolated_x, inplace=True)

# --- MAIN ARCHITECTURE ---

class PointTransformer(nn.Module):
    def __init__(self, num_classes, num_points=4096, k=16):
        """
        U-Net style architecture for dense prediction tasks.
        Downsampling sequence: N -> N/4 -> N/16 -> N/64 -> N/256.
        The 'k' parameter controls the nearest neighbors for all self-attention layers.
        """
        super().__init__()
        
        n = num_points
        self.k = k # Stored at the top level for synchronized reference
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(6, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        
        # --- ENCODER (Passes 'k' to Attention and Downsampling) ---
        self.stage1 = PointTransformerBlock(32, k=self.k)
        self.td1 = TransitionDown(32, 64, n // 4, k=self.k)
        
        self.stage2 = PointTransformerBlock(64, k=self.k)
        self.td2 = TransitionDown(64, 128, n // 16, k=self.k)
        
        self.stage3 = PointTransformerBlock(128, k=self.k)
        self.td3 = TransitionDown(128, 256, n // 64, k=self.k)
        
        self.stage4 = PointTransformerBlock(256, k=self.k)
        self.td4 = TransitionDown(256, 512, n // 256, k=self.k)
        
        self.stage5 = PointTransformerBlock(512, k=self.k)
        
        # --- DECODER (Passes 'k' to Attention ONLY) ---
        # Note: TransitionUp inherently uses k=3 for interpolation. Do not override it.
        self.tu1 = TransitionUp(512, 256)
        self.stage6 = PointTransformerBlock(256, k=self.k)
        
        self.tu2 = TransitionUp(256, 128)
        self.stage7 = PointTransformerBlock(128, k=self.k)
        
        self.tu3 = TransitionUp(128, 64)
        self.stage8 = PointTransformerBlock(64, k=self.k)
        
        self.tu4 = TransitionUp(64, 32)
        self.stage9 = PointTransformerBlock(32, k=self.k)
        
        # --- OUTPUT HEAD ---
        self.head = nn.Sequential(
            nn.Conv1d(32, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, num_classes, 1)
        )

    def forward(self, pos, x):
        """ Compatible with train_universal.py signature """
        # Stem
        x0 = self.stem(x) # [B, 32, N]
        
        # Encoder
        x1 = self.stage1(pos, x0) # N points
        pos2, x2 = self.td1(pos, x1)
        x2 = self.stage2(pos2, x2) # N/4 points
        
        pos3, x3 = self.td2(pos2, x2)
        x3 = self.stage3(pos3, x3) # N/16 points
        
        pos4, x4 = self.td3(pos3, x3)
        x4 = self.stage4(pos4, x4) # N/64 points
        
        pos5, x5 = self.td4(pos4, x4)
        x5 = self.stage5(pos5, x5) # N/256 points
        
        # Decoder
        x6 = self.tu1(pos4, x4, pos5, x5)
        x6 = self.stage6(pos4, x6)
        
        x7 = self.tu2(pos3, x3, pos4, x6)
        x7 = self.stage7(pos3, x7)
        
        x8 = self.tu3(pos2, x2, pos3, x7)
        x8 = self.stage8(pos2, x8)
        
        x9 = self.tu4(pos, x1, pos2, x8)
        x9 = self.stage9(pos, x9)
        
        # Head
        pred = self.head(x9)
        return F.log_softmax(pred, dim=1), None