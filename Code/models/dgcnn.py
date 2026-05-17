import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    """
    K-Nearest Neighbors implementation in pure PyTorch.
    Input:
        x: [B, C, N] (Feature or coordinate tensor to find neighbors)
        k: int (Number of neighbors)
    Output:
        idx: [B, N, k] (Indices of the nearest neighbors)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    # Retrieve indices of the k nearest neighbors via negative distance
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # Shape: (Batch, N, k)
    return idx

def get_graph_feature(x, k=40, idx=None, pos=None):
    """
    Constructs the dynamic graph (EdgeConv)[cite: 30, 208].
    If 'pos' is provided, neighbors are calculated based on geometry (XYZ)[cite: 289],
    but features are aggregated from 'x'.
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        if pos is not None:
            idx = knn(pos, k=k) # Fixed/Geometric Graph Context [cite: 289]
        else:
            idx = knn(x, k=k)   # Dynamic Graph Context [cite: 160, 270]
    
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x_trans = x.transpose(2, 1).contiguous()
    feature = x_trans.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # EdgeConv Logic: Concatenate (Neighbor - Center, Center) [cite: 257, 379]
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature   # Output Shape: (Batch, 2*Channels, N, k)


class DGCNN(nn.Module):
    def __init__(self, num_classes, k=40, dropout=0.5):
        super(DGCNN, self).__init__()
        self.k = k
        self.dropout = dropout
        
        # --- 1. THREE DISTINCT LITERALLY SPECIFIED EDGECONV LAYERS [cite: 414] ---
        
        # EdgeConv 1: mlp(64, 64) acting on 6D inputs (XYZ + Normals) [cite: 172]
        self.conv1 = nn.Sequential(nn.Conv2d(6*2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # EdgeConv 2: mlp(64, 64) acting on Layer 1 output [cite: 180]
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # EdgeConv 3: mlp(64) acting on Layer 2 output [cite: 182]
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # --- 2. SHARED GLOBAL AGGREGATION MULTI-LAYER PERCEPTRON (1024) [cite: 415] ---
        self.conv6 = nn.Sequential(nn.Conv1d(64, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # --- 3. THREE LITERALLY SPECIFIED SEGMENTATION HEAD FUALFCs (256, 256, 128) [cite: 417] ---
        # Feature Concat Size calculation[cite: 204, 416]: 
        # EdgeConv 1 (64) + EdgeConv 2 (64) + EdgeConv 3 (64) + Global Descriptor (1024) = 1212 Channels
        self.conv7 = nn.Sequential(nn.Conv1d(64 + 64 + 64 + 1024, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv8 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv9 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # Final Point-wise Classification Mapping Layer
        self.dp1 = nn.Dropout(p=self.dropout) # [cite: 418]
        self.conv10 = nn.Conv1d(128, num_classes, kernel_size=1, bias=True)

    def forward(self, pos, x):
        """
        Matches your unified training script signature exactly.
        Input:
            pos: [B, 3, N] (Raw spatial coordinates for anchoring graph lookups)
            x:   [B, 6, N] (Features: XYZ coordinate points + Normal vectors)
        Returns:
            Log-probabilities tensor layout and placeholder tracker matching: pred, _ = model(pos, x)
        """
        batch_size = x.size(0)
        num_points = x.size(2)

        # --- Layer 1: EdgeConv Block 1 (mlp: 64, 64) ---
        # Anchored on true geometry input positions 'pos' for stable spatial clustering [cite: 289]
        x_g1 = get_graph_feature(x, k=self.k, pos=pos)  # Shape: (Batch, 12, N, k)
        x_g1 = self.conv1(x_g1)                         # Shape: (Batch, 64, N, k)
        x_g1 = self.conv2(x_g1)                         # Shape: (Batch, 64, N, k)
        x1 = x_g1.max(dim=-1)[0]                        # Shape: (Batch, 64, N) [cite: 208, 267]

        # --- Layer 2: EdgeConv Block 2 (mlp: 64, 64) ---
        # Recomputed dynamically inside the shifting learning feature space [cite: 160, 270]
        x_g2 = get_graph_feature(x1, k=self.k)          # Shape: (Batch, 128, N, k)
        x_g2 = self.conv3(x_g2)                         # Shape: (Batch, 64, N, k)
        x_g2 = self.conv4(x_g2)                         # Shape: (Batch, 64, N, k)
        x2 = x_g2.max(dim=-1)[0]                        # Shape: (Batch, 64, N)

        # --- Layer 3: EdgeConv Block 3 (mlp: 64) ---
        x_g3 = get_graph_feature(x2, k=self.k)          # Shape: (Batch, 128, N, k)
        x_g3 = self.conv5(x_g3)                         # Shape: (Batch, 64, N, k)
        x3 = x_g3.max(dim=-1)[0]                        # Shape: (Batch, 64, N)

        # --- Layer 4: Global Shape Aggregation via Max Pooling (1024) [cite: 415] ---
        x_global = self.conv6(x3)                       # Shape: (Batch, 1024, N)
        x_global = x_global.max(dim=-1, keepdim=True)[0]# Shape: (Batch, 1024, 1) [cite: 204]
        x_global = x_global.repeat(1, 1, num_points)    # Shape: (Batch, 1024, N) [cite: 198]

        # --- Layer 5: Dense Multi-Scale Feature Shortcut Concatenation [cite: 204, 416] ---
        # Combines all layered localized descriptions with the global background context block
        # Total channels = 64 + 64 + 64 + 1024 = 1212 channels
        x_combined = torch.cat((x1, x2, x3, x_global), dim=1) # Shape: (Batch, 1212, N)

        # --- Layer 6: Segmentation Head Processing (256, 256, 128) [cite: 417] ---
        x_out = self.conv7(x_combined)                  # Shape: (Batch, 256, N)
        x_out = self.conv8(x_out)                       # Shape: (Batch, 256, N)
        x_out = self.conv9(x_out)                       # Shape: (Batch, 128, N)
        
        x_out = self.dp1(x_out)                         # Regularization Dropout [cite: 418]
        pred = self.conv10(x_out)                       # Shape: (Batch, num_classes, N)

        # Return log_softmax over the class dimension (dim=1) to feed into your nn.NLLLoss()
        return F.log_softmax(pred, dim=1), None