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
    Constructs the dynamic graph (EdgeConv).
    If 'pos' is provided, neighbors are calculated based on geometry (XYZ),
    but features are aggregated from 'x'.
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        if pos is not None:
            idx = knn(pos, k=k) # Fixed/Geometric Graph Context (first layer)
        else:
            idx = knn(x, k=k)   # Dynamic Graph Context (deeper layers)
    
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x_trans = x.transpose(2, 1).contiguous()
    feature = x_trans.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # EdgeConv Logic: Concatenate (Neighbor - Center, Center)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature   # Output Shape: (Batch, 2*Channels, N, k)


class DGCNN(nn.Module):
    """
    DGCNN for part segmentation.

    Architecture follows Wang et al. (2019) Sec. 4.4 and matches the official
    TensorFlow part-segmentation reference (WangYueFt/dgcnn) for the global
    aggregation and seg-head regularisation structure.

    Documented deviations from the official TF reference:
      (i)   k=40 nearest neighbours per EdgeConv (scaled from paper's k=20 at
            N=2048 to maintain fractional receptive field at N=4096).
      (ii)  Input Spatial Transformer (T-Net) omitted — consistent with the
            authors' own PyTorch port (WangYueFt/dgcnn/pytorch) and the de
            facto community part-seg reproduction (antao97/dgcnn.pytorch).
      (iii) Input features are 6-D (XYZ + surface normals) instead of 3-D,
            exploiting normals available in the Fusion360 dataset.
    """
    def __init__(self, num_classes, k=40, dropout=0.4):
        super(DGCNN, self).__init__()
        self.k = k
        self.dropout = dropout
        
        # --- 1. THREE EDGECONV LAYERS (matches paper Sec. 4.4) ---
        
        # EdgeConv 1: mlp(64, 64) acting on 6D inputs (XYZ + Normals)
        self.conv1 = nn.Sequential(nn.Conv2d(6*2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # EdgeConv 2: mlp(64, 64) acting on Layer 1 output
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # EdgeConv 3: mlp(64) acting on Layer 2 output
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # --- 2. GLOBAL AGGREGATION MLP (1024) ---
        # FIX (Deviation #3): The 1024-D global descriptor is computed from the
        # CONCATENATION of all three EdgeConv outputs, not just the last one.
        # This matches the official TF reference where:
        #   out7 = conv2d(concat(net_1, net_2, net_3), 1024, ...)
        # Input channels: 64 (x1) + 64 (x2) + 64 (x3) = 192
        self.conv6 = nn.Sequential(nn.Conv1d(64 + 64 + 64, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # --- 3. SEGMENTATION HEAD (256, 256, 128) ---
        # Feature Concat: x1 (64) + x2 (64) + x3 (64) + Global (1024) = 1216
        self.conv7 = nn.Sequential(nn.Conv1d(64 + 64 + 64 + 1024, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv8 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv9 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # FIX (Deviation #4): Two dropouts placed inside the seg head, after
        # conv7 and conv8 — matching the official TF reference's pattern:
        #   conv1 → dropout(0.4) → conv2 → dropout(0.4) → conv3 → conv4
        # Default rate 0.4 matches the official's keep_prob=0.6 (drop 40%).
        self.dp1 = nn.Dropout(p=self.dropout)
        self.dp2 = nn.Dropout(p=self.dropout)
        
        # Final Point-wise Classification Mapping Layer
        self.conv10 = nn.Conv1d(128, num_classes, kernel_size=1, bias=True)

    def forward(self, pos, x):
        """
        Matches your unified training script signature exactly.
        Input:
            pos: [B, 3, N] (Raw spatial coordinates for anchoring graph lookups)
            x:   [B, 6, N] (Features: XYZ coordinate points + Normal vectors)
        Returns:
            Log-probabilities tensor + None placeholder matching: pred, _ = model(pos, x)
        """
        batch_size = x.size(0)
        num_points = x.size(2)

        # --- Layer 1: EdgeConv Block 1 (mlp: 64, 64) ---
        # First layer uses input positions for kNN (more stable than feature kNN
        # when features are still raw XYZ + normals)
        x_g1 = get_graph_feature(x, k=self.k, pos=pos)  # Shape: (Batch, 12, N, k)
        x_g1 = self.conv1(x_g1)                         # Shape: (Batch, 64, N, k)
        x_g1 = self.conv2(x_g1)                         # Shape: (Batch, 64, N, k)
        x1 = x_g1.max(dim=-1)[0]                        # Shape: (Batch, 64, N)

        # --- Layer 2: EdgeConv Block 2 (mlp: 64, 64) ---
        # Dynamic graph: recompute kNN in feature space
        x_g2 = get_graph_feature(x1, k=self.k)          # Shape: (Batch, 128, N, k)
        x_g2 = self.conv3(x_g2)                         # Shape: (Batch, 64, N, k)
        x_g2 = self.conv4(x_g2)                         # Shape: (Batch, 64, N, k)
        x2 = x_g2.max(dim=-1)[0]                        # Shape: (Batch, 64, N)

        # --- Layer 3: EdgeConv Block 3 (mlp: 64) ---
        x_g3 = get_graph_feature(x2, k=self.k)          # Shape: (Batch, 128, N, k)
        x_g3 = self.conv5(x_g3)                         # Shape: (Batch, 64, N, k)
        x3 = x_g3.max(dim=-1)[0]                        # Shape: (Batch, 64, N)

        # --- Layer 4: Global Shape Aggregation via Max Pooling ---
        # FIX (Deviation #3): The 1024-D global descriptor sees features from
        # ALL THREE EdgeConv scales (matching the official TF reference), not
        # just x3 alone.
        x_multi = torch.cat((x1, x2, x3), dim=1)        # Shape: (Batch, 192, N)
        x_global = self.conv6(x_multi)                  # Shape: (Batch, 1024, N)
        x_global = x_global.max(dim=-1, keepdim=True)[0]# Shape: (Batch, 1024, 1)
        x_global = x_global.repeat(1, 1, num_points)    # Shape: (Batch, 1024, N)

        # --- Layer 5: Multi-Scale Feature Concatenation ---
        # Combines all layered local descriptors with the global context
        # Total channels = 64 + 64 + 64 + 1024 = 1216 channels
        x_combined = torch.cat((x1, x2, x3, x_global), dim=1) # Shape: (Batch, 1216, N)

        # --- Layer 6: Segmentation Head Processing (256, 256, 128) ---
        # FIX (Deviation #4): Two dropouts inside the seg head, matching the
        # official TF pattern. This is stronger and more distributed
        # regularisation than a single dropout at the end.
        x_out = self.conv7(x_combined)                  # Shape: (Batch, 256, N)
        x_out = self.dp1(x_out)                         # ← Dropout #1 (after conv7)
        x_out = self.conv8(x_out)                       # Shape: (Batch, 256, N)
        x_out = self.dp2(x_out)                         # ← Dropout #2 (after conv8)
        x_out = self.conv9(x_out)                       # Shape: (Batch, 128, N)
        
        pred = self.conv10(x_out)                       # Shape: (Batch, num_classes, N)

        # Return log_softmax over the class dimension (dim=1) for nn.NLLLoss()
        return F.log_softmax(pred, dim=1), None