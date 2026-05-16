import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    """
    K-Nearest Neighbors implementation in pure PyTorch.
    Input:
        x: [B, 3, N] (We use geometry 'pos' to find neighbors)
        k: int (Number of neighbors)
    Output:
        idx: [B, N, k] (Indices of the nearest neighbors)
    """
    # Calculate Pairwise Distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    # Retrieve indices of the k nearest neighbors
    # (topk finds largest, so we use negative distance)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, pos=None):
    """
    Constructs the dynamic graph (EdgeConv).
    If 'pos' is provided, we calculate neighbors based on geometry (XYZ),
    but aggregate the features from 'x'.
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    # 1. Find Neighbors using Geometry (pos) if available, else Features (x)
    if idx is None:
        if pos is not None:
            idx = knn(pos, k=k) # Use XYZ for structure
        else:
            idx = knn(x, k=k)   # Use features for structure (Dynamic)
    
    device = x.device

    # 2. Prepare Indices for Gathering
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    # 3. Gather Neighbor Features
    # x: [B, C, N] -> [B, N, C]
    x_trans = x.transpose(2, 1).contiguous()
    
    # Flatten to gather: [B*N, C]
    feature = x_trans.view(batch_size*num_points, -1)[idx, :]
    
    # Reshape back: [B, N, k, C]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    
    # 4. Repeat Central Point Features
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # 5. EdgeConv Logic: Concatenate (Center, Neighbor - Center)
    # This captures both local structure and global position
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # Output: (batch_size, 2*num_dims, num_points, k)

class DGCNN(nn.Module):
    def __init__(self, num_classes=10, k=20, emb_dims=1024, dropout=0.5):
        super(DGCNN, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout
        
        # --- EdgeConv Layers ---
        # Layer 1: Input 6 (XYZ+Norm) -> Output 64
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(nn.Conv2d(6*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # Layer 2: Input 64 -> Output 64
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # Layer 3: Input 64 -> Output 128
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # Layer 4: Input 128 -> Output 256
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # --- Aggregation (Global Feature) ---
        self.bn5 = nn.BatchNorm1d(emb_dims)
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # --- Segmentation Head ---
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(128)
        
        # Input to head: 512 (Local Features) + 1024 (Global Feature) = 1536
        self.conv6 = nn.Sequential(nn.Conv1d(1536, 256, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv8 = nn.Conv1d(128, num_classes, kernel_size=1, bias=False)
        self.dp1 = nn.Dropout(p=dropout)

    def forward(self, pos, x):
        """
        Input:
            pos: [B, 3, N] (Geometry for neighbors)
            x:   [B, 6, N] (Features: XYZ + Normals)
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # 1. EdgeConv Block 1
        # We use 'pos' to find neighbors, but 'x' to build features
        x = get_graph_feature(x, k=self.k, pos=pos)      # (B, 12, N, k)
        x = self.conv1(x)                                # (B, 64, N, k)
        x1 = x.max(dim=-1, keepdim=False)[0]             # (B, 64, N)

        # 2. EdgeConv Block 2 (Dynamic: Neighbors based on previous features x1)
        x = get_graph_feature(x1, k=self.k)              # (B, 128, N, k)
        x = self.conv2(x)                                # (B, 64, N, k)
        x2 = x.max(dim=-1, keepdim=False)[0]             # (B, 64, N)

        # 3. EdgeConv Block 3
        x = get_graph_feature(x2, k=self.k)              # (B, 128, N, k)
        x = self.conv3(x)                                # (B, 128, N, k)
        x3 = x.max(dim=-1, keepdim=False)[0]             # (B, 128, N)

        # 4. EdgeConv Block 4
        x = get_graph_feature(x3, k=self.k)              # (B, 256, N, k)
        x = self.conv4(x)                                # (B, 256, N, k)
        x4 = x.max(dim=-1, keepdim=False)[0]             # (B, 256, N)

        # 5. Aggregate Local Features
        x_local = torch.cat((x1, x2, x3, x4), dim=1)     # (B, 64+64+128+256=512, N)

        # 6. Global Feature (Max Pooling over all points)
        x = self.conv5(x_local)                          # (B, 1024, N)
        x_global = x.max(dim=-1, keepdim=True)[0]        # (B, 1024, 1)
        
        # Expand Global Feature to match N points
        x_global = x_global.repeat(1, 1, num_points)     # (B, 1024, N)

        # 7. Concatenate Local + Global
        x = torch.cat((x_local, x_global), dim=1)        # (B, 1536, N)

        # 8. Segmentation Head
        x = self.conv6(x)
        x = self.dp1(x)
        x = self.conv7(x)
        x = self.conv8(x)
        
        # Return log_softmax for NLLLoss, and features (optional, for consistency)
        return F.log_softmax(x, dim=1), x