"""
pt.py — Point Transformer V1 for Part Segmentation

Mathematically equivalent to Pointcept's PartSeg26 (Wu et al.), adapted to
[B, C, N] fixed-size tensor format for compatibility with the unified
training pipeline. Pure PyTorch — no pointops CUDA dependency.

PartSeg26 reference (Pointcept):
    https://github.com/Pointcept/Pointcept
    pointcept/models/point_transformer/point_transformer_partseg.py

Key mathematical components (all replicated here):
  - PointTransformerLayer with share_planes=8 channel-grouped vector attention
  - Position encoding with intermediate dim 3, BatchNorm (see note), sum-reduction
  - Three-layer γ (attn) MLP with input BN + middle BN
  - Bottleneck with three linear projections + BN+ReLU after pt_layer
  - TransitionDown with relative XYZ concatenated to features before MLP
  - TransitionUp with linear+BN+ReLU on both branches, raw sum (no ReLU)
  - dec5 with batch-mean aggregation at the deepest level (no shape categories)
  - Per-stage nsample [16,32,32,32,32] — scaled from PartSeg26's
    [8,16,16,16,16] for N=4096 (2x the paper's N=2048)
  - 32→32→num_classes output head with no dropout

Important note on PartSeg26's `LayerNorm1d`:
  Despite the name, PartSeg26's `LayerNorm1d` (defined in pointcept's utils.py)
  inherits from nn.BatchNorm1d, NOT nn.LayerNorm. Its forward transposes the
  input from [N,k,C] to [N,C,k], applies BatchNorm1d (normalizing per-channel
  across all (N,k) elements), then transposes back.

  For [B,C,N,k] tensors in this rewrite, the equivalent is nn.BatchNorm2d,
  which normalizes per-channel across (B,N,k) — semantically identical to
  PartSeg26's BN-over-(N_total,k) per channel.

Tensor format note:
  PartSeg26 uses stacked variable-batch format [N_total, C] with offsets.
  This file uses [B, C, N] fixed-size format. All operations are
  mathematically equivalent — the format change does not affect math.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Geometric utilities (pure PyTorch equivalents of pointops kernels)
# ─────────────────────────────────────────────────────────────────────────────

def knn(x, y, k):
    """
    k-Nearest Neighbors by squared Euclidean distance.

    Args
    ────
    x : [B, 3, N]  query points
    y : [B, 3, M]  source points
    k : int

    Returns
    ───────
    idx : [B, N, k]  indices into y for each query in x
    """
    num_available = y.shape[2]
    actual_k = min(k, num_available)

    # ||x-y||² = ||x||² - 2 x·y + ||y||²
    inner = -2.0 * torch.matmul(x.transpose(2, 1), y)                 # [B, N, M]
    xx = (x ** 2).sum(dim=1, keepdim=True).transpose(2, 1)             # [B, N, 1]
    yy = (y ** 2).sum(dim=1, keepdim=True)                             # [B, 1, M]
    sq_dist = xx + inner + yy                                          # [B, N, M]

    # Smallest distances → nearest neighbors
    idx = sq_dist.topk(actual_k, dim=-1, largest=False)[1]            # [B, N, k]
    return idx


def index_points(points, idx):
    """
    Gather features by indices.

    Args
    ────
    points : [B, C, N]
    idx    : [B, M, k]

    Returns
    ───────
    [B, C, M, k]
    """
    B, C, N = points.shape
    points_t = points.transpose(1, 2).contiguous()                    # [B, N, C]
    batch_idx = torch.arange(B, dtype=torch.long, device=points.device).view(-1, 1, 1)
    gathered = points_t[batch_idx, idx, :]                            # [B, M, k, C]
    return gathered.permute(0, 3, 1, 2).contiguous()                  # [B, C, M, k]


def farthest_point_sample(xyz, npoint):
    """
    Iterative Farthest Point Sampling.

    Args
    ────
    xyz    : [B, 3, N]
    npoint : int

    Returns
    ───────
    centroid_idx : [B, npoint]
    """
    device = xyz.device
    B, _, N = xyz.shape
    xyz = xyz.transpose(1, 2).contiguous()                            # [B, N, 3]

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.max(dim=-1)[1]

    return centroids


# ─────────────────────────────────────────────────────────────────────────────
# PointTransformerLayer — equivalent to PartSeg26's PointTransformerLayer
# ─────────────────────────────────────────────────────────────────────────────

class PointTransformerLayer(nn.Module):
    """
    Vector self-attention with channel-grouped attention weights.

    PartSeg26 reference: PointTransformerLayer in point_transformer_partseg.py

    Args
    ────
    channels      : feature dimension (called out_planes in PartSeg26)
    k             : neighborhood size (called nsample in PartSeg26)
    share_planes  : channels per attention group (channels // share_planes attention weights)
    """

    def __init__(self, channels, k=16, share_planes=8):
        super().__init__()
        self.channels = channels
        self.k = k
        self.share_planes = share_planes
        # PartSeg26 sets mid_planes = out_planes // 1 = out_planes (no reduction).
        self.mid_planes = channels

        # Q, K, V projections — bias=True matches PartSeg26's nn.Linear defaults.
        # In PartSeg26: linear_q, linear_k, linear_v.
        self.linear_q = nn.Conv1d(channels, self.mid_planes, 1, bias=True)
        self.linear_k = nn.Conv1d(channels, self.mid_planes, 1, bias=True)
        self.linear_v = nn.Conv1d(channels, channels, 1, bias=True)

        # Position encoding MLP — matches PartSeg26's linear_p.
        # PartSeg26: Linear(3,3) → LayerNorm1d(3) → ReLU → Linear(3, out_planes).
        # Intermediate dim is 3 (not channels).
        #
        # IMPORTANT: PartSeg26's `LayerNorm1d` is misleadingly named — it actually
        # inherits from `nn.BatchNorm1d` (not nn.LayerNorm). Its forward transposes
        # the input then applies BatchNorm. For our [B,3,N,k] tensors, the
        # equivalent operation is plain nn.BatchNorm2d, which normalizes per-channel
        # across all (B,N,k) elements — semantically identical to PartSeg26's
        # BN-over-(N_total,k) per channel.
        self.linear_p = nn.Sequential(
            nn.Conv2d(3, 3, 1, bias=True),
            nn.BatchNorm2d(3),                # equivalent to PartSeg26's LayerNorm1d(3)
            nn.ReLU(inplace=True),
            nn.Conv2d(3, channels, 1, bias=True),
        )

        # γ attention MLP — matches PartSeg26's linear_w.
        # PartSeg26: LN(mid) → ReLU → Linear(mid, out/share) → LN(out/share) → ReLU
        #            → Linear(out/share, out/share)
        # Output dim is channels // share_planes (one attention vector per group).
        #
        # Note: PartSeg26's `LayerNorm1d` is BatchNorm in disguise (see linear_p
        # comment above). The equivalent for [B,C,N,k] tensors is nn.BatchNorm2d.
        attn_out = channels // share_planes
        self.linear_w = nn.Sequential(
            nn.BatchNorm2d(self.mid_planes),  # equivalent to PartSeg26's LayerNorm1d(mid_planes)
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_planes, attn_out, 1, bias=True),
            nn.BatchNorm2d(attn_out),          # equivalent to PartSeg26's LayerNorm1d(attn_out)
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_out, attn_out, 1, bias=True),
        )

        # Softmax over the neighborhood dimension k.
        # In PartSeg26's flat format this is dim=1; in our [B,C,N,k] format it's dim=-1.

    def forward(self, pos, x):
        """
        Args:
            pos : [B, 3, N]
            x   : [B, C, N]
        Returns:
            [B, C, N]
        """
        B, C, N = x.shape
        k = min(self.k, N)

        # Q, K, V projections.
        q = self.linear_q(x)                            # [B, C, N]
        kf = self.linear_k(x)                           # [B, C, N]
        v = self.linear_v(x)                            # [B, C, N]

        # k-NN in position space.
        idx = knn(pos, pos, k)                          # [B, N, k]

        # Gather K and V for each neighborhood.
        k_neigh = index_points(kf, idx)                 # [B, C, N, k]
        v_neigh = index_points(v, idx)                  # [B, C, N, k]

        # Relative position encoding.
        # PartSeg26 uses (neighbor - query) sign convention via pointops with_xyz=True.
        # The learned MLP absorbs sign conventions, so either sign works.
        pos_self = pos.unsqueeze(-1)                    # [B, 3, N, 1]
        pos_neigh = index_points(pos, idx)              # [B, 3, N, k]
        rel_pos = pos_neigh - pos_self                  # [B, 3, N, k]  (neighbor - query)
        p_r = self.linear_p(rel_pos)                    # [B, C, N, k]

        # Sum-reduction of position encoding for attention computation.
        # PartSeg26: einops.reduce(p_r, "n ns (i j) -> n ns j", "sum", j=mid_planes)
        # This reduces out_planes channels to mid_planes by summing groups.
        # Since we have mid_planes = channels, the reduction is over 1-element groups
        # (i.e., identity). But we implement the general case explicitly: reshape to
        # [B, mid_planes, groups, N, k], sum over groups, get [B, mid_planes, N, k].
        # Here groups = channels // mid_planes = 1, so this is just p_r unchanged.
        groups = self.channels // self.mid_planes        # = 1 with our setup
        if groups == 1:
            p_r_reduced = p_r                            # [B, mid_planes, N, k]
        else:
            p_r_reduced = p_r.view(B, self.mid_planes, groups, N, k).sum(dim=2)

        # Attention input: K - Q + position_encoding_reduced.
        # PartSeg26: r_qk = x_k - x_q.unsqueeze(1) + p_r_reduced
        q_expanded = q.unsqueeze(-1)                    # [B, C, N, 1]
        r_qk = k_neigh - q_expanded + p_r_reduced       # [B, mid_planes, N, k]

        # Apply γ MLP → attention weights of shape [B, attn_out, N, k].
        # attn_out = channels // share_planes.
        w = self.linear_w(r_qk)                         # [B, channels//share_planes, N, k]

        # Softmax over neighbors (last dim in our format).
        # PartSeg26 uses dim=1 in flat [N, k, C] format; equivalent to dim=-1 here.
        w = F.softmax(w, dim=-1)                        # [B, channels//share_planes, N, k]

        # Value path: V + position encoding (full, not reduced).
        v_with_pos = v_neigh + p_r                      # [B, C, N, k]

        # Grouped attention application.
        # PartSeg26 reshapes V to [N, k, share_planes, channels/share_planes] and
        # applies attention via einsum so each group of share_planes channels
        # shares the same attention weights.
        # In our [B, C, N, k] format: reshape C → (share_planes, attn_out_per_share)
        # where attn_out_per_share = channels // share_planes.
        attn_out_per_share = self.channels // self.share_planes
        v_with_pos_reshape = v_with_pos.view(B, self.share_planes, attn_out_per_share, N, k)
        # v_with_pos_reshape : [B, share_planes, attn_out_per_share, N, k]

        # Weight broadcast: w has shape [B, attn_out_per_share, N, k].
        # Each of the share_planes channel groups uses the same w.
        w_expanded = w.unsqueeze(1)                     # [B, 1, attn_out_per_share, N, k]

        # Apply attention: weighted sum over k.
        out = (v_with_pos_reshape * w_expanded).sum(dim=-1)  # [B, share_planes, attn_out_per_share, N]

        # Reshape back to [B, C, N].
        out = out.view(B, self.channels, N)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck — equivalent to PartSeg26's Bottleneck class
# ─────────────────────────────────────────────────────────────────────────────

class Bottleneck(nn.Module):
    """
    Three-linear residual block with PT layer in the middle.

    PartSeg26 reference: Bottleneck class.
    Structure: linear1 → bn1 → ReLU → pt_layer → bn2 → ReLU → linear3 → bn3 → (+identity) → ReLU
    """

    def __init__(self, channels, k=16, share_planes=8):
        super().__init__()
        # PartSeg26: linear1 = Linear(in_planes, planes, bias=False), bn1 = BN1d(planes)
        # With expansion=1, all dims are equal.
        self.linear1 = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)

        # PartSeg26: self.transformer = PointTransformerLayer(planes, planes, ...)
        self.transformer = PointTransformerLayer(channels, k=k, share_planes=share_planes)
        # PartSeg26: bn2 after transformer.
        self.bn2 = nn.BatchNorm1d(channels)

        # PartSeg26: linear3 = Linear(planes, planes * expansion, bias=False), expansion=1.
        self.linear3 = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, pos, x):
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer(pos, x)))
        x = self.bn3(self.linear3(x))
        x = self.relu(x + identity)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# TransitionDown — equivalent to PartSeg26's TransitionDown
# ─────────────────────────────────────────────────────────────────────────────

class TransitionDown(nn.Module):
    """
    Downsamples points via FPS and gathers local neighborhoods with relative XYZ.

    PartSeg26 reference: TransitionDown class.

    For stride > 1:
        FPS → k-NN gather (with relative XYZ concatenated to features)
            → linear(3+in → out) + BN + ReLU → max-pool over neighbors

    For stride = 1:
        linear(in → out) + BN + ReLU (no gather, no pooling)
    """

    def __init__(self, in_channels, out_channels, stride, nsample, npoint_out=None):
        super().__init__()
        assert stride >= 1
        self.stride = stride
        self.nsample = nsample
        self.npoint_out = npoint_out          # number of points after FPS (used when stride > 1)

        if stride != 1:
            # PartSeg26: Linear(3 + in_planes, out_planes, bias=False).
            # The +3 is for relative XYZ concatenation.
            self.linear = nn.Conv2d(3 + in_channels, out_channels, 1, bias=False)
        else:
            self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.bn = nn.BatchNorm1d(out_channels) if stride == 1 else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pos, x):
        """
        Args:
            pos : [B, 3, N]
            x   : [B, C, N]
        Returns:
            new_pos : [B, 3, M]  (M = npoint_out if stride>1, else N)
            new_x   : [B, C_out, M]
        """
        if self.stride == 1:
            # Simple linear projection at the same resolution.
            return pos, self.relu(self.bn(self.linear(x)))

        B, _, N = pos.shape
        M = self.npoint_out

        # FPS to choose M centroids.
        fps_idx = farthest_point_sample(pos, M)                       # [B, M]
        new_pos = index_points(pos, fps_idx.unsqueeze(-1)).squeeze(-1) # [B, 3, M]

        # k-NN from M new centroids into N old points.
        nb_idx = knn(new_pos, pos, self.nsample)                       # [B, M, nsample]

        # Gather neighbor features and neighbor positions.
        nb_x = index_points(x, nb_idx)                                 # [B, C_in, M, nsample]
        nb_pos = index_points(pos, nb_idx)                             # [B, 3, M, nsample]

        # Relative positions: neighbor - centroid.
        # PartSeg26's pointops returns this directly via with_xyz=True.
        rel_pos = nb_pos - new_pos.unsqueeze(-1)                       # [B, 3, M, nsample]

        # Concatenate relative XYZ with features along channel axis.
        nb_combined = torch.cat([rel_pos, nb_x], dim=1)                # [B, 3+C_in, M, nsample]

        # Linear projection + BN + ReLU.
        nb_combined = self.linear(nb_combined)                         # [B, C_out, M, nsample]
        nb_combined = self.relu(self.bn(nb_combined))

        # Max-pool over neighbors.
        new_x = nb_combined.max(dim=-1)[0]                             # [B, C_out, M]

        return new_pos, new_x


# ─────────────────────────────────────────────────────────────────────────────
# TransitionUp — equivalent to PartSeg26's TransitionUp (non-head variant)
# ─────────────────────────────────────────────────────────────────────────────

class TransitionUp(nn.Module):
    """
    Upsamples low-res features to high-res positions via 3-NN IDW interpolation,
    adds high-res skip connection.

    PartSeg26 reference: TransitionUp class (the `out_planes is not None` branch).

    For the non-head variant:
        out = linear1(x_skip) + 3NN_interp(linear2(x_lowres))
    No ReLU after the sum.
    """

    def __init__(self, in_channels_low, in_channels_skip, out_channels):
        super().__init__()
        # PartSeg26: linear1 on the skip (high-res) input.
        self.linear1 = nn.Sequential(
            nn.Conv1d(in_channels_skip, out_channels, 1, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        # PartSeg26: linear2 on the low-res input before interpolation.
        self.linear2 = nn.Sequential(
            nn.Conv1d(in_channels_low, out_channels, 1, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, pos_high, x_high, pos_low, x_low):
        """
        Args:
            pos_high : [B, 3, N_high]  high-res skip positions
            x_high   : [B, C_skip, N_high]  high-res skip features
            pos_low  : [B, 3, N_low]  low-res positions
            x_low    : [B, C_low, N_low]  low-res features
        Returns:
            [B, C_out, N_high]
        """
        # Project low-res features.
        x_low_proj = self.linear2(x_low)                              # [B, C_out, N_low]

        # 3-NN interpolation: find 3 nearest low-res points for each high-res point.
        idx = knn(pos_high, pos_low, k=3)                              # [B, N_high, 3]

        # Get the positions of those 3 NNs and compute distances.
        pos_low_gathered = index_points(pos_low, idx)                  # [B, 3, N_high, 3]
        diff = pos_high.unsqueeze(-1) - pos_low_gathered               # [B, 3, N_high, 3]
        dist = diff.norm(dim=1)                                        # [B, N_high, 3]

        # Inverse-distance weights.
        dist_recip = 1.0 / (dist + 1e-8)                               # [B, N_high, 3]
        norm = dist_recip.sum(dim=-1, keepdim=True)                    # [B, N_high, 1]
        weight = dist_recip / norm                                     # [B, N_high, 3]

        # Gather features of the 3 NNs and apply weights.
        x_low_gathered = index_points(x_low_proj, idx)                 # [B, C_out, N_high, 3]
        interp = (x_low_gathered * weight.unsqueeze(1)).sum(dim=-1)   # [B, C_out, N_high]

        # Skip connection: linear1 on x_high, then sum.
        x_high_proj = self.linear1(x_high)                             # [B, C_out, N_high]

        # PartSeg26: raw sum, no ReLU.
        return x_high_proj + interp


# ─────────────────────────────────────────────────────────────────────────────
# Dec5Head — equivalent to PartSeg26's `dec5[0]` (TransitionUp with out_planes=None)
# ─────────────────────────────────────────────────────────────────────────────

class Dec5Head(nn.Module):
    """
    Special "head" transition used at the deepest level of the decoder.

    PartSeg26 reference: TransitionUp class (the `out_planes is None` branch),
    used as the first layer of dec5 (with is_head=True in _make_dec).

    For each batch:
      - Compute the mean of features over all points in that batch.
      - Project the mean via linear2.
      - Concatenate to each point's features [x, projected_mean].
      - Apply linear1 to get back to the original channel count.

    No shape category conditioning (num_shape_classes=None branch in PartSeg26).
    """

    def __init__(self, channels):
        super().__init__()
        # PartSeg26 (num_shape_class is None branch):
        # linear1: Linear(2*in_planes, in_planes) + BN + ReLU
        # linear2: Linear(in_planes, in_planes) + ReLU (no BN)
        self.linear1 = nn.Sequential(
            nn.Conv1d(2 * channels, channels, 1, bias=True),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.linear2 = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, pos, x):
        """
        Args:
            pos : [B, 3, N]  (not used here; kept for interface consistency)
            x   : [B, C, N]
        Returns:
            [B, C, N]
        """
        B, C, N = x.shape

        # Per-batch mean over points.
        x_mean = x.mean(dim=-1, keepdim=True)                         # [B, C, 1]

        # Project mean.
        x_mean_proj = self.linear2(x_mean)                            # [B, C, 1]

        # Broadcast to N points and concatenate with original features.
        x_mean_broadcast = x_mean_proj.expand(-1, -1, N)               # [B, C, N]
        x_cat = torch.cat([x, x_mean_broadcast], dim=1)                # [B, 2C, N]

        # Project back to C channels.
        out = self.linear1(x_cat)                                      # [B, C, N]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Top-level network — equivalent to PartSeg26's PointTransformerSeg
# ─────────────────────────────────────────────────────────────────────────────

class PointTransformer(nn.Module):
    """
    PartSeg26-equivalent Point Transformer V1 for part segmentation.

    Structure (matching PartSeg26 with blocks=[1,1,1,1,1]):

      Encoder:
        enc1: TransitionDown(stride=1, 6→32)    + 1 × Bottleneck(32)
        enc2: TransitionDown(stride=4, 32→64)   + 1 × Bottleneck(64)
        enc3: TransitionDown(stride=4, 64→128)  + 1 × Bottleneck(128)
        enc4: TransitionDown(stride=4, 128→256) + 1 × Bottleneck(256)
        enc5: TransitionDown(stride=4, 256→512) + 1 × Bottleneck(512)

      Decoder:
        dec5: Dec5Head(512) + 1 × Bottleneck(512)   (batch-mean aggregation at deepest level)
        dec4: TransitionUp(512→256) + 1 × Bottleneck(256)
        dec3: TransitionUp(256→128) + 1 × Bottleneck(128)
        dec2: TransitionUp(128→64)  + 1 × Bottleneck(64)
        dec1: TransitionUp(64→32)   + 1 × Bottleneck(32)

      Output head:
        Linear(32→32) + BN + ReLU + Linear(32→num_classes)
        (No dropout.)

    Per-stage nsample [16, 32, 32, 32, 32] — scaled from PartSeg26's
    [8, 16, 16, 16, 16] for N=4096 input points (paper uses N=2048).
    """

    def __init__(self, num_classes, num_points=4096, k=None):
        """
        Args:
            num_classes : output classes for segmentation
            num_points  : input point count (default 4096)
            k           : optional override — if provided, used as nsample at
                          every stage. If None (default), nsample is auto-scaled:
                              N=2048  → [8, 16, 16, 16, 16]  (PartSeg26 literal)
                              N=4096  → [16, 32, 32, 32, 32] (scaled 2x)
                              other N → linearly scaled from PartSeg26 base
                          The scaling preserves the same fractional receptive
                          field per stage across different point densities.
        """
        super().__init__()
        self.num_points = num_points

        # Channel progression matching PartSeg26.
        planes = [32, 64, 128, 256, 512]
        # PartSeg26 uses stride [1, 4, 4, 4, 4]: first stage is stride-1.
        strides = [1, 4, 4, 4, 4]

        # Per-stage nsample, auto-scaled with num_points.
        # PartSeg26's calibration at N=2048 is [8, 16, 16, 16, 16].
        # We scale linearly with sqrt(N/2048) rounded to nearest power-of-2-friendly value.
        # For the two common cases:
        #   N=2048: [8, 16, 16, 16, 16]
        #   N=4096: [16, 32, 32, 32, 32]
        if k is None:
            if num_points == 2048:
                nsamples = [8, 16, 16, 16, 16]   # PartSeg26 literal
            elif num_points == 4096:
                nsamples = [16, 32, 32, 32, 32]  # scaled 2x for higher density
            else:
                # General scaling: round to nearest power of 2.
                scale = num_points / 2048.0
                base = [8, 16, 16, 16, 16]
                nsamples = [max(4, int(round(b * scale))) for b in base]
        else:
            nsamples = [k] * 5
        share_planes = 8                  # PartSeg26 default

        # Compute the point count at each encoder stage.
        # enc1: N (stride 1, no FPS)
        # enc2: N/4
        # enc3: N/16
        # enc4: N/64
        # enc5: N/256
        npts = [num_points]
        for s in strides[1:]:
            npts.append(npts[-1] // s)

        # ── Encoder stages ────────────────────────────────────────────
        # Each stage = TransitionDown + Bottleneck (matching _make_enc in PartSeg26).
        # Input feature dim = 6 (XYZ + normals).
        self.enc1_td = TransitionDown(6, planes[0], strides[0], nsamples[0])
        self.enc1_block = Bottleneck(planes[0], k=nsamples[0], share_planes=share_planes)

        self.enc2_td = TransitionDown(planes[0], planes[1], strides[1], nsamples[1],
                                       npoint_out=npts[1])
        self.enc2_block = Bottleneck(planes[1], k=nsamples[1], share_planes=share_planes)

        self.enc3_td = TransitionDown(planes[1], planes[2], strides[2], nsamples[2],
                                       npoint_out=npts[2])
        self.enc3_block = Bottleneck(planes[2], k=nsamples[2], share_planes=share_planes)

        self.enc4_td = TransitionDown(planes[2], planes[3], strides[3], nsamples[3],
                                       npoint_out=npts[3])
        self.enc4_block = Bottleneck(planes[3], k=nsamples[3], share_planes=share_planes)

        self.enc5_td = TransitionDown(planes[3], planes[4], strides[4], nsamples[4],
                                       npoint_out=npts[4])
        self.enc5_block = Bottleneck(planes[4], k=nsamples[4], share_planes=share_planes)

        # ── Decoder stages ────────────────────────────────────────────
        # dec5: special head (Dec5Head with batch-mean aggregation) + Bottleneck.
        # Operates at the deepest level (N/256) BEFORE any upsampling begins.
        self.dec5_head = Dec5Head(planes[4])
        self.dec5_block = Bottleneck(planes[4], k=nsamples[4], share_planes=share_planes)

        # dec4..dec1: TransitionUp + Bottleneck.
        # Each TransitionUp(in_low, in_skip, out): from lower-res to higher-res.
        self.dec4_up = TransitionUp(planes[4], planes[3], planes[3])
        self.dec4_block = Bottleneck(planes[3], k=nsamples[3], share_planes=share_planes)

        self.dec3_up = TransitionUp(planes[3], planes[2], planes[2])
        self.dec3_block = Bottleneck(planes[2], k=nsamples[2], share_planes=share_planes)

        self.dec2_up = TransitionUp(planes[2], planes[1], planes[1])
        self.dec2_block = Bottleneck(planes[1], k=nsamples[1], share_planes=share_planes)

        self.dec1_up = TransitionUp(planes[1], planes[0], planes[0])
        self.dec1_block = Bottleneck(planes[0], k=nsamples[0], share_planes=share_planes)

        # ── Output head ───────────────────────────────────────────────
        # PartSeg26's self.cls: Linear(32,32) + BN + ReLU + Linear(32, num_classes).
        # No dropout.
        self.cls = nn.Sequential(
            nn.Conv1d(planes[0], planes[0], 1, bias=True),
            nn.BatchNorm1d(planes[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes[0], num_classes, 1, bias=True),
        )

    def forward(self, pos, x):
        """
        Args:
            pos : [B, 3, N]   raw XYZ
            x   : [B, 6, N]   features (XYZ + normals)
        Returns:
            log_probs : [B, num_classes, N]
            None      : placeholder, matches train_universal.py signature
        """
        # ── Encoder ───────────────────────────────────────────────────
        # enc1: stride=1, no FPS, just MLP from 6 → 32.
        # PartSeg26: enc1's TransitionDown receives x0 (features=6-D), produces 32-D.
        p1, x1 = self.enc1_td(pos, x)
        x1 = self.enc1_block(p1, x1)

        p2, x2 = self.enc2_td(p1, x1)
        x2 = self.enc2_block(p2, x2)

        p3, x3 = self.enc3_td(p2, x2)
        x3 = self.enc3_block(p3, x3)

        p4, x4 = self.enc4_td(p3, x3)
        x4 = self.enc4_block(p4, x4)

        p5, x5 = self.enc5_td(p4, x4)
        x5 = self.enc5_block(p5, x5)

        # ── Decoder ───────────────────────────────────────────────────
        # dec5: Dec5Head (batch-mean) + Bottleneck, both at the deepest level (N/256).
        x5 = self.dec5_head(p5, x5)
        x5 = self.dec5_block(p5, x5)

        # dec4: upsample N/256 → N/64, then Bottleneck.
        x4 = self.dec4_up(p4, x4, p5, x5)
        x4 = self.dec4_block(p4, x4)

        # dec3: N/64 → N/16.
        x3 = self.dec3_up(p3, x3, p4, x4)
        x3 = self.dec3_block(p3, x3)

        # dec2: N/16 → N/4.
        x2 = self.dec2_up(p2, x2, p3, x3)
        x2 = self.dec2_block(p2, x2)

        # dec1: N/4 → N.
        x1 = self.dec1_up(p1, x1, p2, x2)
        x1 = self.dec1_block(p1, x1)

        # ── Output head ──────────────────────────────────────────────
        logits = self.cls(x1)                                          # [B, num_classes, N]

        return F.log_softmax(logits, dim=1), None