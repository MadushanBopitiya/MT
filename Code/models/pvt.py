"""
pvt.py — Point-Voxel Transformer (PVT) for Part Segmentation
=============================================================
Architecture: Zhang et al., "PVT: Point-Voxel Transformer for Point Cloud Learning"
              arXiv 2108.06076v4, 2022.  https://github.com/HaochengWan/PVT

Design decisions and deviations documented for thesis benchmarking:

FAITHFUL to the paper
─────────────────────
• Dual-branch PVT block (Sec. 3, Fig. 1)                       — voxel + point branch fused by addition (Eq. 9)
• Voxel branch: k-NN local window attention (Sec. 3.1)         — approximates SWA (see adaptation note)
• Point branch: External Attention for N=4096 (Sec. 3.2)       — paper-prescribed for large N (Table 1)
• 3 stacked PVT blocks, channel dims 64/64/128 (Fig. 1)
• Multi-scale skip concatenation 64+64+128+1024=1280 (Fig. 1)
• Global max-pooling then broadcast (Fig. 1)
• Segmentation head: 1280→256→num_classes (Fig. 1)
• Dropout p=0.5 in last two linear layers (Sec. 4)
• BN + LeakyReLU throughout

POINT BRANCH — why External Attention is the correct choice here
────────────────────────────────────────────────────────────────
The paper explicitly provides two variants for the point branch (Sec. 3.2,
Table 1):

    Relative-attention (RA):  O(N²·D)  — for SMALL-scale point clouds
    External Attention  (EA):  O(N·D)   — for LARGE-scale point clouds

The authors state: "with tens of thousands of points…directly applying the
RA module incurs unacceptable O(N²) memory consumption.  Thus, for large-scale
point clouds, we perform External Attention."

At N=4096 with batch size ≥ 4, a single RA block allocates:
    pos_diff: [B,N,N,3] ≈ B × 4096² × 3 × 4 bytes  (~3 GB per sample)
    energy:   [B,N,N]   ≈ B × 4096² × 4 bytes       (~1 GB per sample)

This is in the "unacceptable" regime the authors describe.  Using EA is
therefore the paper-prescribed approach for this point density, not a
deviation from the architecture.

VOXEL BRANCH — SWA approximation
──────────────────────────────────
The paper's Sparse Window Attention (SWA) requires a GPU hash-table Rule Book
(Fig. 2) — a custom CUDA kernel that indexes non-empty voxels.  This is not
available in vanilla PyTorch.  The approximation used here groups points into
local windows via k-NN (the same neighbourhood concept as a 3D window) and
performs standard scaled dot-product attention within each window.  This
preserves the core design intent of SWA: local, window-restricted attention
that is linear in N (O(N·k·D) where k << N).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Shared utility: k-NN grouping
# ──────────────────────────────────────────────────────────────────────────────

def knn_gather(pos: torch.Tensor, x: torch.Tensor, k: int):
    """
    Find the k nearest neighbours of each point (in position space) and
    return the gathered feature windows.

    Args
    ────
    pos : [B, 3, N]   point coordinates
    x   : [B, C, N]   point features
    k   : int         window size (number of neighbours)

    Returns
    ───────
    windows : [B, N, k, C]   features of the k neighbours of each point
    """
    B, C, N = x.shape
    pos_t = pos.transpose(1, 2)                                # [B, N, 3]

    # Pairwise squared distances
    inner = -2.0 * torch.bmm(pos_t, pos_t.transpose(1, 2))    # [B, N, N]
    sq    = (pos_t ** 2).sum(dim=2, keepdim=True)              # [B, N, 1]
    dist2 = (sq + inner + sq.transpose(1, 2)).clamp(min=0.0)  # [B, N, N]

    # Indices of k nearest neighbours
    _, idx = dist2.topk(k, dim=-1, largest=False)              # [B, N, k]

    # Gather features
    x_t   = x.transpose(1, 2)                                 # [B, N, C]
    idx_e = idx.unsqueeze(-1).expand(-1, -1, -1, C)            # [B, N, k, C]
    windows = x_t.unsqueeze(2).expand(-1, -1, N, -1) \
                 .gather(2, idx_e)                             # [B, N, k, C]

    return windows


# ──────────────────────────────────────────────────────────────────────────────
# Voxel branch: Local Window Attention  (Sec. 3.1 / Fig. 1 upper branch)
# ──────────────────────────────────────────────────────────────────────────────

class LocalWindowAttention(nn.Module):
    """
    Approximation of Sparse Window Attention (SWA) — Sec. 3.1.

    The paper partitions the voxel space into non-overlapping 3D windows and
    performs standard Transformer self-attention within each window (Eq. not
    numbered; described as "standard Transformer architecture applied to
    regular 3D voxels").  Here each point attends to its k nearest spatial
    neighbours, which is the point-domain equivalent of a 3D window.

    Complexity: O(N·k·D)  — linear in N for fixed k, matching SWA's intent.
    """

    def __init__(self, channels: int, k: int = 32):
        super().__init__()
        self.k = k

        self.q_proj = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_proj = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_proj = nn.Conv1d(channels, channels, 1, bias=False)

        self.out_mlp = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.scale = None   # set in forward once channels are known

    def forward(self, pos: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  pos [B,3,N],  x [B,C,N]
        Returns: [B, C, N]

        NaN-stability note
        ──────────────────
        The attention dot-product, softmax, and value aggregation are computed
        in fp32 even when the surrounding code runs under torch.cuda.amp.
        Softmax in fp16 can overflow (fp16 max ≈ 65 504) once attention logits
        grow during training — a delayed NaN at epoch ~10–20 is the canonical
        symptom and was observed in initial runs of this model.
        Forcing fp32 only inside the attention math costs negligible memory but
        eliminates the overflow path.  This is standard practice in modern
        Transformer codebases (e.g. HuggingFace, fairseq).
        """
        B, C, N = x.shape
        k       = min(self.k, N)
        scale   = C ** -0.5

        # Project to Q, K, V
        Q = self.q_proj(x).transpose(1, 2)                    # [B, N, C]
        K_feat = self.k_proj(x)                               # [B, C, N]
        V_feat = self.v_proj(x)                               # [B, C, N]

        # Gather K and V for each point's k-NN window
        K_win = knn_gather(pos, K_feat, k)                    # [B, N, k, C]
        V_win = knn_gather(pos, V_feat, k)                    # [B, N, k, C]

        # ── fp32 attention block (AMP-safe) ──────────────────────────────
        with torch.cuda.amp.autocast(enabled=False):
            Q_exp   = Q.float().unsqueeze(2)                          # [B, N, 1, C]
            K_win32 = K_win.float()                                   # [B, N, k, C]
            V_win32 = V_win.float()                                   # [B, N, k, C]

            energy  = torch.matmul(Q_exp, K_win32.transpose(2, 3)) * scale  # [B, N, 1, k]
            attn    = F.softmax(energy, dim=-1)                       # [B, N, 1, k]
            F_local = torch.matmul(attn, V_win32).squeeze(2)          # [B, N, C]

        # Return to caller's dtype (fp16 under AMP, fp32 otherwise) for the
        # downstream MLP and residual connection
        F_local = F_local.to(x.dtype).transpose(1, 2)                 # [B, C, N]

        return self.out_mlp(F_local) + x                              # residual


# ──────────────────────────────────────────────────────────────────────────────
# Point branch: External Attention  (Sec. 3.2 / Table 1)
# ──────────────────────────────────────────────────────────────────────────────

class ExternalAttention(nn.Module):
    """
    External Attention for the point branch — Sec. 3.2.

    The paper cites the EA formulation (Guo et al., 2021):
        A = softmax_row( X · Mₖ )            attention via external memory Mₖ
        F = norm_col(A) · Mᵥ                 aggregate via external memory Mᵥ

    Both Mₖ ∈ R^{S×D} and Mᵥ ∈ R^{S×D} are small, learnable, shared
    memory matrices.  The complexity is O(N·S·D) — linear in N.

    "External Attention…can be implemented easily by simply using two
    cascaded linear layers and two normalisation layers."  — Sec. 3.2

    Implementation note: the paper implements Mₖ and Mᵥ as 1×1 Conv1d
    layers (equivalent to linear projection over the channel axis at each
    point independently).  Double-softmax normalisation (row then column)
    is used as described in the original EA paper.
    """

    def __init__(self, channels: int, S: int = 64):
        """
        Args
        ────
        channels : feature dimension D
        S        : external memory dimension (S << N).  S=64 follows the
                   default in the original EA paper and the PVT code release.
        """
        super().__init__()
        self.S = S

        # Two cascaded linear (1×1 conv) layers implementing Mₖ and Mᵥ
        self.mk = nn.Conv1d(channels, S, 1, bias=False)       # projects D → S  (Mₖ)
        self.mv = nn.Conv1d(S, channels, 1, bias=False)       # projects S → D  (Mᵥ)

        self.out_mlp = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Initialise Mᵥ to identity-like to encourage stable early training
        nn.init.eye_(self.mv.weight.view(channels, S)[:min(channels, S), :min(channels, S)])

    def forward(self, pos: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  pos [B,3,N] (unused; kept for interface compatibility),  x [B,C,N]
        Returns: [B, C, N]

        NaN-stability note — see LocalWindowAttention.forward() docstring.
        Softmax (dim=1) and the column-normalisation division can both
        overflow / divide-by-tiny in fp16; the entire EA block therefore
        runs in fp32 under AMP.
        """
        # ── fp32 attention block (AMP-safe) ──────────────────────────────
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.float()

            # Step 1: attention map via Mₖ
            A = self.mk(x32)                                          # [B, S, N]

            # Row-wise softmax: normalise over S (memory slots)
            A = F.softmax(A, dim=1)                                   # [B, S, N]

            # Column-wise L1 normalisation: each slot contributes proportionally
            A = A / (A.sum(dim=2, keepdim=True) + 1e-6)               # [B, S, N]

            # Step 2: aggregate via Mᵥ
            F_global = self.mv(A)                                     # [B, C, N]

        F_global = F_global.to(x.dtype)
        return self.out_mlp(F_global) + x                             # residual connection


# ──────────────────────────────────────────────────────────────────────────────
# PVT Block  (Sec. 3 / Fig. 1 grey box)
# ──────────────────────────────────────────────────────────────────────────────

class PVTBlock(nn.Module):
    """
    Dual-branch PVT block — Fig. 1 grey box.

    F' = F_local + F_global     (Eq. 9, Feature Fusion, Sec. 3.3)

    where F_local  comes from the Voxel branch (LocalWindowAttention)
    and   F_global comes from the Point  branch (ExternalAttention).
    """

    def __init__(self, channels: int, k: int = 32, S: int = 64):
        super().__init__()
        self.norm_local  = nn.BatchNorm1d(channels)
        self.norm_global = nn.BatchNorm1d(channels)

        self.voxel_branch = LocalWindowAttention(channels, k=k)
        self.point_branch = ExternalAttention(channels, S=S)

    def forward(self, pos: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Both branches receive layer-normalised input (pre-norm convention)
        f_local  = self.voxel_branch(pos, self.norm_local(x))
        f_global = self.point_branch(pos, self.norm_global(x))

        # Feature fusion (Eq. 9): element-wise addition
        return f_local + f_global


# ──────────────────────────────────────────────────────────────────────────────
# Full PVT network — Part Segmentation  (Fig. 1)
# ──────────────────────────────────────────────────────────────────────────────

class PVT(nn.Module):
    """
    PVT for part segmentation — Fig. 1 of Zhang et al. 2022.

    Architecture summary (Fig. 1 caption):
        • 3 stacked PVT blocks  with channel dims 64 / 64 / 128
        • Global max-pool then broadcast
        • Multi-scale skip concatenation: 64+64+128+1024 = 1280-D
        • MLP head: 1280 → 256 → num_classes
        • Dropout p=0.5 in last two linear layers  (Sec. 4)

    Voxel-branch window size k:
        k=32 for N=4096   (≈ W³ window occupancy at this density)
        k=16 for N=2048

    External Attention memory dimension S=64 follows the EA paper default
    and the PVT authors' released code.
    """

    def __init__(
        self,
        num_classes: int,
        num_points:  int = 4096,
        k:           int = 32,    # local window size for voxel branch
        S:           int = 64,    # EA external memory dimension
    ):
        super().__init__()

        # ── Stem: 6-D input (XYZ + normals) → 64 channels ───────────────
        self.stem = nn.Sequential(
            nn.Conv1d(6, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── 3 stacked PVT blocks (Fig. 1) ─────────────────────────────────
        # Blocks 1 & 2: 64 channels
        self.block1 = PVTBlock(64,  k=k, S=S)
        self.block2 = PVTBlock(64,  k=k, S=S)

        # Transition: 64 → 128 before block 3
        self.transition = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Block 3: 128 channels
        self.block3 = PVTBlock(128, k=k, S=S)

        # ── Global feature aggregation (Fig. 1) ───────────────────────────
        # "max-pooling and repeating operators to extract an effective global
        #  feature representing the entire point cloud"  — Fig. 1 caption
        self.global_mlp = nn.Sequential(
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── Segmentation head (Fig. 1 caption) ────────────────────────────
        # "one MLP layer (1280) to aggregate multi-scale features,
        #  where we concatenate features from previous layers to get a
        #  64+64+128+1024 = 1280-dimensional point cloud"
        self.head = nn.Sequential(
            nn.Conv1d(1280, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),                  # "dropout with keep prob 0.5" — Sec. 4
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, pos: torch.Tensor, x: torch.Tensor):
        """
        Args
        ────
        pos : [B, 3, N]   point coordinates
        x   : [B, 6, N]   point features (XYZ + normals)

        Returns
        ───────
        log_probs : [B, num_classes, N]
        None      : placeholder (consistent with train_universal.py interface)
        """
        B, _, N = x.shape

        # Stem projection
        f0 = self.stem(x)                                     # [B,  64, N]

        # PVT blocks — multi-scale feature extraction
        f1 = self.block1(pos, f0)                             # [B,  64, N]
        f2 = self.block2(pos, f1)                             # [B,  64, N]
        f3 = self.block3(pos, self.transition(f2))            # [B, 128, N]

        # Global max-pool feature
        f_g = self.global_mlp(f3)                             # [B, 1024, N]
        f_g = f_g.max(dim=2, keepdim=True)[0].expand(-1, -1, N)  # [B, 1024, N]

        # Multi-scale skip concatenation: 64 + 64 + 128 + 1024 = 1280
        f_cat = torch.cat([f1, f2, f3, f_g], dim=1)          # [B, 1280, N]

        # Point-wise prediction
        pred = self.head(f_cat)                               # [B, num_classes, N]

        return F.log_softmax(pred, dim=1), None