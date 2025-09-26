import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from contextlib import nullcontext




######------------------- computes on-the-fly ----------------------
#def fast_sct_diffusion(W, order, feature):
#    """Row-stochastic power iteration with degree clamp"""
#    degrees = W.sum(dim=2, keepdim=True).clamp_min(1e-6)
#    D_inv = 1.0 / degrees
#    P = W * D_inv.transpose(-1, -2)   # row-stochastic
#    max_iter = 1 << order
#    scale_points = [(1 << i) - 1 for i in range(order + 1)]
#    x = feature
#    snapshots = []
#    for i in range(max_iter):
#        x = 0.5 * (x + torch.bmm(P, x))
#        if i in scale_points:
#            snapshots.append(x)
#    sct_features = [F.gelu(snapshots[i] - snapshots[i + 1]) for i in range(order)]
#    return sct_features


# ------------------- RMSNorm -------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight



#class CoordinateFeatureExtractor(nn.Module):
#    """
#    Equivariant coordinate feature extractor - drop-in replacement.
#
#    Maintains equivariance by using only node-relative features:
#    - Polar coordinates relative to principal axes (equivariant)
#    - Distance statistics to all other nodes (equivariant)
#    - Angular distribution features (equivariant)
#    - Global point cloud statistics (invariant)
#
#    Removes position encoding that was tied to node indices.
#    """
#
#    def __init__(self, phi_harmonics=4, pos_harmonics=3, include_global_stats=True):
#        super().__init__()
#        self.phi_harmonics = phi_harmonics
#        self.pos_harmonics = pos_harmonics  # Keep for compatibility, but won't use
#        self.include_global_stats = include_global_stats
#
#        # Feature dimensions (modified for equivariance)
#        self.base_feat_dim = 3 + 2 * phi_harmonics  # r, ax, ay, sin/cos harmonics
#        self.distance_stats_dim = 2 * pos_harmonics  # Repurpose pos_harmonics for distance stats
#        self.angular_stats_dim = 2 * pos_harmonics   # Angular distribution features
#        self.global_stats_dim = 8 if include_global_stats else 0
#
#        self.total_feat_dim = (self.base_feat_dim + self.distance_stats_dim +
#                              self.angular_stats_dim + self.global_stats_dim)
#
#        print(f"🎯 Equivariant Coordinate Feature Extractor:")
#        print(f"   • Polar harmonics: {phi_harmonics}")
#        print(f"   • Distance stats: {self.distance_stats_dim}")
#        print(f"   • Angular stats: {self.angular_stats_dim}")
#        print(f"   • Global stats: {include_global_stats}")
#        print(f"   • Output feature dim: {self.total_feat_dim}")
#
#    def polar_feats_fast(self, coords: torch.Tensor, c: torch.Tensor,
#                         u: torch.Tensor, u_perp: torch.Tensor) -> torch.Tensor:
#        """Fast polar feature computation with fused operations"""
#        B, N, _ = coords.shape
#        x = coords - c.unsqueeze(1)               # [B,N,2]
#        ax = torch.sum(x * u.unsqueeze(1), dim=-1)
#        ay = torch.sum(x * u_perp.unsqueeze(1), dim=-1)
#        r = torch.sqrt(ax.pow(2) + ay.pow(2) + 1e-6)
#        theta = torch.atan2(ay, ax)
#        m = torch.arange(1, self.phi_harmonics + 1, device=coords.device, dtype=coords.dtype)
#        thm = theta.unsqueeze(-1) * m.unsqueeze(0).unsqueeze(0)
#        feats = torch.cat([
#            r.unsqueeze(-1),
#            ax.unsqueeze(-1),
#            ay.unsqueeze(-1),
#            torch.sin(thm),
#            torch.cos(thm)
#        ], dim=-1)
#        return feats
#
#    def compute_principal_axes(self, coords: torch.Tensor) -> tuple:
#        """Stable principal axes via SVD per batch"""
#        B, N, _ = coords.shape
#        centers = coords.mean(dim=1, keepdim=True)        # [B,1,2]
#        centered = coords - centers                       # [B,N,2]
#        # SVD is numerically stabler than eigh for small 2x2 cov
#        Vh = torch.linalg.svd(centered, full_matrices=False).Vh   # [B,2,2]
#        V = Vh.transpose(-1, -2)
#        u = V[:, :, 0]       # [B,2]
#        u_perp = V[:, :, 1]  # [B,2]
#        return centers.squeeze(1), u, u_perp
#
#    def compute_global_stats(self, coords: torch.Tensor) -> torch.Tensor:
#        """Compute global coordinate statistics"""
#        B, N, _ = coords.shape
#        mean_coords = coords.mean(dim=1)                          # [B,2]
#        std_coords = coords.std(dim=1, unbiased=False).clamp_min(1e-6)  # [B,2]
#        min_coords = coords.min(dim=1)[0]
#        max_coords = coords.max(dim=1)[0]
#        bbox_size = (max_coords - min_coords).clamp_min(1e-6)
#        bbox_center = (max_coords + min_coords) / 2
#        global_stats = torch.cat([mean_coords, std_coords, bbox_size, bbox_center], dim=-1)  # [B,8]
#        return global_stats
#
#    def compute_distance_stats(self, coords: torch.Tensor) -> torch.Tensor:
#        """
#        Compute equivariant distance statistics for each node.
#        Each node gets statistics about its distances to all other nodes.
#        """
#        B, N, _ = coords.shape
#
#        # Compute pairwise distances [B, N, N]
#        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, N, N, 2]
#        distances = torch.norm(diff, dim=-1)  # [B, N, N]
#
#        # For each node, compute stats over distances to other nodes (exclude self)
#        distance_stats = []
#        for i in range(N):
#            # Get distances from node i to all others (exclude self-distance)
#            node_dists = torch.cat([
#                distances[:, i, :i],
#                distances[:, i, i+1:]
#            ], dim=1)  # [B, N-1]
#
#            # Compute various distance statistics
#            stats = []
#            for k in range(1, self.pos_harmonics + 1):
#                # Use different statistical moments
#                mean_dist = node_dists.mean(dim=1)  # [B]
#                if k == 1:
#                    stats.append(mean_dist)
#                    stats.append(node_dists.std(dim=1, unbiased=False))  # [B]
#                elif k == 2:
#                    stats.append(node_dists.min(dim=1)[0])  # [B] - min distance
#                    stats.append(node_dists.max(dim=1)[0])  # [B] - max distance
#                elif k == 3:
#                    stats.append(torch.median(node_dists, dim=1)[0])  # [B] - median
#                    stats.append(torch.quantile(node_dists, 0.9, dim=1))  # [B] - 90th percentile
#
#            node_stats = torch.stack(stats[:2 * self.pos_harmonics], dim=1)  # [B, 2*pos_harmonics]
#            distance_stats.append(node_stats)
#
#        return torch.stack(distance_stats, dim=1)  # [B, N, 2*pos_harmonics]
#
#    def compute_angular_stats(self, coords: torch.Tensor) -> torch.Tensor:
#        """
#        Compute equivariant angular distribution statistics for each node.
#        Each node gets statistics about angles to all other nodes.
#        """
#        B, N, _ = coords.shape
#
#        angular_stats = []
#        for i in range(N):
#            node_coord = coords[:, i:i+1, :]  # [B, 1, 2]
#            other_coords = torch.cat([
#                coords[:, :i, :],
#                coords[:, i+1:, :]
#            ], dim=1)  # [B, N-1, 2]
#
#            # Compute relative vectors and angles
#            rel_vectors = other_coords - node_coord  # [B, N-1, 2]
#            angles = torch.atan2(rel_vectors[:, :, 1], rel_vectors[:, :, 0])  # [B, N-1]
#
#            # Compute angular distribution features using Fourier components
#            stats = []
#            for k in range(1, self.pos_harmonics + 1):
#                # Different frequency components of angular distribution
#                freq = k * 2.0
#                sin_component = torch.sin(angles * freq).mean(dim=1)  # [B]
#                cos_component = torch.cos(angles * freq).mean(dim=1)  # [B]
#                stats.extend([sin_component, cos_component])
#
#            node_angular_stats = torch.stack(stats[:2 * self.pos_harmonics], dim=1)  # [B, 2*pos_harmonics]
#            angular_stats.append(node_angular_stats)
#
#        return torch.stack(angular_stats, dim=1)  # [B, N, 2*pos_harmonics]
#
#    def forward(self, coords: torch.Tensor) -> torch.Tensor:
#        B, N, _ = coords.shape
#        device = coords.device
#        dtype = coords.dtype
#
#        # Compute principal axes and polar features (equivariant)
#        centers, u, u_perp = self.compute_principal_axes(coords)
#        polar_feats = self.polar_feats_fast(coords, centers, u, u_perp)  # [B,N,base]
#
#        # Compute distance statistics (equivariant)
#        distance_feats = self.compute_distance_stats(coords)  # [B,N,2*pos_harmonics]
#
#        # Compute angular statistics (equivariant)
#        angular_feats = self.compute_angular_stats(coords)  # [B,N,2*pos_harmonics]
#
#        # Combine equivariant features
#        features = torch.cat([polar_feats, distance_feats, angular_feats], dim=-1)
#
#        # Add global statistics (invariant) if requested
#        if self.include_global_stats:
#            global_stats = self.compute_global_stats(coords).unsqueeze(1).expand(-1, N, -1)
#            features = torch.cat([features, global_stats], dim=-1)
#
#        return features
#


# ------------------- (Optional) Coordinate extractor -------------------
#class CoordinateFeatureExtractor(nn.Module):
#    def __init__(self, phi_harmonics=4, pos_harmonics=3, include_global_stats=True):
#        super().__init__()
#        self.phi_harmonics = phi_harmonics
#        self.pos_harmonics = pos_harmonics
#        self.include_global_stats = include_global_stats
#        self.base_feat_dim = 3 + 2 * phi_harmonics
#        self.pos_encoding_dim = 2 * pos_harmonics
#        self.global_stats_dim = 8 if include_global_stats else 0
#        self.total_feat_dim = self.base_feat_dim + self.pos_encoding_dim + self.global_stats_dim
#
#    @staticmethod
#    def _cycle_pos_encoding(k, num_harmonics, device, dtype):
#        two_pi = 2.0 * math.pi
#        p = torch.arange(k, device=device, dtype=dtype)
#        m = torch.arange(1, num_harmonics + 1, device=device, dtype=dtype)
#        ang = two_pi * torch.outer(p, m) / float(k)
#        return torch.cat([torch.cos(ang), torch.sin(ang)], dim=1)
#
#    def _principal_axes(self, coords: torch.Tensor):
#        centers = coords.mean(dim=1, keepdim=True)
#        centered = coords - centers
#        Vh = torch.linalg.svd(centered, full_matrices=False).Vh
#        V = Vh.transpose(-1, -2)
#        u = V[:, :, 0]; u_perp = V[:, :, 1]
#        return centers.squeeze(1), u, u_perp
#
#    def _polar_feats(self, coords, c, u, u_perp):
#        x = coords - c.unsqueeze(1)
#        ax = (x * u.unsqueeze(1)).sum(-1)
#        ay = (x * u_perp.unsqueeze(1)).sum(-1)
#        r = torch.sqrt(ax.pow(2) + ay.pow(2) + 1e-6)
#        theta = torch.atan2(ay, ax)
#        m = torch.arange(1, self.phi_harmonics + 1, device=coords.device, dtype=coords.dtype)
#        thm = theta.unsqueeze(-1) * m.unsqueeze(0).unsqueeze(0)
#        return torch.cat([r.unsqueeze(-1), ax.unsqueeze(-1), ay.unsqueeze(-1),
#                          torch.sin(thm), torch.cos(thm)], dim=-1)
#
#    def _global_stats(self, coords):
#        mean_coords = coords.mean(dim=1)
#        std_coords = coords.std(dim=1, unbiased=False).clamp_min(1e-6)
#        min_coords = coords.min(dim=1)[0]; max_coords = coords.max(dim=1)[0]
#        bbox_size = (max_coords - min_coords).clamp_min(1e-6)
#        bbox_center = (max_coords + min_coords) / 2
#        return torch.cat([mean_coords, std_coords, bbox_size, bbox_center], dim=-1)
#
#    def forward(self, coords: torch.Tensor) -> torch.Tensor:
#        B, N, _ = coords.shape
#        centers, u, u_perp = self._principal_axes(coords)
#        polar_feats = self._polar_feats(coords, centers, u, u_perp)
#        psi = self._cycle_pos_encoding(N, self.pos_harmonics, coords.device, coords.dtype).unsqueeze(0).expand(B, -1, -1)
#        feats = torch.cat([polar_feats, psi], dim=-1)
#        if self.include_global_stats:
#            gs = self._global_stats(coords).unsqueeze(1).expand(-1, N, -1)
#            feats = torch.cat([feats, gs], dim=-1)
#        return feats



class EquivariantCoordinateFeatureExtractor(nn.Module):
    """
    Strictly permutation-equivariant feature extractor:
    - Uses only relative, order-invariant features
    - Removes index-based cycle encodings
    - Fixes SVD sign ambiguity
    """

    def __init__(self, phi_harmonics=4, include_global_stats=True):
        super().__init__()
        self.phi_harmonics = phi_harmonics
        self.include_global_stats = include_global_stats
        # r, ax, ay + (sin/cos harmonics) + optional global stats
        self.base_feat_dim = 3 + 2 * phi_harmonics
        self.global_stats_dim = 8 if include_global_stats else 0
        self.total_feat_dim = self.base_feat_dim + self.global_stats_dim

    def _principal_axes(self, coords: torch.Tensor):
        """
        Compute principal axes (u, u_perp), with deterministic sign convention
        to avoid arbitrary flips from SVD.
        """
        centers = coords.mean(dim=1, keepdim=True)
        centered = coords - centers

        # [B, N, 2] -> covariance matrix [B, 2, 2]
        cov = torch.matmul(centered.transpose(1, 2), centered) / coords.shape[1]

        # Eigen-decomposition
        eigvals, eigvecs = torch.linalg.eigh(cov)  # sorted ascending
        u = eigvecs[:, :, -1]      # principal axis
        u_perp = eigvecs[:, :, 0]  # orthogonal axis

        # Fix sign: enforce that the first component of u is non-negative
        sign = torch.where(u[:, 0] < 0, -1.0, 1.0).unsqueeze(-1)
        u = u * sign
        u_perp = u_perp * sign

        return centers.squeeze(1), u, u_perp

    def _polar_feats(self, coords, c, u, u_perp):
        """
        Relative polar features w.r.t. principal axes.
        """
        x = coords - c.unsqueeze(1)         # [B, N, 2]
        ax = (x * u.unsqueeze(1)).sum(-1)   # projection onto u
        ay = (x * u_perp.unsqueeze(1)).sum(-1)
        r = torch.sqrt(ax.pow(2) + ay.pow(2) + 1e-6)
        theta = torch.atan2(ay, ax)

        # Harmonics of angle
        m = torch.arange(1, self.phi_harmonics + 1, device=coords.device, dtype=coords.dtype)
        thm = theta.unsqueeze(-1) * m.unsqueeze(0).unsqueeze(0)
        return torch.cat([r.unsqueeze(-1),
                          ax.unsqueeze(-1),
                          ay.unsqueeze(-1),
                          torch.sin(thm),
                          torch.cos(thm)], dim=-1)

    def _global_stats(self, coords):
        mean_coords = coords.mean(dim=1)
        std_coords = coords.std(dim=1, unbiased=False).clamp_min(1e-6)
        min_coords = coords.min(dim=1)[0]
        max_coords = coords.max(dim=1)[0]
        bbox_size = (max_coords - min_coords).clamp_min(1e-6)
        bbox_center = (max_coords + min_coords) / 2
        return torch.cat([mean_coords, std_coords, bbox_size, bbox_center], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [B, N, 2]
        returns: [B, N, F] permutation-equivariant features
        """
        B, N, _ = coords.shape
        c, u, u_perp = self._principal_axes(coords)
        polar_feats = self._polar_feats(coords, c, u, u_perp)

        feats = polar_feats
        if self.include_global_stats:
            gs = self._global_stats(coords).unsqueeze(1).expand(-1, N, -1)
            feats = torch.cat([feats, gs], dim=-1)

        return feats

# ------------------- Graph operator precompute -------------------
@torch.no_grad()
def precompute_graph_operators(W: torch.Tensor, gcn_order: int, sct_order: int) -> Dict[str, List[torch.Tensor]]:
    B, N, _ = W.shape
    device, dtype = W.device, W.dtype

    # A_norm = (A+I) D^{-1/2} (A+I) D^{-1/2}
    A = W.clone()
    idx = torch.arange(N, device=device)
    A[:, idx, idx] = A[:, idx, idx] + 1.0
    deg = A.sum(-1, keepdim=True).clamp_min(1e-6)
    Dm12 = torch.rsqrt(deg)
    A_norm = A * Dm12 * Dm12.transpose(-1, -2)

    gcn_ops = [A_norm]
    for _ in range(1, max(gcn_order, 1)):
        gcn_ops.append(torch.bmm(gcn_ops[-1], A_norm))

    # Row-stochastic P and S = 0.5(I + P)
    deg_row = W.sum(-1, keepdim=True).clamp_min(1e-6)
    P = W * (1.0 / deg_row.transpose(-1, -2))
    I = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    S = 0.5 * (I + P)

    # Compute S powers by squaring: S, S², S⁴, S⁸, ...
    S_powers = [S]  # S^(2^0) = S^1
    for _ in range(sct_order):
        S_powers.append(torch.bmm(S_powers[-1], S_powers[-1]))
    
    # FIXED: SCT operators are differences: S^(2^i) - S^(2^(i+1))
    sct_ops = []
    for i in range(sct_order):
        M_i = S_powers[i] - S_powers[i + 1]  # S^(2^i) - S^(2^(i+1))
        sct_ops.append(M_i)


    return {"gcn": gcn_ops, "sct": sct_ops}

# ------------------- Turbo Layer with ATTENTION-based channel mixing -------------------
class FastSCTLayer(nn.Module):
    def __init__(self, hidden_dim, num_channels, attn_dropout=0.05, ff_dropout=0.1, res_scale=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.res_scale = res_scale
        
        # Pre and post norms
        self.norm_in = RMSNorm(hidden_dim)
        self.norm_post = RMSNorm(hidden_dim)
        
        # Attention projection: [Xin, feature] -> attention logits
        self.attention_proj = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        
        # Feed-forward
        self.proj1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ff_drop = nn.Dropout(ff_dropout)
        
        # Initialize
        nn.init.xavier_uniform_(self.attention_proj.weight, gain=0.8)
        nn.init.xavier_uniform_(self.proj1.weight, gain=0.8)
        nn.init.xavier_uniform_(self.proj2.weight, gain=0.8)

    def forward(self, X, diffusion_outs: List[torch.Tensor]):
        """
        X: [B, N, H]
        diffusion_outs: list of [B, N, H] tensors (one per channel)
        """
        Xin = self.norm_in(X)
        
        # Concatenate [Xin, feature] for each channel
        # concat_features: [B, C, N, 2H]
        concat_features = torch.stack(
            [torch.cat([Xin, f], dim=-1) for f in diffusion_outs], dim=1
        )
        
        # Compute attention logits per channel and per node
        attn_logits = self.attention_proj(concat_features).squeeze(-1)  # [B, C, N]
        
        # Scale by sqrt(d) like in the original
        attn_logits = attn_logits / math.sqrt(self.hidden_dim)
        
        # Softmax over channels (dim=1)
        attn = F.softmax(attn_logits, dim=1)  # [B, C, N]
        attn = self.attn_drop(attn)
        
        # Stack all channel features and weighted sum
        stacked = torch.stack(diffusion_outs, dim=1)  # [B, C, N, H]
        mixed = (attn.unsqueeze(-1) * stacked).sum(dim=1)  # [B, N, H]
        
        # Feed-forward with residual
        out = F.gelu(self.proj1(self.norm_post(mixed)))
        out = self.proj2(out)
        out = self.ff_drop(out)
        
        return X + self.res_scale * out

# ------------------- Gumbel–Sinkhorn -------------------
def fast_sinkhorn_iteration(Z, n_iter=10):
    Z = Z.to(torch.float32)
    for _ in range(n_iter):
        Z = Z - torch.logsumexp(Z, dim=-1, keepdim=True)
        Z = Z - torch.logsumexp(Z, dim=-2, keepdim=True)
    return Z

def fast_gumbel_sinkhorn(logits, tau=0.5, n_iter=10, noise_scale=0.05, clamp_val=20.0):
    #logits = logits.clamp(-clamp_val, clamp_val)
    if noise_scale > 0:
        uniform = torch.rand_like(logits, dtype=torch.float32)
        g = -torch.log(-torch.log(uniform + 1e-20) + 1e-20) * noise_scale
    else:
        g = torch.zeros_like(logits, dtype=torch.float32)
    Z_init = (logits.to(torch.float32) + g) / max(tau, 1e-3)
    # Z_init = Z_init.clamp(-clamp_val, clamp_val) # when set tau > 1, don't need this
    Z = fast_sinkhorn_iteration(Z_init, n_iter)
    P = F.softmax(Z, dim=-1)
    P = P / P.sum(dim=-2, keepdim=True).clamp_min(1e-6)
    return P.to(logits.dtype), Z_init.to(logits.dtype)

# ------------------- The fast model with ATTENTION mixing -------------------
class FastSCTGNN(nn.Module):
    def __init__(
        self,
        input_dim, hidden_dim, output_dim, inference_mode=False,
        n_layers=32, sct_order=2, gcn_order=1, tanh_scale=0.0,
        coord_input=True, phi_harmonics=4, pos_harmonics=3, include_global_stats=True,
        tau=0.5, n_iter=10, noise_scale=0.05, logit_clamp=20.0,
        amp_autocast=False, attn_dropout=0.05, ff_dropout=0.1
    ):
        super().__init__()
        self.coord_input = coord_input
        self.sct_order = sct_order
        self.gcn_order = gcn_order
        self.tau = tau; self.n_iter = n_iter
        self.noise_scale = noise_scale
        self.logit_clamp = logit_clamp
        self.amp_autocast = amp_autocast
        self.tanh_scale = tanh_scale
        self.inference = inference_mode

        if coord_input:
            assert input_dim == 2
            # self.coord_extractor = CoordinateFeatureExtractor(phi_harmonics, pos_harmonics, include_global_stats)
            # actual_in = self.coord_extractor.total_feat_dim
            self.coord_extractor = EquivariantCoordinateFeatureExtractor(phi_harmonics, include_global_stats)
            actual_in = self.coord_extractor.total_feat_dim
        else:
            self.coord_extractor = None
            actual_in = input_dim

        self.in_norm = RMSNorm(actual_in)
        self.in_proj = nn.Linear(actual_in, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.8)

        # Number of channels = sct_order + gcn_order
        num_channels = sct_order + max(gcn_order, 0)
        res_scale = 1.0 / math.sqrt(max(n_layers, 1))
        
        # Use ATTENTION-based layers 
        self.layers = nn.ModuleList([
            FastSCTLayer(hidden_dim, num_channels, attn_dropout, ff_dropout, res_scale=res_scale) 
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(hidden_dim * (1 + n_layers))
        self.out1 = nn.Linear(hidden_dim * (1 + n_layers), hidden_dim, bias=False)
        self.out2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        nn.init.xavier_uniform_(self.out1.weight, gain=0.8)
        nn.init.xavier_uniform_(self.out2.weight, gain=0.8)

    def _encode_inputs(self, X):
        if self.coord_input and self.coord_extractor is not None:
            X = self.coord_extractor(X)
        return self.in_proj(self.in_norm(X))

    def _make_diffusion_features(self, Xin, ops: Dict[str, List[torch.Tensor]], moment=1.0):
        outs = []
        
        # GCN features first
        for Aop in ops["gcn"]:
            outs.append(F.gelu(torch.bmm(Aop, Xin)))
        
        # Then SCT features
        for Mop in ops["sct"]:
            f = torch.bmm(Mop, Xin)
            if moment != 1.0:
                f = torch.abs(f).pow(moment)
            outs.append(F.gelu(f))
        
        return outs

    def forward(self, X, adj, moment: float = 1.0, cache: Optional[Dict]=None, inference: bool = False):
        # Choose the right autocast context
        if self.amp_autocast and X.is_cuda:
            ac = torch.amp.autocast('cuda')
        elif self.amp_autocast and not X.is_cuda:
            ac = torch.amp.autocast('cpu')
        else:
            ac = nullcontext()
    
        with ac:
            # Precompute or use cached operators
            if cache is not None and "ops" in cache:
                ops = cache["ops"]
            else:
                ops = precompute_graph_operators(adj, self.gcn_order, self.sct_order)
                if cache is not None:
                    cache["ops"] = ops
    
            # Encode inputs
            Xin = self._encode_inputs(X)
            states = [Xin]
            
            # Process through layers with attention-based mixing
            for layer in self.layers:
                # Pre-compute normalized version for diffusion
                Xin_normed = layer.norm_in(Xin)
                feats = self._make_diffusion_features(Xin_normed, ops, moment)
                #feats = self._make_diffusion_features(Xin, ops, moment=moment)
                Xin = layer(Xin, feats)
                states.append(Xin)
    
            # Final output projection
            H = torch.cat(states, dim=-1)
            H = self.final_norm(H)
            h = F.gelu(self.out1(H))
            logits = self.out2(h) * self.logit_scale
            
            # Optional tanh squeezing (default 0, i.e., disabled)
            if self.tanh_scale and self.tanh_scale > 0:
                logits = torch.tanh(logits) * self.tanh_scale
            
            # Clamp to avoid softmax/sinkhorn overflow
            logits = logits.clamp(-self.logit_clamp, self.logit_clamp)
        
        # Return based on mode (use parameter 'inference' if provided, else use self.inference)
        if self.training and not (inference or self.inference):
            P, Z = fast_gumbel_sinkhorn(logits, tau=self.tau, n_iter=self.n_iter,
                                        noise_scale=self.noise_scale, clamp_val=self.logit_clamp)
            return P, Z
        else:
            # print(logits)
            return logits

# ------------------- Factory -------------------
def create_fast_sct_gnn(**kwargs):
    return FastSCTGNN(**kwargs)

# ------------------- Main: Equivariance test -------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, N, Fin = 2, 100, 2
    H, L = 256, 16
    sct_order, gcn_order = 3, 2
    Fout = N

    # Random coords/features and symmetric adjacency
    X = torch.randn(B, N, Fin, device=device)
    A = torch.rand(B, N, N, device=device)
    A = 0.5 * (A + A.transpose(-1, -2))

    model = create_fast_sct_gnn(
        input_dim=Fin, hidden_dim=H, output_dim=Fout,
        n_layers=L, sct_order=sct_order, gcn_order=gcn_order,
        coord_input=True, tau=0.5, n_iter=10, noise_scale=0.05,
        logit_clamp=20.0, amp_autocast=False, attn_dropout=0.05, ff_dropout=0.1
    ).to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    model.train()
    P, Z = model(X, A)
    print(f"\nTraining mode output: P shape={P.shape}, Z shape={Z.shape}")
    print(f"Doubly stochastic check: rows={P.sum(-1).mean():.4f}, cols={P.sum(-2).mean():.4f}")

    model.eval()
    logits = model(X, A)
    print(f"\nInference mode output: logits shape={logits.shape}")

    # Equivariance test
    with torch.no_grad():
        perm = torch.randperm(N, device=device)
        P_mat = torch.zeros(N, N, device=device)
        P_mat[torch.arange(N, device=device), perm] = 1.0
        Pb = P_mat.unsqueeze(0).expand(B, -1, -1)

        Xp = torch.bmm(Pb, X)
        Ap = torch.bmm(torch.bmm(Pb, A), Pb.transpose(-1, -2))

        logits_orig = model(X, A)
        logits_perm = model(Xp, Ap)

        left = torch.bmm(Pb.to(logits_orig.dtype), logits_orig)
        err = (left - logits_perm).abs().max().item()
        print(f"\n✓ Equivariance test: Max |P f(X) - f(PX)| = {err:.3e}")

