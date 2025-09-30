import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from contextlib import nullcontext

# ------------------- RMSNorm -------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

# ------------------- Equivariant Coordinate Feature Extractor -------------------
class EquivariantCoordinateFeatureExtractor(nn.Module):
    """
    Strictly permutation-equivariant feature extractor:
    - Uses only relative, order-invariant features
    - Removes index-based cycle encodings
    - Fixes eigenvector sign ambiguity
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
        centers = coords.mean(dim=1, keepdim=True)
        centered = coords - centers
        # covariance [B,2,2]
        cov = torch.matmul(centered.transpose(1, 2), centered) / coords.shape[1]
        eigvals, eigvecs = torch.linalg.eigh(cov)
        u = eigvecs[:, :, -1]      # principal
        u_perp = eigvecs[:, :, 0]  # orthogonal
        # deterministic sign: first component non-negative
        sign = torch.where(u[:, 0] < 0, -1.0, 1.0).unsqueeze(-1)
        u = u * sign
        u_perp = u_perp * sign
        return centers.squeeze(1), u, u_perp

    def _polar_feats(self, coords, c, u, u_perp):
        x = coords - c.unsqueeze(1)         # [B,N,2]
        ax = (x * u.unsqueeze(1)).sum(-1)
        ay = (x * u_perp.unsqueeze(1)).sum(-1)
        r = torch.sqrt(ax.pow(2) + ay.pow(2) + 1e-6)
        theta = torch.atan2(ay, ax)
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

    # S powers: S, S^2, S^4, ...
    S_powers = [S]
    for _ in range(sct_order):
        S_powers.append(torch.bmm(S_powers[-1], S_powers[-1]))

    # SCT operators as differences
    sct_ops = []
    for i in range(sct_order):
        M_i = S_powers[i] - S_powers[i + 1]
        sct_ops.append(M_i)

    return {"gcn": gcn_ops if gcn_order > 0 else [], "sct": sct_ops if sct_order > 0 else []}

# ------------------- Attention-mixing layer (matches sagmodel FastSCTLayer) -------------------
class FastSCTLayer(nn.Module):
    def __init__(self, hidden_dim, num_channels, attn_dropout=0.05, ff_dropout=0.1, res_scale=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.res_scale = res_scale

        self.norm_in = RMSNorm(hidden_dim)
        self.norm_post = RMSNorm(hidden_dim)

        self.attention_proj = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.proj1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ff_drop = nn.Dropout(ff_dropout)

        nn.init.xavier_uniform_(self.attention_proj.weight, gain=0.8)
        nn.init.xavier_uniform_(self.proj1.weight, gain=0.8)
        nn.init.xavier_uniform_(self.proj2.weight, gain=0.8)

    def forward(self, X, diffusion_outs: List[torch.Tensor]):
        Xin = self.norm_in(X)
        concat_features = torch.stack([torch.cat([Xin, f], dim=-1) for f in diffusion_outs], dim=1)
        attn_logits = self.attention_proj(concat_features).squeeze(-1)  # [B, C, N]
        attn_logits = attn_logits / math.sqrt(self.hidden_dim)
        attn = F.softmax(attn_logits, dim=1)
        attn = self.attn_drop(attn)

        stacked = torch.stack(diffusion_outs, dim=1)  # [B, C, N, H]
        mixed = (attn.unsqueeze(-1) * stacked).sum(dim=1)  # [B, N, H]

        out = F.gelu(self.proj1(self.norm_post(mixed)))
        out = self.proj2(out)
        out = self.ff_drop(out)
        return X + self.res_scale * out

# ------------------- Gumbel–Sinkhorn utilities -------------------
def fast_sinkhorn_iteration(Z, n_iter=10):
    Z = Z.to(torch.float32)
    for _ in range(n_iter):
        Z = Z - torch.logsumexp(Z, dim=-1, keepdim=True)
        Z = Z - torch.logsumexp(Z, dim=-2, keepdim=True)
    return Z

def fast_gumbel_sinkhorn(logits, tau=0.5, n_iter=10, noise_scale=0.05, clamp_val=20.0):
    if noise_scale > 0:
        uniform = torch.rand_like(logits, dtype=torch.float32)
        g = -torch.log(-torch.log(uniform + 1e-20) + 1e-20) * noise_scale
    else:
        g = torch.zeros_like(logits, dtype=torch.float32)
    Z_init = (logits.to(torch.float32) + g) / max(tau, 1e-3)
    Z = fast_sinkhorn_iteration(Z_init, n_iter)
    P = F.softmax(Z, dim=-1)
    P = P / P.sum(dim=-2, keepdim=True).clamp_min(1e-6)
    return P.to(logits.dtype), Z_init.to(logits.dtype)

# ------------------- EnhancedUltraFastSCTGNN (API-compatible with amp_autocast) -------------------
class EnhancedUltraFastSCTGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        inference_mode: bool = False,
        n_layers: int = 32,
        sct_order: int = 2,
        gcn_order: int = 1,
        tanh_scale: float = 0.0,
        coord_input: bool = True,
        phi_harmonics: int = 4,
        include_global_stats: bool = True,
        tau: float = 0.5,
        n_iter: int = 10,
        noise_scale: float = 0.05,
        logit_clamp: float = 20.0,
        amp_autocast: bool = False,           # <-- accept amp_autocast
        attn_dropout: float = 0.05,
        ff_dropout: float = 0.1,
    ):
        super().__init__()
        self.coord_input = coord_input
        self.sct_order = sct_order
        self.gcn_order = gcn_order
        self.tau = tau
        self.n_iter = n_iter
        self.noise_scale = noise_scale
        self.logit_clamp = logit_clamp
        self.amp_autocast = amp_autocast
        self.tanh_scale = tanh_scale
        self.inference = inference_mode

        if coord_input:
            assert input_dim == 2, "coord_input=True expects 2D coordinates"
            self.coord_extractor = EquivariantCoordinateFeatureExtractor(phi_harmonics, include_global_stats)
            actual_in = self.coord_extractor.total_feat_dim
        else:
            self.coord_extractor = None
            actual_in = input_dim

        self.in_norm = RMSNorm(actual_in)
        self.in_proj = nn.Linear(actual_in, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.8)

        num_channels = sct_order + max(gcn_order, 0)
        res_scale = 1.0 / math.sqrt(max(n_layers, 1))

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

    def _make_diffusion_features(self, Xin, ops: Dict[str, List[torch.Tensor]], moment: float = 1.0):
        outs = []
        for Aop in ops["gcn"]:
            outs.append(F.gelu(torch.bmm(Aop, Xin)))
        for Mop in ops["sct"]:
            f = torch.bmm(Mop, Xin)
            if moment != 1.0:
                f = torch.abs(f).pow(moment)
            outs.append(F.gelu(f))
        return outs

    def forward(self, X, adj, moment: float = 1.0, cache: Optional[Dict] = None, inference: bool = False):
        # autocast context mirrors sagmodel behavior
        if self.amp_autocast and X.is_cuda:
            ac = torch.amp.autocast('cuda')
        elif self.amp_autocast and not X.is_cuda:
            ac = torch.amp.autocast('cpu')
        else:
            ac = nullcontext()

        with ac:
            # Precompute or use cached operators (DDP-friendly: no cross-rank tensors here)
            if cache is not None and "ops" in cache:
                ops = cache["ops"]
            else:
                ops = precompute_graph_operators(adj, self.gcn_order, self.sct_order)
                if cache is not None:
                    cache["ops"] = ops

            Xin = self._encode_inputs(X)
            states = [Xin]

            for layer in self.layers:
                Xin_normed = layer.norm_in(Xin)  # ensure diffusion runs on normalized features
                feats = self._make_diffusion_features(Xin_normed, ops, moment)
                Xin = layer(Xin, feats)
                states.append(Xin)

            H = torch.cat(states, dim=-1)
            H = self.final_norm(H)
            h = F.gelu(self.out1(H))
            logits = self.out2(h) * self.logit_scale

            if self.tanh_scale and self.tanh_scale > 0:
                logits = torch.tanh(logits) * self.tanh_scale

            logits = logits.clamp(-self.logit_clamp, self.logit_clamp)

        if self.training and not (inference or self.inference):
            P, Z = fast_gumbel_sinkhorn(
                logits, tau=self.tau, n_iter=self.n_iter,
                noise_scale=self.noise_scale, clamp_val=self.logit_clamp
            )
            return P, Z
        else:
            return logits

# ------------------- Factory -------------------
def create_enhanced_ultrafast_sct_gnn(**kwargs):
    return EnhancedUltraFastSCTGNN(**kwargs)

# ------------------- Minimal self-test -------------------
# if __name__ == "__main__":
#    torch.manual_seed(0)
#    device = "cuda" if torch.cuda.is_available() else "cpu"
#
#    B, N, Fin = 2, 64, 2
#    H, L = 128, 8
#    sct_order, gcn_order = 3, 2
#    Fout = N
#
#    X = torch.randn(B, N, Fin, device=device)
#    A = torch.rand(B, N, N, device=device)
#    A = 0.5 * (A + A.transpose(-1, -2))
#
#    model = create_enhanced_ultrafast_sct_gnn(
#        input_dim=Fin, hidden_dim=H, output_dim=Fout,
#        n_layers=L, sct_order=sct_order, gcn_order=gcn_order,
#        coord_input=True, tau=0.5, n_iter=10, noise_scale=0.05,
#        logit_clamp=20.0, amp_autocast=False, attn_dropout=0.05, ff_dropout=0.1
#    ).to(device)
#
#    model.train()
#    P, Z = model(X, A)
#    print("Train shapes:", P.shape, Z.shape)
#    model.eval()
#    logits = model(X, A)
#    print("Eval logits:", logits.shape)
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

    model = create_enhanced_ultrafast_sct_gnn(
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


