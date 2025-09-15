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

# ------------------- (Optional) Coordinate extractor -------------------
class CoordinateFeatureExtractor(nn.Module):
    def __init__(self, phi_harmonics=4, pos_harmonics=3, include_global_stats=True):
        super().__init__()
        self.phi_harmonics = phi_harmonics
        self.pos_harmonics = pos_harmonics
        self.include_global_stats = include_global_stats
        self.base_feat_dim = 3 + 2 * phi_harmonics
        self.pos_encoding_dim = 2 * pos_harmonics
        self.global_stats_dim = 8 if include_global_stats else 0
        self.total_feat_dim = self.base_feat_dim + self.pos_encoding_dim + self.global_stats_dim

    @staticmethod
    def _cycle_pos_encoding(k, num_harmonics, device, dtype):
        two_pi = 2.0 * math.pi
        p = torch.arange(k, device=device, dtype=dtype)
        m = torch.arange(1, num_harmonics + 1, device=device, dtype=dtype)
        ang = two_pi * torch.outer(p, m) / float(k)
        return torch.cat([torch.cos(ang), torch.sin(ang)], dim=1)

    def _principal_axes(self, coords: torch.Tensor):
        centers = coords.mean(dim=1, keepdim=True)
        centered = coords - centers
        Vh = torch.linalg.svd(centered, full_matrices=False).Vh
        V = Vh.transpose(-1, -2)
        u = V[:, :, 0]; u_perp = V[:, :, 1]
        return centers.squeeze(1), u, u_perp

    def _polar_feats(self, coords, c, u, u_perp):
        x = coords - c.unsqueeze(1)
        ax = (x * u.unsqueeze(1)).sum(-1)
        ay = (x * u_perp.unsqueeze(1)).sum(-1)
        r = torch.sqrt(ax.pow(2) + ay.pow(2) + 1e-6)
        theta = torch.atan2(ay, ax)
        m = torch.arange(1, self.phi_harmonics + 1, device=coords.device, dtype=coords.dtype)
        thm = theta.unsqueeze(-1) * m.unsqueeze(0).unsqueeze(0)
        return torch.cat([r.unsqueeze(-1), ax.unsqueeze(-1), ay.unsqueeze(-1),
                          torch.sin(thm), torch.cos(thm)], dim=-1)

    def _global_stats(self, coords):
        mean_coords = coords.mean(dim=1)
        std_coords = coords.std(dim=1, unbiased=False).clamp_min(1e-6)
        min_coords = coords.min(dim=1)[0]; max_coords = coords.max(dim=1)[0]
        bbox_size = (max_coords - min_coords).clamp_min(1e-6)
        bbox_center = (max_coords + min_coords) / 2
        return torch.cat([mean_coords, std_coords, bbox_size, bbox_center], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        B, N, _ = coords.shape
        centers, u, u_perp = self._principal_axes(coords)
        polar_feats = self._polar_feats(coords, centers, u, u_perp)
        psi = self._cycle_pos_encoding(N, self.pos_harmonics, coords.device, coords.dtype).unsqueeze(0).expand(B, -1, -1)
        feats = torch.cat([polar_feats, psi], dim=-1)
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
    P = W * (1.0 / deg_row)
    I = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    S = 0.5 * (I + P)

    # Powers of S by squaring
    S_powers = []
    cur = S
    for _ in range(max(sct_order, 1)):
        S_powers.append(cur)
        cur = torch.bmm(cur, cur)

    # E_t = S^(2^t - 1), M_t = E_t (I - S^(2^t))
    E_list = [I]
    E = I
    for i in range(1, sct_order + 1):
        E = torch.bmm(E, S_powers[i - 1])
        E_list.append(E)

    sct_ops = []
    for t in range(sct_order):
        M_t = torch.bmm(E_list[t], (I - S_powers[t]))
        sct_ops.append(M_t)

    return {"gcn": gcn_ops, "sct": sct_ops}

# ------------------- Turbo Layer (channel gating) -------------------
class TurboSCTLayer(nn.Module):
    def __init__(self, hidden_dim, num_channels, ff_dropout=0.1, res_scale=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.res_scale = res_scale
        self.channel_logits = nn.Parameter(torch.zeros(num_channels))
        self.pre = RMSNorm(hidden_dim)
        self.post = RMSNorm(hidden_dim)
        self.proj1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drop = nn.Dropout(ff_dropout)
        nn.init.xavier_uniform_(self.proj1.weight, gain=0.8)
        nn.init.xavier_uniform_(self.proj2.weight, gain=0.8)

    def forward(self, X, diffusion_outs: List[torch.Tensor]):
        Xin = self.pre(X)
        gate = torch.softmax(self.channel_logits, dim=0)     # [C]
        stacked = torch.stack(diffusion_outs, dim=0)         # [C,B,N,H]
        mixed = torch.einsum('c,cbnh->bnh', gate, stacked)
        out = F.gelu(self.proj1(self.post(mixed)))
        out = self.proj2(out)
        out = self.drop(out)
        return X + self.res_scale * out

# ------------------- Gumbel–Sinkhorn -------------------
def fast_sinkhorn_iteration(Z, n_iter=10):
    Z = Z.to(torch.float32)
    for _ in range(n_iter):
        Z = Z - torch.logsumexp(Z, dim=-1, keepdim=True)
        Z = Z - torch.logsumexp(Z, dim=-2, keepdim=True)
    return Z

def fast_gumbel_sinkhorn(logits, tau=0.5, n_iter=10, noise_scale=0.05, clamp_val=20.0):
    logits = logits.clamp(-clamp_val, clamp_val)
    if noise_scale > 0:
        uniform = torch.rand_like(logits, dtype=torch.float32)
        g = -torch.log(-torch.log(uniform + 1e-20) + 1e-20) * noise_scale
    else:
        g = torch.zeros_like(logits, dtype=torch.float32)
    Z_init = (logits.to(torch.float32) + g) / max(tau, 1e-3)
    Z_init = Z_init.clamp(-clamp_val, clamp_val)
    Z = fast_sinkhorn_iteration(Z_init, n_iter)
    P = F.softmax(Z, dim=-1)
    P = P / P.sum(dim=-2, keepdim=True).clamp_min(1e-6)
    return P.to(logits.dtype), Z_init.to(logits.dtype)

# ------------------- The fast model -------------------
class TurboUltraFastSCTGNN(nn.Module):
    def __init__(
        self,
        input_dim, hidden_dim, output_dim,inference_mode=False,
        n_layers=32, sct_order=2, gcn_order=1, tanh_scale=0.0,
        coord_input=True, phi_harmonics=4, pos_harmonics=3, include_global_stats=True,
        tau=0.5, n_iter=10, noise_scale=0.05, logit_clamp=20.0,
        amp_autocast=False, netdropout=0.1
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
        self.netdropout = netdropout

        if coord_input:
            assert input_dim == 2
            self.coord_extractor = CoordinateFeatureExtractor(phi_harmonics, pos_harmonics, include_global_stats)
            actual_in = self.coord_extractor.total_feat_dim
        else:
            self.coord_extractor = None
            actual_in = input_dim

        self.in_norm = RMSNorm(actual_in)
        self.in_proj = nn.Linear(actual_in, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.8)

        C = sct_order + max(gcn_order, 0)
        res_scale = 1.0 / math.sqrt(max(n_layers, 1))
        self.layers = nn.ModuleList([TurboSCTLayer(hidden_dim, C,  self.netdropout, res_scale=res_scale) for _ in range(n_layers)])

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
        outs = [torch.bmm(Aop, Xin) for Aop in ops["gcn"]]
        for Mop in ops["sct"]:
            f = torch.bmm(Mop, Xin)
            if moment != 1.0:
                f = torch.abs(f).pow(moment)
            outs.append(f)
        return [F.gelu(o) for o in outs]


    def forward(self, X, adj, moment: float = 1.0, cache: Optional[Dict]=None, inference: bool=False):
        # choose the right autocast context
        if self.amp_autocast and X.is_cuda:
            ac = torch.amp.autocast('cuda')   # NEW: torch.cuda.amp.autocast is deprecated
        elif self.amp_autocast and not X.is_cuda:
            ac = torch.amp.autocast('cpu')    # optional: enable CPU autocast if you like
        else:
            ac = nullcontext()
    
        with ac:
            if cache is not None and "ops" in cache:
                ops = cache["ops"]
            else:
                ops = precompute_graph_operators(adj, self.gcn_order, self.sct_order)
                if cache is not None:
                    cache["ops"] = ops
    
            Xin = self._encode_inputs(X)
            states = [Xin]
            for layer in self.layers:
                feats = self._make_diffusion_features(Xin, ops, moment=moment)
                Xin = layer(Xin, feats)
                states.append(Xin)
    
            H = torch.cat(states, dim=-1)
            H = self.final_norm(H)
            h = F.gelu(self.out1(H))
            logits = self.out2(h) * self.logit_scale
            #logits = (self.out2(h) * self.logit_scale).clamp(-self.logit_clamp, self.logit_clamp)
            # optional tanh squeezing (default 0, i.e., disabled)
            if self.tanh_scale and self.tanh_scale > 0:
                logits = torch.tanh(logits) * self.tanh_scale
            # clamp to avoid softmax/sinkhorn overflow
            logits = logits.clamp(-self.logit_clamp, self.logit_clamp)
        if self.training and not self.inference:
            P, Z = fast_gumbel_sinkhorn(logits, tau=self.tau, n_iter=self.n_iter,
                                        noise_scale=self.noise_scale, clamp_val=self.logit_clamp)
            return P, Z
        else:
            return logits

#    def forward(self, X, adj, moment: float = 1.0, cache: Optional[Dict]=None, inference_mode: bool=False):
#        ctx = torch.cuda.amp.autocast if self.amp_autocast else torch.cpu.amp.autocast
#        with ctx(enabled=self.amp_autocast):
#            if cache is not None and "ops" in cache:
#                ops = cache["ops"]
#            else:
#                ops = precompute_graph_operators(adj, self.gcn_order, self.sct_order)
#                if cache is not None:
#                    cache["ops"] = ops
#
#            Xin = self._encode_inputs(X)
#            states = [Xin]
#            for layer in self.layers:
#                feats = self._make_diffusion_features(Xin, ops, moment=moment)
#                Xin = layer(Xin, feats)
#                states.append(Xin)
#
#            H = torch.cat(states, dim=-1)
#            H = self.final_norm(H)
#            h = F.gelu(self.out1(H))
#            logits = (self.out2(h) * self.logit_scale).clamp(-self.logit_clamp, self.logit_clamp)
#
#        if self.training and not inference_mode:
#            P, Z = fast_gumbel_sinkhorn(logits, tau=self.tau, n_iter=self.n_iter,
#                                        noise_scale=self.noise_scale, clamp_val=self.logit_clamp)
#            return P, Z
#        else:
#            return logits

# ------------------- Factory -------------------
def create_turbo_ultra_fast_sct_gnn(**kwargs):
    return TurboUltraFastSCTGNN(**kwargs)

# ------------------- Main: keep Sinkhorn + equivariance test -------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, N, Fin = 2, 500, 2
    H, L = 64, 32
    sct_order, gcn_order = 2, 1
    Fout = N  # permutation-sized output (typical for matching)

    # Random coords/features and symmetric adjacency
    X = torch.randn(B, N, Fin, device=device)
    A = torch.rand(B, N, N, device=device)
    A = 0.5 * (A + A.transpose(-1, -2))  # make symmetric

    model = create_turbo_ultra_fast_sct_gnn(
        input_dim=Fin, hidden_dim=H, output_dim=Fout,
        n_layers=L, sct_order=sct_order, gcn_order=gcn_order,
        coord_input=False, tau=0.5, n_iter=10, noise_scale=0.05,
        logit_clamp=20.0, amp_autocast=True
    ).to(device)

    # ---------- (1) LOGIT equivariance: P f(X,A) ?= f(PX, PAP^T) ----------
    with torch.no_grad():
        # Build a single permutation P (shared across batch for clarity)
        perm = torch.randperm(N, device=device)
        P = torch.zeros(N, N, device=device)
        P[torch.arange(N, device=device), perm] = 1.0
        Pb = P.unsqueeze(0).expand(B, -1, -1)  # [B,N,N]

        Xp = torch.bmm(Pb, X)                          # X' = P X
        Ap = torch.bmm(torch.bmm(Pb, A), Pb.transpose(-1, -2))  # A' = P A P^T

        model.eval()
        logits = model(X, A, inference=True)           # [B,N,N] or [B,N,Fout]
        logits_p = model(Xp, Ap, inference=True)

        left = torch.bmm(Pb.to(logits.dtype), logits)   # cast P to match logits dtype
        logit_err = (left - logits_p).abs().max().item()
        print(f"[LOGITS] Max |P f(X) - f(PX)|: {logit_err:.3e}")

    # ---------- (2) SINKHORN equivariance (deterministic): set noise_scale=0 ----------
    model.train()  # enable Sinkhorn branch
    old_noise = model.noise_scale
    model.noise_scale = 0.0  # make deterministic for the test
    with torch.no_grad():
        P_out, _ = model(X, A, inference=False)        # [B,N,N] doubly-stochastic
        P_out_p, _ = model(Xp, Ap, inference=False)

        leftP = torch.bmm(Pb.to(P_out.dtype), P_out)     # cast P to match P_out dtype
        sinkhorn_err = (leftP - P_out_p).abs().max().item()
        print(f"[SINKHORN(noise=0)] Max |P f(X) - f(PX)|: {sinkhorn_err:.3e}")

    # restore original noise scale for normal training
    model.noise_scale = old_noise

    # quick sanity: row/col sums ~ 1
    rsum = P_out.sum(-1).mean().item()
    csum = P_out.sum(-2).mean().item()
    print(f"Doubly-stochastic check: rows≈{rsum:.4f}, cols≈{csum:.4f}")

    print("✅ Equivariance tests complete.")


